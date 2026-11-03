import re
import tomllib
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass

from mostt_model_builder.domain.lock.declared_requirements import (
    DeclaredRequirement,
    DeclaredRequirementError,
    read_declared_requirements,
)

# The form uv records for the manifest's one environment, Linux on x86-64.
SUPPORTED_MARKERS = ["platform_machine == 'x86_64' and sys_platform == 'linux'"]
ROOT_KEYS = frozenset({"name", "version", "source", "optional-dependencies", "metadata"})
ROOT_METADATA_KEYS = frozenset({"requires-dist", "provides-extras"})
REQUIRES_DIST_KEYS = frozenset({"name", "extras", "marker", "specifier"})
EXTRA_MARKER_GRAMMAR = re.compile(r"extra == '(?P<extra>[a-z0-9]+(-[a-z0-9]+)*)'")
SPECIFIER_GRAMMAR = re.compile(r"==(?P<version>[0-9]+(\.[0-9]+)*)")


class LockFreshnessError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class FreshnessInputs:
    project: Mapping[str, object]
    uv_settings: Mapping[str, object]
    lock: Mapping[str, object]
    root: Mapping[str, object]


def check_lock_freshness(manifest_text: str, lock_text: str, root_name: str) -> tuple[str, ...]:
    inputs = read_freshness_inputs(manifest_text, lock_text, root_name)
    problems = [
        *compare_recorded(
            "requires-python",
            inputs.lock.get("requires-python"),
            inputs.project.get("requires-python"),
        ),
        *compare_recorded(
            "supported-markers", inputs.lock.get("supported-markers"), SUPPORTED_MARKERS
        ),
        *find_root_problems(inputs),
        *compare_recorded(
            "dependency-metadata",
            normalize_dependency_metadata(
                read_table(inputs.lock, "manifest").get("dependency-metadata")
            ),
            normalize_dependency_metadata(inputs.uv_settings.get("dependency-metadata")),
        ),
        *compare_recorded(
            "conflicts",
            normalize_conflicts(inputs.lock.get("conflicts"), root_name),
            normalize_conflicts(inputs.uv_settings.get("conflicts"), None),
        ),
        *compare_recorded(
            "exclude-newer",
            read_table(inputs.lock, "options").get("exclude-newer"),
            inputs.uv_settings.get("exclude-newer"),
        ),
    ]
    if problems:
        raise LockFreshnessError(
            "the lock does not record what the manifest declares:\n"
            + "\n".join(f"- {line}" for line in problems)
        )
    return read_recorded_extras(inputs.root)


def read_recorded_extras(root: Mapping[str, object]) -> tuple[str, ...]:
    provided_extras = read_table(root, "metadata").get("provides-extras", [])
    if not isinstance(provided_extras, list) or not all(
        isinstance(extra, str) for extra in provided_extras
    ):
        raise LockFreshnessError("the root's provides-extras is not an array of strings")
    return tuple(sorted(provided_extras))


def read_declared_extras(project: Mapping[str, object]) -> list[str]:
    optional_dependencies = project.get("optional-dependencies", {})
    return sorted(optional_dependencies) if isinstance(optional_dependencies, dict) else []


def read_freshness_inputs(manifest_text: str, lock_text: str, root_name: str) -> FreshnessInputs:
    try:
        manifest = tomllib.loads(manifest_text)
        lock = tomllib.loads(lock_text)
    except (tomllib.TOMLDecodeError, RecursionError) as error:
        raise LockFreshnessError(f"the manifest or the lock is not valid TOML: {error}") from error
    tool = read_table(manifest, "tool")
    packages = lock.get("package", [])
    if not isinstance(packages, list):
        raise LockFreshnessError("the lock's packages are not an array of tables")
    roots = [
        entry for entry in packages if isinstance(entry, dict) and entry.get("name") == root_name
    ]
    if len(roots) != 1:
        raise LockFreshnessError(
            f"the lock holds the root {root_name} {len(roots)} times, not once"
        )
    return FreshnessInputs(
        project=read_table(manifest, "project"),
        uv_settings=read_table(tool, "uv"),
        lock=lock,
        root=roots[0],
    )


def read_table(table: Mapping[str, object], key: str) -> Mapping[str, object]:
    nested = table.get(key, {})
    if not isinstance(nested, dict):
        raise LockFreshnessError(f"{key} is not a table")
    return nested


def compare_recorded(key: str, recorded: object, declared: object) -> list[str]:
    if recorded != declared:
        return [f"{key}: the lock records {recorded!a}, the manifest declares {declared!a}"]
    return []


def find_root_problems(inputs: FreshnessInputs) -> list[str]:
    unknown_keys = sorted(set(inputs.root) - ROOT_KEYS)
    if unknown_keys:
        return [f"the root holds the unknown key {unknown_keys[0]!a}"]
    metadata = read_table(inputs.root, "metadata")
    unknown_metadata_keys = sorted(set(metadata) - ROOT_METADATA_KEYS)
    if unknown_metadata_keys:
        return [f"the root's metadata holds the unknown key {unknown_metadata_keys[0]!a}"]
    try:
        declared = read_declared_requirements(inputs.project.get("optional-dependencies"))
    except DeclaredRequirementError as error:
        return [str(error)]
    declared_extras = read_declared_extras(inputs.project)
    provided_extras = metadata.get("provides-extras", [])
    problems = compare_recorded(
        "provides-extras",
        sorted(provided_extras) if isinstance(provided_extras, list) else provided_extras,
        declared_extras,
    )
    locked = read_locked_requirements(metadata.get("requires-dist", []), problems)
    problems.extend(
        compare_recorded(
            "requires-dist",
            sorted(locked.elements(), key=repr),
            sorted(Counter(declared).elements(), key=repr),
        )
    )
    return problems


def read_locked_requirements(entries: object, problems: list[str]) -> Counter[DeclaredRequirement]:
    locked: Counter[DeclaredRequirement] = Counter()
    if not isinstance(entries, list):
        problems.append("the root's requires-dist is not an array of tables")
        return locked
    for entry in entries:
        requirement = read_locked_requirement(entry)
        if requirement is None:
            problems.append(f"the root's requires-dist holds {entry!a}, which no extra declares")
        else:
            locked[requirement] += 1
    return locked


def read_locked_requirement(entry: object) -> DeclaredRequirement | None:
    if not isinstance(entry, dict) or not set(entry) <= REQUIRES_DIST_KEYS:
        return None
    name, extras, marker, specifier = (
        entry.get("name"),
        entry.get("extras", []),
        entry.get("marker"),
        entry.get("specifier"),
    )
    marker_match = EXTRA_MARKER_GRAMMAR.fullmatch(marker) if isinstance(marker, str) else None
    specifier_match = SPECIFIER_GRAMMAR.fullmatch(specifier) if isinstance(specifier, str) else None
    if (
        not isinstance(name, str)
        or not isinstance(extras, list)
        or not all(isinstance(extra, str) for extra in extras)
        or marker_match is None
        or specifier_match is None
    ):
        return None
    return DeclaredRequirement(
        extra=marker_match["extra"],
        name=name,
        extras=tuple(sorted(extras)),
        version=specifier_match["version"],
    )


def normalize_dependency_metadata(entries: object) -> object:
    if entries is None:
        return []
    if not isinstance(entries, list) or not all(isinstance(entry, dict) for entry in entries):
        return entries
    return sorted(
        (str(entry.get("name")), str(entry.get("version")), repr(entry.get("requires-dist", [])))
        for entry in entries
    )


def normalize_conflicts(groups: object, root_name: str | None) -> object:
    if groups is None:
        return []
    if not isinstance(groups, list) or not all(isinstance(group, list) for group in groups):
        return groups
    normalized: list[list[str]] = []
    for group in groups:
        extras: list[str] = []
        for member in group:
            if not isinstance(member, dict) or member.get("package", root_name) != root_name:
                return groups
            extras.append(str(member.get("extra")))
        normalized.append(sorted(extras))
    return sorted(normalized)
