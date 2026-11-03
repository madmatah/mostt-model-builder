import re
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass

MANIFEST_FILE_NAME = "pyproject.toml"
LOCK_FILE_NAME = "uv.lock"
LOCK_MAX_SIZE_BYTES = 16 * 2**20
LOCK_VERSION = 1
HOST_ROOT_NAME = "mostt-model-builder"
IMAGES_ROOT_NAME = "mostt-model-builder-images"
REGISTRY_SOURCE = {"registry": "https://pypi.org/simple"}
ROOT_SOURCE = {"virtual": "."}
DISTRIBUTION_URL_PREFIX = "https://files.pythonhosted.org/"
HASH_GRAMMAR = re.compile(r"sha256:[0-9a-f]{64}")
PACKAGE_NAME_GRAMMAR = re.compile(r"[a-z0-9]+([._-][a-z0-9]+)*")
VERSION_GRAMMAR = re.compile(r"[A-Za-z0-9.+!_-]{1,64}")
WHEEL_SUFFIX = ".whl"
TAG_SEPARATOR = "."
NAME_SEPARATOR = "-"
SUPPORTED_PYTHON_MINOR = 12
# The images run Python 3.12 on Debian bookworm for x86-64.
COMPLETE_PYTHON_TAGS = frozenset({"cp312", "py3"})
STABLE_ABI_PYTHON_TAG_GRAMMAR = re.compile(r"cp3([0-9]{1,2})")
STABLE_ABI_TAG = "abi3"
PURE_PYTHON_TAG_GRAMMAR = re.compile(r"py3([0-9]{1,2})")
PURE_PYTHON_ABI_TAG = "none"
PLATFORM_TAGS_WITHOUT_A_GLIBC = frozenset({"any", "linux_x86_64"})
MANYLINUX_TAG_GRAMMAR = re.compile(r"manylinux(?:(1|2010|2014)|_([0-9]{1,3})_([0-9]{1,3}))_x86_64")
# PEP 600 reads the legacy manylinux tags as these glibc versions.
LEGACY_MANYLINUX_GLIBC_VERSIONS = {"1": (2, 5), "2010": (2, 12), "2014": (2, 17)}
# The glibc of Debian bookworm, the base of the images. CI's runner and the host run newer ones,
# so the ceiling holds for the host's lock too; no x86-64 wheel of either lock goes above it.
MAX_GLIBC_VERSION = (2, 36)


class LockError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class LockedPackage:
    name: str
    version: str


@dataclass(frozen=True, slots=True)
class CheckedLock:
    root_name: str
    packages: tuple[LockedPackage, ...]


def load_lock(text: str) -> dict[str, object]:
    try:
        return tomllib.loads(text)
    except (tomllib.TOMLDecodeError, RecursionError) as error:
        raise LockError(f"the lock is not valid TOML: {error}") from error


def check_lock(text: str, root_name: str) -> CheckedLock:
    document = load_lock(text)
    if document.get("version") != LOCK_VERSION:
        raise LockError(
            f"the lock's version is {document.get('version')!a}, not {LOCK_VERSION}, "
            "the layout this check knows"
        )
    entries = document.get("package")
    if not isinstance(entries, list) or not all(isinstance(entry, dict) for entry in entries):
        raise LockError("the lock's packages are not an array of tables")
    exempt_names = find_dependency_metadata_names(document)
    problems: list[str] = []
    packages: list[LockedPackage] = []
    root_count = 0
    for entry in entries:
        package = read_locked_package(entry)
        if package.name == root_name:
            root_count += 1
            if entry.get("source") != ROOT_SOURCE:
                problems.append(f"the root {root_name} is not the virtual project '.'")
            continue
        packages.append(package)
        problems.extend(find_package_problems(entry, package, package.name in exempt_names))
    if root_count != 1:
        problems.append(f"the lock holds the root {root_name} {root_count} times, not once")
    if problems:
        raise LockError("the lock is refused:\n" + "\n".join(f"- {line}" for line in problems))
    return CheckedLock(root_name, tuple(packages))


def find_dependency_metadata_names(document: Mapping[str, object]) -> frozenset[str]:
    manifest = document.get("manifest", {})
    if not isinstance(manifest, dict):
        raise LockError("the lock's manifest is not a table")
    entries = manifest.get("dependency-metadata", [])
    if not isinstance(entries, list) or not all(
        isinstance(entry, dict) and isinstance(entry.get("name"), str) for entry in entries
    ):
        raise LockError("the lock's dependency metadata is not an array of named tables")
    return frozenset(str(entry["name"]) for entry in entries)


def read_locked_package(entry: Mapping[str, object]) -> LockedPackage:
    name = entry.get("name")
    version = entry.get("version")
    if not isinstance(name, str) or not PACKAGE_NAME_GRAMMAR.fullmatch(name):
        raise LockError(
            f"the lock holds a package named {name!a}, outside {PACKAGE_NAME_GRAMMAR.pattern}"
        )
    if not isinstance(version, str) or not VERSION_GRAMMAR.fullmatch(version):
        raise LockError(
            f"the package {name} has the version {version!a}, outside {VERSION_GRAMMAR.pattern}"
        )
    return LockedPackage(name, version)


def find_package_problems(
    entry: Mapping[str, object], package: LockedPackage, is_built_by_a_stage: bool
) -> list[str]:
    where = f"{package.name} {package.version}"
    source = entry.get("source")
    if source != REGISTRY_SOURCE:
        return [f"{where} comes from {source!a}, not from {REGISTRY_SOURCE['registry']}"]
    problems: list[str] = []
    distributions = list_distributions(entry, where, problems)
    for distribution in distributions:
        problems.extend(find_distribution_problems(distribution, where))
    wheel_names = [
        str(distribution["url"]).rsplit("/", 1)[-1]
        for distribution in distributions
        if str(distribution.get("url", "")).endswith(WHEEL_SUFFIX)
    ]
    if not is_built_by_a_stage and not any(is_installable_wheel(name) for name in wheel_names):
        problems.append(
            f"{where} has no wheel that installs on Python 3.12 for Linux x86-64: "
            "decide whether a build stage may build it"
        )
    return problems


def list_distributions(
    entry: Mapping[str, object], where: str, problems: list[str]
) -> list[Mapping[str, object]]:
    distributions: list[Mapping[str, object]] = []
    sdist = entry.get("sdist")
    if sdist is not None:
        if isinstance(sdist, dict):
            distributions.append(sdist)
        else:
            problems.append(f"{where} has an sdist that is not a table")
    wheels = entry.get("wheels", [])
    if isinstance(wheels, list) and all(isinstance(wheel, dict) for wheel in wheels):
        distributions.extend(wheels)
    else:
        problems.append(f"{where} has wheels that are not an array of tables")
    return distributions


def find_distribution_problems(distribution: Mapping[str, object], where: str) -> list[str]:
    url = distribution.get("url")
    problems: list[str] = []
    if not isinstance(url, str) or not url.startswith(DISTRIBUTION_URL_PREFIX):
        problems.append(f"{where} has a distribution at {url!a}, outside {DISTRIBUTION_URL_PREFIX}")
    file_hash = distribution.get("hash")
    if not isinstance(file_hash, str) or not HASH_GRAMMAR.fullmatch(file_hash):
        problems.append(f"{where} has a distribution without its SHA-256: {url!a}")
    return problems


def is_installable_wheel(file_name: str) -> bool:
    if not file_name.endswith(WHEEL_SUFFIX):
        return False
    parts = file_name[: -len(WHEEL_SUFFIX)].split(NAME_SEPARATOR)
    if len(parts) not in (5, 6):
        return False
    python_tags, abi_tags, platform_tags = (part.split(TAG_SEPARATOR) for part in parts[-3:])
    return any(
        is_installable_python(python_tag, abi_tag)
        for python_tag in python_tags
        for abi_tag in abi_tags
    ) and any(is_installable_platform(platform_tag) for platform_tag in platform_tags)


def is_installable_platform(platform_tag: str) -> bool:
    if platform_tag in PLATFORM_TAGS_WITHOUT_A_GLIBC:
        return True
    match = MANYLINUX_TAG_GRAMMAR.fullmatch(platform_tag)
    if match is None:
        return False
    legacy_tag, glibc_major, glibc_minor = match.groups()
    glibc_version = (
        LEGACY_MANYLINUX_GLIBC_VERSIONS[legacy_tag]
        if legacy_tag is not None
        else (int(glibc_major), int(glibc_minor))
    )
    return glibc_version <= MAX_GLIBC_VERSION


def is_installable_python(python_tag: str, abi_tag: str) -> bool:
    if python_tag in COMPLETE_PYTHON_TAGS:
        return True
    return is_minor_at_most_supported(
        STABLE_ABI_PYTHON_TAG_GRAMMAR, python_tag, abi_tag, STABLE_ABI_TAG
    ) or is_minor_at_most_supported(
        PURE_PYTHON_TAG_GRAMMAR, python_tag, abi_tag, PURE_PYTHON_ABI_TAG
    )


def is_minor_at_most_supported(
    python_tag_grammar: re.Pattern[str], python_tag: str, abi_tag: str, expected_abi_tag: str
) -> bool:
    match = python_tag_grammar.fullmatch(python_tag)
    return (
        match is not None
        and abi_tag == expected_abi_tag
        and int(match[1]) <= SUPPORTED_PYTHON_MINOR
    )
