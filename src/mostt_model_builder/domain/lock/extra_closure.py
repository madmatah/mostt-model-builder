from collections.abc import Mapping
from dataclasses import dataclass

from mostt_model_builder.domain.lock.lock_check import (
    IMAGES_ROOT_NAME,
    LockError,
    check_lock,
    load_lock,
)


@dataclass(frozen=True, slots=True)
class LockedDistribution:
    url: str
    sha256: str


@dataclass(frozen=True, slots=True)
class LockedEntry:
    name: str
    version: str
    distributions: tuple[LockedDistribution, ...]


def read_dependencies(field: object, where: str) -> list[tuple[str, tuple[str, ...]]]:
    if not isinstance(field, list):
        raise LockError(f"{where} are not an array of tables")
    dependencies = []
    for entry in field:
        if not isinstance(entry, dict) or not isinstance(entry.get("name"), str):
            raise LockError(f"{where} hold an entry without a name")
        extras = entry.get("extra", [])
        if not isinstance(extras, list) or not all(isinstance(extra, str) for extra in extras):
            raise LockError(f"{where}: {entry['name']} names extras that are not strings")
        dependencies.append((str(entry["name"]), tuple(str(extra) for extra in extras)))
    return dependencies


def read_optional_dependencies(
    entry: Mapping[str, object], extra: str, where: str
) -> list[tuple[str, tuple[str, ...]]] | None:
    optional = entry.get("optional-dependencies", {})
    if not isinstance(optional, dict):
        raise LockError(f"the optional dependencies of {where} are not a table")
    if extra not in optional:
        return None
    return read_dependencies(optional[extra], f"the dependencies of {where}[{extra}]")


def read_distributions(entry: Mapping[str, object]) -> tuple[LockedDistribution, ...]:
    sdist = entry.get("sdist")
    wheels = entry.get("wheels", [])
    tables = (
        [*([sdist] if isinstance(sdist, dict) else []), *wheels] if isinstance(wheels, list) else []
    )
    return tuple(
        LockedDistribution(str(table["url"]), str(table["hash"]))
        for table in tables
        if isinstance(table, dict)
    )


# Markers are ignored and every entry of a name is taken, which keys an image with more packages
# than it may install, never fewer: uv's own evaluation of the markers runs only in a container.
def close_extra(lock_text: str, extra: str | None) -> tuple[LockedEntry, ...]:
    if extra is None:
        return ()
    check_lock(lock_text, IMAGES_ROOT_NAME)
    entries = load_lock(lock_text)["package"]
    if not isinstance(entries, list):
        raise LockError("the lock's packages are not an array of tables")
    entries_by_name: dict[str, list[Mapping[str, object]]] = {}
    for entry in entries:
        entries_by_name.setdefault(str(entry["name"]), []).append(entry)
    (root,) = entries_by_name[IMAGES_ROOT_NAME]
    root_dependencies = read_optional_dependencies(root, extra, IMAGES_ROOT_NAME)
    if root_dependencies is None:
        raise LockError(f"the lock holds no extra {extra} of {IMAGES_ROOT_NAME}")
    pending = [
        *read_dependencies(root.get("dependencies", []), f"the dependencies of {IMAGES_ROOT_NAME}"),
        *root_dependencies,
    ]
    seen: set[tuple[str, str | None]] = set()
    closure: dict[tuple[str, str], LockedEntry] = {}
    while pending:
        name, extras = pending.pop()
        for requested in (None, *extras):
            if (name, requested) in seen:
                continue
            seen.add((name, requested))
            for entry in entries_by_name.get(name, []):
                where = f"{name} {entry['version']}"
                if requested is None:
                    closure[(name, str(entry["version"]))] = LockedEntry(
                        name, str(entry["version"]), read_distributions(entry)
                    )
                    pending.extend(
                        read_dependencies(
                            entry.get("dependencies", []), f"the dependencies of {where}"
                        )
                    )
                else:
                    pending.extend(read_optional_dependencies(entry, requested, where) or [])
    return tuple(closure[key] for key in sorted(closure))
