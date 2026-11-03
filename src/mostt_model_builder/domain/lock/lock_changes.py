from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

from mostt_model_builder.domain.lock.lock_check import LockedPackage

VERSION_SEPARATOR = ", "


@dataclass(frozen=True, slots=True)
class VersionChange:
    name: str
    previous_versions: tuple[str, ...]
    current_versions: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class LockChanges:
    added: tuple[LockedPackage, ...]
    removed: tuple[LockedPackage, ...]
    changed: tuple[VersionChange, ...]


def group_versions(packages: Sequence[LockedPackage]) -> dict[str, tuple[str, ...]]:
    versions: defaultdict[str, list[str]] = defaultdict(list)
    for package in packages:
        versions[package.name].append(package.version)
    return {name: tuple(sorted(package_versions)) for name, package_versions in versions.items()}


def compare_locks(
    previous: Sequence[LockedPackage] | None, current: Sequence[LockedPackage]
) -> LockChanges:
    previous_versions = group_versions(previous or ())
    current_versions = group_versions(current)
    return LockChanges(
        added=tuple(
            LockedPackage(name, version)
            for name in sorted(current_versions.keys() - previous_versions.keys())
            for version in current_versions[name]
        ),
        removed=tuple(
            LockedPackage(name, version)
            for name in sorted(previous_versions.keys() - current_versions.keys())
            for version in previous_versions[name]
        ),
        changed=tuple(
            VersionChange(name, previous_versions[name], current_versions[name])
            for name in sorted(previous_versions.keys() & current_versions.keys())
            if previous_versions[name] != current_versions[name]
        ),
    )


def describe_lock_changes(changes: LockChanges) -> list[str]:
    if not (changes.added or changes.removed or changes.changed):
        return ["no package is added, removed or changed in version"]
    return [
        *(f"added {package.name} {package.version}" for package in changes.added),
        *(f"removed {package.name} {package.version}" for package in changes.removed),
        *(
            f"changed {change.name} from {VERSION_SEPARATOR.join(change.previous_versions)} "
            f"to {VERSION_SEPARATOR.join(change.current_versions)}"
            for change in changes.changed
        ),
    ]
