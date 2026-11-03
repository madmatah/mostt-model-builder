import re
from collections.abc import Mapping
from dataclasses import dataclass

NORMALIZED_NAME = r"[a-z0-9]+(-[a-z0-9]+)*"
NORMALIZED_NAME_GRAMMAR = re.compile(NORMALIZED_NAME)
VERSION = r"[0-9]+(\.[0-9]+)*"
REQUIREMENT_GRAMMAR = re.compile(
    rf"(?P<name>{NORMALIZED_NAME})"
    rf"(\[(?P<extras>{NORMALIZED_NAME}(,{NORMALIZED_NAME})*)\])?"
    rf"==(?P<version>{VERSION})"
)
EXTRAS_SEPARATOR = ","


class DeclaredRequirementError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class DeclaredRequirement:
    extra: str
    name: str
    extras: tuple[str, ...]
    version: str


def parse_declared_requirement(text: str, extra: str) -> DeclaredRequirement:
    match = REQUIREMENT_GRAMMAR.fullmatch(text)
    if match is None:
        raise DeclaredRequirementError(
            f"the requirement {text!a} of the extra {extra!a} is not "
            "<name>==<version> or <name>[<extra>,…]==<version>, with a normalized name"
        )
    extras = match["extras"]
    return DeclaredRequirement(
        extra=extra,
        name=match["name"],
        extras=tuple(sorted(extras.split(EXTRAS_SEPARATOR))) if extras else (),
        version=match["version"],
    )


def read_declared_requirements(optional_dependencies: object) -> tuple[DeclaredRequirement, ...]:
    if optional_dependencies is None:
        return ()
    if not isinstance(optional_dependencies, Mapping):
        raise DeclaredRequirementError("the optional dependencies are not a table")
    requirements: list[DeclaredRequirement] = []
    for extra, entries in optional_dependencies.items():
        if not isinstance(extra, str) or not NORMALIZED_NAME_GRAMMAR.fullmatch(extra):
            raise DeclaredRequirementError(f"the extra {extra!a} is not a normalized name")
        if not isinstance(entries, list) or not all(isinstance(entry, str) for entry in entries):
            raise DeclaredRequirementError(f"the extra {extra} is not a list of requirements")
        requirements.extend(parse_declared_requirement(entry, extra) for entry in entries)
    return tuple(requirements)
