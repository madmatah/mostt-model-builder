import stat
from collections.abc import Collection, Sequence
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import PurePosixPath

# Every step leaves its outputs in this mode, readable by the user's group, which the host
# runs as, since onnx writes its external data in 0600.
WRITTEN_FILE_MODE = 0o640
SINGLE_LINK_COUNT = 1


class ContainerFileError(ValueError):
    pass


class FileKind(Enum):
    REGULAR = auto()
    DIRECTORY = auto()
    LINK = auto()
    FIFO = auto()
    SOCKET = auto()
    DEVICE = auto()
    MISSING = auto()
    UNREADABLE = auto()


@dataclass(frozen=True, slots=True)
class FileFacts:
    kind: FileKind
    owner_id: int
    permission_bits: int
    size_bytes: int
    link_count: int


@dataclass(frozen=True, slots=True)
class PathComponent:
    path: PurePosixPath
    facts: FileFacts


KIND_REFUSALS = {
    FileKind.DIRECTORY: "is a directory",
    FileKind.LINK: "is a link",
    FileKind.FIFO: "is a FIFO",
    FileKind.SOCKET: "is a socket",
    FileKind.DEVICE: "is a device",
    FileKind.MISSING: "is missing",
    FileKind.UNREADABLE: "cannot be read",
}


def judge_output_entries(entry_names: Collection[str], declared_names: Collection[str]) -> None:
    undeclared = sorted(set(entry_names) - set(declared_names))
    if undeclared:
        raise ContainerFileError(
            f"the step wrote {undeclared[0]!a}, which its row does not declare"
        )
    missing = sorted(set(declared_names) - set(entry_names))
    if missing:
        raise ContainerFileError(f"the step did not write its output {missing[0]}")


def judge_written_file(
    name: str, facts: FileFacts, max_size_bytes: int, container_user_id: int
) -> None:
    where = f"the output {name}"
    if facts.kind is not FileKind.REGULAR:
        raise ContainerFileError(f"{where} {KIND_REFUSALS[facts.kind]}, not a regular file")
    if facts.owner_id != container_user_id:
        raise ContainerFileError(
            f"{where} is owned by the user id {facts.owner_id}, not by {container_user_id}"
        )
    if facts.permission_bits & (stat.S_ISUID | stat.S_ISGID):
        raise ContainerFileError(f"{where} is setuid or setgid")
    if facts.permission_bits != WRITTEN_FILE_MODE:
        raise ContainerFileError(
            f"{where} has the mode {facts.permission_bits:04o}, not {WRITTEN_FILE_MODE:04o}"
        )
    if facts.size_bytes > max_size_bytes:
        raise ContainerFileError(
            f"{where} holds {facts.size_bytes} bytes, beyond its bound of {max_size_bytes}"
        )
    if facts.link_count != SINGLE_LINK_COUNT:
        raise ContainerFileError(f"{where} has {facts.link_count} links, not one")


def judge_mount(components: Sequence[PathComponent], is_file: bool, user_id: int) -> None:
    if not components:
        raise ContainerFileError("a mount names no path")
    *directories, leaf = components
    for directory in directories:
        refuse_foreign_directory(directory, user_id)
    if is_file:
        if leaf.facts.kind is not FileKind.REGULAR:
            raise ContainerFileError(
                f"{leaf.path} {KIND_REFUSALS[leaf.facts.kind]}, where a file is declared"
            )
        if leaf.facts.owner_id != user_id:
            raise ContainerFileError(
                f"{leaf.path} is owned by the user id {leaf.facts.owner_id}, not by {user_id}"
            )
        return
    refuse_foreign_directory(leaf, user_id)


def refuse_foreign_directory(component: PathComponent, user_id: int) -> None:
    if component.facts.kind is not FileKind.DIRECTORY:
        reason = KIND_REFUSALS.get(component.facts.kind, "is a file")
        raise ContainerFileError(f"{component.path} {reason}, where a directory is declared")
    if component.facts.owner_id != user_id:
        raise ContainerFileError(
            f"{component.path} is owned by the user id {component.facts.owner_id}, not by {user_id}"
        )
