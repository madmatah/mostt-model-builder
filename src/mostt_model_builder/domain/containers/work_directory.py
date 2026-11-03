import re
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import PurePosixPath

from mostt_model_builder.domain.sources.source_pins import SourcePin

WORK_DIRECTORY_GRAMMAR = re.compile(r"/[A-Za-z0-9._/-]+")
WORK_DIRECTORY_MODE = 0o700
SOURCES_DIRECTORY_NAME = "sources"


class WorkDirectoryError(ValueError):
    pass


class EntryKind(Enum):
    MISSING = auto()
    LINK = auto()
    DIRECTORY = auto()
    OTHER = auto()


@dataclass(frozen=True, slots=True)
class DirectoryFacts:
    kind: EntryKind
    canonical_path: str
    owner_id: int
    permission_bits: int


@dataclass(frozen=True, slots=True)
class WorkDirectory:
    path: str


def refuse_malformed_work_directory_path(path: str) -> None:
    if not WORK_DIRECTORY_GRAMMAR.fullmatch(path):
        raise WorkDirectoryError(
            f"the work directory {path!a} does not match {WORK_DIRECTORY_GRAMMAR.pattern}"
        )


def require_absolute_home(home: str | None, origin: str) -> str:
    if not home:
        raise WorkDirectoryError(f"{origin} is missing, so the home directory cannot be avoided")
    if not PurePosixPath(home).is_absolute():
        raise WorkDirectoryError(f"{origin} is {home!a}, which is not an absolute path")
    return home


def judge_work_directory(
    path: str,
    facts: DirectoryFacts,
    user_id: int,
    canonical_homes: Sequence[str],
    canonical_clone_root: str,
) -> WorkDirectory:
    refuse_malformed_work_directory_path(path)
    refuse_other_kinds(path, facts.kind)
    if facts.canonical_path != path:
        raise WorkDirectoryError(
            f"the work directory {path} is not canonical: it resolves to {facts.canonical_path!a}"
        )
    if facts.owner_id != user_id:
        raise WorkDirectoryError(
            f"the work directory {path} is owned by the user id {facts.owner_id}, not by {user_id}"
        )
    if facts.permission_bits != WORK_DIRECTORY_MODE:
        raise WorkDirectoryError(
            f"the work directory {path} has the mode {facts.permission_bits:04o}, "
            f"not {WORK_DIRECTORY_MODE:04o}"
        )
    for home in canonical_homes:
        refuse_home_overlap(PurePosixPath(path), PurePosixPath(home))
    refuse_clone_overlap(PurePosixPath(path), PurePosixPath(canonical_clone_root))
    return WorkDirectory(path)


def refuse_other_kinds(path: str, kind: EntryKind) -> None:
    match kind:
        case EntryKind.LINK:
            raise WorkDirectoryError(f"the work directory {path} is a link")
        case EntryKind.MISSING:
            raise WorkDirectoryError(f"the work directory {path} does not exist")
        case EntryKind.OTHER:
            raise WorkDirectoryError(f"the work directory {path} is not a directory")
        case EntryKind.DIRECTORY:
            return


def refuse_home_overlap(work_directory: PurePosixPath, home: PurePosixPath) -> None:
    if work_directory == home:
        raise WorkDirectoryError(f"the work directory {work_directory} is the home directory")
    if work_directory.is_relative_to(home):
        raise WorkDirectoryError(
            f"the work directory {work_directory} lies under the home directory {str(home)!a}"
        )
    if home.is_relative_to(work_directory):
        raise WorkDirectoryError(
            f"the work directory {work_directory} holds the home directory {str(home)!a}"
        )


def refuse_clone_overlap(work_directory: PurePosixPath, clone_root: PurePosixPath) -> None:
    if work_directory.is_relative_to(clone_root):
        raise WorkDirectoryError(
            f"the work directory {work_directory} lies in the clone {str(clone_root)!a}"
        )
    if clone_root.is_relative_to(work_directory):
        raise WorkDirectoryError(
            f"the work directory {work_directory} holds the clone {str(clone_root)!a}"
        )


def locate_source(pin: SourcePin) -> PurePosixPath:
    return PurePosixPath(SOURCES_DIRECTORY_NAME, *pin.layout_segments)
