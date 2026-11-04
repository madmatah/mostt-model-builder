import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import PurePosixPath

from mostt_model_builder.domain.containers.container_steps import (
    CLEAN_STEP_NAME,
    LOCK_STEP_NAME,
    SourceReference,
    StepPlacement,
)
from mostt_model_builder.domain.sources.source_pins import SourcePin

WORK_DIRECTORY_GRAMMAR = re.compile(r"/[A-Za-z0-9._/-]+")
WORK_DIRECTORY_MODE = 0o700
SOURCES_DIRECTORY_NAME = "sources"
RUNS_DIRECTORY_NAME = "runs"
REQUESTS_DIRECTORY_NAME = "requests"
LOGS_DIRECTORY_NAME = "logs"
CANDIDATE_DIRECTORY_NAME = "candidate"
# The name under which a Hugging Face repository takes its model card.
MODEL_CARD_FILE_NAME = "README.md"
LOCK_DIRECTORY_NAME = "lock"
LOCK_REQUEST_DIRECTORY_NAME = "request"
LOCK_OUTPUT_DIRECTORY_NAME = "output"
IMAGE_BUILDS_DIRECTORY_NAME = "image-builds"
LOG_SUFFIX = ".log"
LOCK_NUMBER_GRAMMAR = re.compile(r"[1-9][0-9]{0,8}")


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


def locate_referenced_sources(
    references: Sequence[SourceReference], pins: Sequence[SourcePin]
) -> Mapping[SourceReference, PurePosixPath]:
    pins_by_reference = {SourceReference(pin.repository, pin.path): pin for pin in pins}
    if len(pins_by_reference) != len(pins):
        raise WorkDirectoryError("two pins give the same file of a repository at two revisions")
    missing = [reference for reference in references if reference not in pins_by_reference]
    if missing:
        raise WorkDirectoryError(
            f"no pin gives {missing[0].path} of {missing[0].repository}, which a step reads"
        )
    return {reference: locate_source(pins_by_reference[reference]) for reference in references}


def locate_run(owner: str, commit: str) -> PurePosixPath:
    return PurePosixPath(RUNS_DIRECTORY_NAME, owner, commit)


def place_step(owner: str, commit: str, step_name: str) -> StepPlacement:
    run = locate_run(owner, commit)
    return StepPlacement(
        output_directory=run / step_name,
        request_directory=run / REQUESTS_DIRECTORY_NAME / step_name,
        log_path=run / LOGS_DIRECTORY_NAME / f"{step_name}{LOG_SUFFIX}",
        candidate_directory=run / CANDIDATE_DIRECTORY_NAME,
    )


def place_clean(owner: str, commit: str, step_name: str) -> StepPlacement:
    run = locate_run(owner, commit)
    return StepPlacement(
        output_directory=run / step_name,
        request_directory=None,
        log_path=run / LOGS_DIRECTORY_NAME / f"{CLEAN_STEP_NAME}-{step_name}{LOG_SUFFIX}",
    )


def find_next_lock_number(entry_names: Sequence[str]) -> int:
    numbers = [int(name) for name in entry_names if LOCK_NUMBER_GRAMMAR.fullmatch(name)]
    return max(numbers, default=0) + 1


def place_lock(lock_number: int) -> StepPlacement:
    lock_run = PurePosixPath(LOCK_DIRECTORY_NAME, str(lock_number))
    return StepPlacement(
        output_directory=lock_run / LOCK_OUTPUT_DIRECTORY_NAME,
        request_directory=lock_run / LOCK_REQUEST_DIRECTORY_NAME,
        log_path=lock_run / LOGS_DIRECTORY_NAME / f"{LOCK_STEP_NAME}{LOG_SUFFIX}",
    )


def locate_image_build_log(image_tag: str) -> PurePosixPath:
    return PurePosixPath(IMAGE_BUILDS_DIRECTORY_NAME, f"{image_tag}{LOG_SUFFIX}")
