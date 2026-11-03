from collections.abc import Callable
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Protocol

from mostt_model_builder.application.run_step import (
    ContainerEngine,
    HostIdentity,
    StepFiles,
    StepRunRequest,
    run_step,
)
from mostt_model_builder.domain.containers.container_steps import LOCK_ROW
from mostt_model_builder.domain.containers.printable_output import show_printable
from mostt_model_builder.domain.containers.step_peaks import StepPeaks
from mostt_model_builder.domain.containers.work_directory import (
    LOCK_DIRECTORY_NAME,
    find_next_lock_number,
    place_lock,
)
from mostt_model_builder.domain.lock.lock_changes import compare_locks, describe_lock_changes
from mostt_model_builder.domain.lock.lock_check import (
    IMAGES_ROOT_NAME,
    LOCK_FILE_NAME,
    MANIFEST_FILE_NAME,
    CheckedLock,
    LockError,
    check_lock,
)
from mostt_model_builder.domain.lock.lock_freshness import check_lock_freshness

TEXT_ENCODING = "utf-8"


class LockFiles(StepFiles, Protocol):
    def list_entry_names(self, directory: PurePosixPath) -> tuple[str, ...]: ...

    def replace_file(self, path: PurePosixPath, content: bytes) -> None: ...


@dataclass(frozen=True, slots=True)
class LockRequest:
    manifest_text: str
    committed_lock_text: str | None
    images_lock_path: PurePosixPath
    work_root: PurePosixPath
    head_tree: str
    host: HostIdentity


@dataclass(frozen=True, slots=True)
class LockResolution:
    lock_run: PurePosixPath
    checked_lock: CheckedLock
    recorded_extras: tuple[str, ...]
    change_lines: tuple[str, ...]
    peaks: StepPeaks


def resolve_lock(
    request: LockRequest,
    engine: ContainerEngine,
    files: LockFiles,
    show_change: Callable[[str], None],
) -> LockResolution:
    lock_directory = PurePosixPath(LOCK_DIRECTORY_NAME)
    files.make_directories(request.work_root, lock_directory)
    lock_number = find_next_lock_number(files.list_entry_names(request.work_root / lock_directory))
    placement = place_lock(lock_number)
    request_files = {MANIFEST_FILE_NAME: request.manifest_text.encode(TEXT_ENCODING)}
    if request.committed_lock_text is not None:
        request_files[LOCK_FILE_NAME] = request.committed_lock_text.encode(TEXT_ENCODING)
    outputs = run_step(
        StepRunRequest(
            row=LOCK_ROW,
            placement=placement,
            request_files=request_files,
            source_paths={},
            work_root=request.work_root,
            head_tree=request.head_tree,
            host=request.host,
            parsed_output_names=frozenset({LOCK_FILE_NAME}),
        ),
        engine,
        files,
    )
    new_lock_content = outputs.contents[LOCK_FILE_NAME]
    new_lock_text = decode_lock(new_lock_content)
    checked_lock = check_lock(new_lock_text, IMAGES_ROOT_NAME)
    recorded_extras = check_lock_freshness(request.manifest_text, new_lock_text, IMAGES_ROOT_NAME)
    previous_packages = (
        None
        if request.committed_lock_text is None
        else check_lock(request.committed_lock_text, IMAGES_ROOT_NAME).packages
    )
    changes = compare_locks(previous_packages, checked_lock.packages)
    change_lines = tuple(
        show_printable(line.encode(TEXT_ENCODING)) for line in describe_lock_changes(changes)
    )
    for line in change_lines:
        show_change(line)
    files.replace_file(request.images_lock_path, new_lock_content)
    return LockResolution(
        lock_run=placement.output_directory.parent,
        checked_lock=checked_lock,
        recorded_extras=recorded_extras,
        change_lines=change_lines,
        peaks=outputs.peaks,
    )


def decode_lock(content: bytes) -> str:
    try:
        return content.decode(TEXT_ENCODING)
    except UnicodeDecodeError as error:
        raise LockError(f"the new lock is not UTF-8: {error}") from error
