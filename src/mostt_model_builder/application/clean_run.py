from dataclasses import dataclass
from enum import Enum, auto
from pathlib import PurePosixPath
from typing import Protocol

from mostt_model_builder.application.run_step import (
    ContainerEngine,
    HostIdentity,
    StepFiles,
    prepare_container,
    run_judged_container,
)
from mostt_model_builder.domain.containers.container_files import FileKind, judge_mount
from mostt_model_builder.domain.containers.container_steps import (
    StepRow,
    build_clean_row,
    plan_container_run,
)
from mostt_model_builder.domain.containers.work_directory import place_clean, place_step


class CleanFiles(StepFiles, Protocol):
    def remove_empty_directory(self, path: PurePosixPath) -> None: ...

    def remove_request_directory(self, path: PurePosixPath) -> None: ...

    def remove_file_if_present(self, path: PurePosixPath) -> bool: ...


class CleanOutcome(Enum):
    CLEANED = auto()
    NOTHING_TO_CLEAN = auto()


@dataclass(frozen=True, slots=True)
class CleanRequest:
    row: StepRow
    recipe_name: str
    commit: str
    work_root: PurePosixPath
    head_tree: str
    host: HostIdentity


def clean_step(request: CleanRequest, engine: ContainerEngine, files: CleanFiles) -> CleanOutcome:
    placement = place_step(request.recipe_name, request.commit, request.row.name)
    output_directory = placement.output_directory
    request_directory = placement.request_directory
    has_outputs = is_judged_directory(request, output_directory, files)
    has_request = request_directory is not None and is_judged_directory(
        request, request_directory, files
    )
    judge_directory_if_present(request, placement.log_path.parent, files)
    if has_outputs:
        empty_output_directory(request, engine, files)
        files.remove_empty_directory(request.work_root / output_directory)
    if has_request and request_directory is not None:
        files.remove_request_directory(request.work_root / request_directory)
    has_log = files.remove_file_if_present(request.work_root / placement.log_path)
    if has_outputs or has_request or has_log:
        return CleanOutcome.CLEANED
    return CleanOutcome.NOTHING_TO_CLEAN


def is_judged_directory(
    request: CleanRequest, relative_path: PurePosixPath, files: CleanFiles
) -> bool:
    components = files.inspect_components(request.work_root, relative_path)
    if components[-1].facts.kind is FileKind.MISSING:
        return False
    judge_mount(components, is_file=False, user_id=request.host.user_id)
    return True


def judge_directory_if_present(
    request: CleanRequest, relative_path: PurePosixPath, files: CleanFiles
) -> None:
    components = files.inspect_components(request.work_root, relative_path)
    if components[-1].facts.kind is not FileKind.MISSING:
        judge_mount(components, is_file=False, user_id=request.host.user_id)


def empty_output_directory(
    request: CleanRequest, engine: ContainerEngine, files: CleanFiles
) -> None:
    clean_row = build_clean_row(request.row.image)
    image_id = prepare_container(clean_row, request.head_tree, request.host, engine)
    clean_placement = place_clean(request.recipe_name, request.commit, request.row.name)
    run = plan_container_run(
        clean_row, image_id, clean_placement, {}, request.work_root, request.host.group_id, {}
    )
    run_judged_container(run, clean_placement.log_path, clean_row.name, request.host, engine, files)


# The candidate holds the host's own copies, in a directory the host made, so the host removes
# them itself, without a container.
def remove_candidate(request: CleanRequest, files: CleanFiles) -> CleanOutcome:
    candidate_directory = place_step(
        request.recipe_name, request.commit, request.row.name
    ).candidate_directory
    if candidate_directory is None or not is_judged_directory(request, candidate_directory, files):
        return CleanOutcome.NOTHING_TO_CLEAN
    files.remove_request_directory(request.work_root / candidate_directory)
    return CleanOutcome.CLEANED
