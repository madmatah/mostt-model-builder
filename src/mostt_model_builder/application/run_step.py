from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import Protocol

from mostt_model_builder.domain.containers.account_ranges import (
    HostAccounts,
    refuse_held_container_user,
)
from mostt_model_builder.domain.containers.container_files import (
    FileFacts,
    PathComponent,
    judge_mount,
    judge_output_entries,
    judge_written_file,
)
from mostt_model_builder.domain.containers.container_steps import (
    CLEAN_STEP_NAME,
    CONTAINER_USER_ID,
    PEAKS_FILE_NAME,
    ContainerRun,
    SourceReference,
    StepPlacement,
    StepRow,
    judge_row,
    plan_container_run,
)
from mostt_model_builder.domain.containers.printable_output import show_printable
from mostt_model_builder.domain.containers.provenance import refuse_foreign_image
from mostt_model_builder.domain.containers.step_peaks import StepPeaks, parse_step_peaks

OUTPUT_DIRECTORY_MODE = 0o2770
REQUEST_DIRECTORY_MODE = 0o750
REQUEST_FILE_MODE = 0o640
IMAGES_COMMAND = "./model-build.sh --work <dir> --recipe <name> images"
CLEAN_COMMAND = f"./model-build.sh --work <dir> --recipe <name> {CLEAN_STEP_NAME}"


class StepError(Exception):
    pass


@dataclass(frozen=True, slots=True)
class ImageFacts:
    image_id: str
    tree_label: str | None


@dataclass(frozen=True, slots=True)
class ContainerExit:
    status: int
    log_tail: bytes


@dataclass(frozen=True, slots=True)
class WrittenFile:
    facts: FileFacts
    content: bytes | None


@dataclass(frozen=True, slots=True)
class WrittenOutputs:
    entry_names: tuple[str, ...]
    files: Mapping[str, WrittenFile]


class ContainerEngine(Protocol):
    def inspect_image(self, tag: str) -> ImageFacts | None: ...

    def run_container(self, run: ContainerRun, log_path: PurePosixPath) -> ContainerExit: ...


class StepFiles(Protocol):
    def make_directories(self, root: PurePosixPath, relative_path: PurePosixPath) -> None: ...

    def make_step_directory(self, path: PurePosixPath, mode: int) -> bool: ...

    def write_new_file(self, path: PurePosixPath, content: bytes, mode: int) -> None: ...

    def inspect_components(
        self, root: PurePosixPath, relative_path: PurePosixPath
    ) -> tuple[PathComponent, ...]: ...

    def read_written_outputs(
        self,
        directory: PurePosixPath,
        max_sizes_bytes: Mapping[str, int],
        parsed_names: frozenset[str],
    ) -> WrittenOutputs: ...

    def list_entry_names(self, directory: PurePosixPath) -> tuple[str, ...]: ...


@dataclass(frozen=True, slots=True)
class HostIdentity:
    user_id: int
    group_id: int
    cpu_count: int
    accounts: HostAccounts


@dataclass(frozen=True, slots=True)
class StepRunRequest:
    row: StepRow
    placement: StepPlacement
    request_files: Mapping[str, bytes]
    source_paths: Mapping[SourceReference, PurePosixPath]
    work_root: PurePosixPath
    head_tree: str
    host: HostIdentity
    earlier_output_directories: Mapping[str, PurePosixPath] = field(default_factory=dict)
    parsed_output_names: frozenset[str] = frozenset()


@dataclass(frozen=True, slots=True)
class StepOutputs:
    peaks: StepPeaks
    contents: Mapping[str, bytes]
    sizes_bytes: Mapping[str, int]


def run_step(request: StepRunRequest, engine: ContainerEngine, files: StepFiles) -> StepOutputs:
    row = request.row
    parsed_names = find_parsed_names(row, request.parsed_output_names)
    image_id = prepare_container(row, request.head_tree, request.host, engine)
    make_output_directory(request, files)
    write_request(request, files)
    run = plan_container_run(
        row,
        image_id,
        request.placement,
        request.source_paths,
        request.work_root,
        request.host.group_id,
        request.earlier_output_directories,
    )
    judge_earlier_outputs(request, files)
    run_judged_container(run, request.placement.log_path, row.name, request.host, engine, files)
    return read_outputs(request, parsed_names, files)


def prepare_container(
    row: StepRow, head_tree: str, host: HostIdentity, engine: ContainerEngine
) -> str:
    judge_row(row, host.cpu_count)
    refuse_held_container_user(host.accounts, CONTAINER_USER_ID)
    image_facts = engine.inspect_image(row.image.tag)
    if image_facts is None:
        raise StepError(f"the image {row.image.tag} is missing: build it with {IMAGES_COMMAND}")
    refuse_foreign_image(row.image.tag, image_facts.tree_label, head_tree)
    return image_facts.image_id


def make_output_directory(request: StepRunRequest, files: StepFiles) -> None:
    output_directory = request.placement.output_directory
    files.make_directories(request.work_root, output_directory.parent)
    if not files.make_step_directory(request.work_root / output_directory, OUTPUT_DIRECTORY_MODE):
        raise StepError(
            f"the outputs of the step {request.row.name} already exist in {output_directory}: "
            f"remove them with {CLEAN_COMMAND} {request.row.name}"
        )


def write_request(request: StepRunRequest, files: StepFiles) -> None:
    request_directory = request.placement.request_directory
    if request_directory is None:
        return
    files.make_directories(request.work_root, request_directory.parent)
    if not files.make_step_directory(request.work_root / request_directory, REQUEST_DIRECTORY_MODE):
        raise StepError(
            f"the request of the step {request.row.name} already exists in {request_directory}: "
            f"remove it with {CLEAN_COMMAND} {request.row.name}"
        )
    for name, content in request.request_files.items():
        files.write_new_file(
            request.work_root / request_directory / name, content, REQUEST_FILE_MODE
        )


def run_judged_container(
    run: ContainerRun,
    log_path: PurePosixPath,
    step_name: str,
    host: HostIdentity,
    engine: ContainerEngine,
    files: StepFiles,
) -> None:
    files.make_directories(run.work_root, log_path.parent)
    judge_mounts(run, host.user_id, files)
    container_exit = engine.run_container(run, run.work_root / log_path)
    if container_exit.status != 0:
        raise StepError(
            f"the step {step_name} exited with the status {container_exit.status}; "
            f"the end of its log, {log_path}:\n{show_printable(container_exit.log_tail)}"
        )


def judge_mounts(run: ContainerRun, user_id: int, files: StepFiles) -> None:
    for mount in run.mounts:
        if mount.is_written_by_a_step:
            continue
        judge_mount(
            files.inspect_components(run.work_root, mount.host_path), mount.is_file, user_id
        )


def judge_earlier_outputs(request: StepRunRequest, files: StepFiles) -> None:
    for earlier_output in request.row.earlier_outputs:
        directory = request.earlier_output_directories[earlier_output.step]
        judge_mount(
            files.inspect_components(request.work_root, directory),
            is_file=False,
            user_id=request.host.user_id,
        )
        judge_output_entries(
            files.list_entry_names(request.work_root / directory),
            earlier_output.step_output_names,
        )
        output = earlier_output.output
        judge_written_file(
            output.name,
            files.inspect_components(request.work_root, directory / output.name)[-1].facts,
            output.max_size_bytes,
            CONTAINER_USER_ID,
        )


def read_outputs(
    request: StepRunRequest, parsed_names: frozenset[str], files: StepFiles
) -> StepOutputs:
    output_directory = request.placement.output_directory
    judge_mount(
        files.inspect_components(request.work_root, output_directory),
        is_file=False,
        user_id=request.host.user_id,
    )
    declared = {output.name: output.max_size_bytes for output in request.row.outputs}
    written = files.read_written_outputs(
        request.work_root / output_directory, declared, parsed_names
    )
    judge_output_entries(written.entry_names, declared)
    for name, max_size_bytes in declared.items():
        judge_written_file(name, written.files[name].facts, max_size_bytes, CONTAINER_USER_ID)
    contents = {name: require_content(name, written.files[name]) for name in sorted(parsed_names)}
    return StepOutputs(
        peaks=parse_step_peaks(contents.pop(PEAKS_FILE_NAME)),
        contents=contents,
        sizes_bytes={
            name: written_file.facts.size_bytes
            for name, written_file in written.files.items()
            if name != PEAKS_FILE_NAME
        },
    )


def find_parsed_names(row: StepRow, parsed_output_names: frozenset[str]) -> frozenset[str]:
    undeclared = sorted(parsed_output_names - {output.name for output in row.outputs})
    if undeclared:
        raise StepError(
            f"the host parses {undeclared[0]}, which the step {row.name} does not declare"
        )
    return parsed_output_names | {PEAKS_FILE_NAME}


def require_content(name: str, written_file: WrittenFile) -> bytes:
    if written_file.content is None:
        raise StepError(f"the output {name}, which the host parses, was not read")
    return written_file.content
