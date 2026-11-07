import hashlib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Protocol

from mostt_model_builder.application.run_step import (
    ContainerEngine,
    HostIdentity,
    StepError,
    StepFiles,
    StepOutputs,
    StepRunRequest,
    WrittenFile,
    find_parsed_names,
    read_step_outputs,
    run_step,
)
from mostt_model_builder.domain.acceptance.verification import CandidateDigest
from mostt_model_builder.domain.containers.container_files import (
    CANDIDATE_FILE_MODE,
    FileFacts,
    MissingCandidateFileError,
    judge_candidate_copy,
    judge_mount,
    judge_output_entries,
    judge_written_file,
)
from mostt_model_builder.domain.containers.container_steps import (
    CONTAINER_USER_ID,
    INSPECT_ROW,
    StepPlacement,
    StepRow,
)
from mostt_model_builder.domain.containers.step_peaks import StepPeaks
from mostt_model_builder.domain.contract.recipe_contract import (
    HostContentOrigin,
    JudgedCandidateFile,
    StepOutputOrigin,
    VariantConversion,
    find_candidate_file_max_size_bytes,
    find_graph_source,
    list_onnx_names,
)
from mostt_model_builder.domain.formats.external_data import judge_external_data
from mostt_model_builder.domain.formats.file_formats import (
    FileFormat,
    FormatCheck,
    has_structural_check,
    start_format_check,
)
from mostt_model_builder.domain.formats.file_signatures import (
    SIGNATURE_HEAD_BYTES,
    find_signature_refusal,
)
from mostt_model_builder.domain.onnx.graph_rules import judge_graph
from mostt_model_builder.domain.onnx.inspection_report import (
    INSPECTION_REPORT_FILE_NAME,
    INSPECTION_REQUEST_FILE_NAME,
    InspectionReport,
    parse_inspection_report,
    render_inspection_request,
)

CANDIDATE_DIRECTORY_MODE = 0o750
DIGEST_ALGORITHM = "sha256"
INSPECT_COMMAND = "./model-build.sh --work <dir> --recipe <name> step inspect"
INSPECTION_PARSED_NAMES = frozenset({INSPECTION_REPORT_FILE_NAME})


class CandidateError(Exception):
    pass


class CandidateFiles(StepFiles, Protocol):
    def copy_written_file(
        self,
        directory: PurePosixPath,
        name: str,
        max_size_bytes: int,
        target: PurePosixPath,
        observe_chunk: Callable[[bytes], None],
    ) -> FileFacts: ...

    def read_host_file(
        self, directory: PurePosixPath, name: str, max_size_bytes: int
    ) -> WrittenFile: ...

    def stream_host_file(
        self,
        directory: PurePosixPath,
        name: str,
        max_size_bytes: int,
        observe_chunk: Callable[[bytes], None],
    ) -> FileFacts: ...


@dataclass(frozen=True, slots=True)
class HashedFile:
    facts: FileFacts
    sha256: str | None


class DigestFiles(StepFiles, Protocol):
    def digest_host_file(
        self, directory: PurePosixPath, name: str, max_size_bytes: int
    ) -> HashedFile: ...


@dataclass(frozen=True, slots=True)
class CandidateRequest:
    work_root: PurePosixPath
    head_tree: str
    host: HostIdentity
    candidate: Sequence[JudgedCandidateFile]
    rows: Sequence[StepRow]
    conversion_input: StepOutputOrigin | None
    candidate_directory: PurePosixPath
    inspection_directory: PurePosixPath
    conversions: Sequence[VariantConversion] = ()


@dataclass(frozen=True, slots=True)
class CandidateAssembly:
    work_root: PurePosixPath
    host: HostIdentity
    candidate: Sequence[JudgedCandidateFile]
    rows: Sequence[StepRow]
    candidate_directory: PurePosixPath
    origin_directories: Mapping[str, PurePosixPath]


@dataclass(frozen=True, slots=True)
class CandidateCopy:
    name: str
    file_format: FileFormat
    size_bytes: int
    sha256: str


@dataclass(frozen=True, slots=True)
class InspectionOutcome:
    report: InspectionReport
    peaks: StepPeaks


class CopyCheck:
    def __init__(self, file_format: FileFormat) -> None:
        self.digest = hashlib.new(DIGEST_ALGORITHM)
        self.head = bytearray()
        self.size_bytes = 0
        self.format_check: FormatCheck | None = (
            start_format_check(file_format) if has_structural_check(file_format) else None
        )

    def observe_chunk(self, chunk: bytes) -> None:
        self.digest.update(chunk)
        self.size_bytes += len(chunk)
        if len(self.head) < SIGNATURE_HEAD_BYTES:
            self.head.extend(chunk[: SIGNATURE_HEAD_BYTES - len(self.head)])
        if self.format_check is not None:
            self.format_check.feed(chunk)

    def finish(self, name: str, file_format: FileFormat) -> CandidateCopy:
        refusal = find_signature_refusal(bytes(self.head))
        if refusal is not None:
            raise CandidateError(f"the candidate file {name} begins as {refusal.marks}")
        if self.format_check is not None:
            mismatch = self.format_check.finish()
            if mismatch is not None:
                raise CandidateError(
                    f"the candidate file {name} is not {file_format.value}: {mismatch.reason}"
                )
        return CandidateCopy(name, file_format, self.size_bytes, self.digest.hexdigest())


def copy_candidate(assembly: CandidateAssembly, files: CandidateFiles) -> tuple[CandidateCopy, ...]:
    candidate_directory = make_candidate_directory(assembly, files)
    rows = {row.name: row for row in assembly.rows}
    return tuple(
        copy_candidate_file(assembly, candidate_file, candidate_directory, rows, files)
        for candidate_file in assembly.candidate
    )


# Today's checks may have grown since the candidate was assembled.
def check_candidate_copies(
    assembly: CandidateAssembly, files: CandidateFiles
) -> tuple[CandidateCopy, ...]:
    candidate_directory = assembly.candidate_directory
    judge_mount(
        files.inspect_components(assembly.work_root, candidate_directory),
        is_file=False,
        user_id=assembly.host.user_id,
    )
    copies = []
    for candidate_file in assembly.candidate:
        max_size_bytes = find_candidate_file_max_size_bytes(candidate_file.origin, assembly.rows)
        check = CopyCheck(candidate_file.file_format)
        facts = files.stream_host_file(
            assembly.work_root / candidate_directory,
            candidate_file.name,
            max_size_bytes,
            check.observe_chunk,
        )
        judge_candidate_copy(candidate_file.name, facts, max_size_bytes, assembly.host.user_id)
        copies.append(check.finish(candidate_file.name, candidate_file.file_format))
    return tuple(copies)


def inspect_candidate(
    request: CandidateRequest,
    placement: StepPlacement,
    sizes_bytes: Mapping[str, int],
    engine: ContainerEngine,
    files: StepFiles,
) -> InspectionOutcome:
    outputs = run_step(
        StepRunRequest(
            row=INSPECT_ROW,
            placement=placement,
            request_files={
                INSPECTION_REQUEST_FILE_NAME: render_inspection_request(
                    list_onnx_names(request.candidate)
                )
            },
            source_paths={},
            work_root=request.work_root,
            head_tree=request.head_tree,
            host=request.host,
            parsed_output_names=INSPECTION_PARSED_NAMES,
        ),
        engine,
        files,
    )
    return read_inspection(request, outputs, sizes_bytes)


def read_inspection(
    request: CandidateRequest, outputs: StepOutputs, sizes_bytes: Mapping[str, int]
) -> InspectionOutcome:
    report = parse_inspection_report(
        outputs.contents[INSPECTION_REPORT_FILE_NAME], list_onnx_names(request.candidate)
    )
    judge_inspected_candidate(request, report, sizes_bytes)
    return InspectionOutcome(report, outputs.peaks)


def make_candidate_directory(assembly: CandidateAssembly, files: CandidateFiles) -> PurePosixPath:
    candidate_directory = assembly.candidate_directory
    files.make_directories(assembly.work_root, candidate_directory.parent)
    if not files.make_step_directory(
        assembly.work_root / candidate_directory, CANDIDATE_DIRECTORY_MODE
    ):
        raise CandidateError(f"the candidate already exists in {candidate_directory}")
    return candidate_directory


def copy_candidate_file(
    assembly: CandidateAssembly,
    candidate_file: JudgedCandidateFile,
    candidate_directory: PurePosixPath,
    rows: Mapping[str, StepRow],
    files: CandidateFiles,
) -> CandidateCopy:
    target = assembly.work_root / candidate_directory / candidate_file.name
    check = CopyCheck(candidate_file.file_format)
    origin = candidate_file.origin
    if isinstance(origin, HostContentOrigin):
        check.observe_chunk(origin.content)
        copy = check.finish(candidate_file.name, candidate_file.file_format)
        files.write_new_file(target, origin.content, CANDIDATE_FILE_MODE)
        return copy
    row = rows.get(origin.step)
    output_directory = assembly.origin_directories.get(origin.step)
    if row is None or output_directory is None:
        raise CandidateError(f"the candidate file {candidate_file.name} comes from no step")
    max_sizes_bytes = {output.name: output.max_size_bytes for output in row.outputs}
    judge_mount(
        files.inspect_components(assembly.work_root, output_directory),
        is_file=False,
        user_id=assembly.host.user_id,
    )
    judge_output_entries(
        files.list_entry_names(assembly.work_root / output_directory), max_sizes_bytes
    )
    judge_written_file(
        origin.output,
        files.inspect_components(assembly.work_root, output_directory / origin.output)[-1].facts,
        max_sizes_bytes[origin.output],
        CONTAINER_USER_ID,
    )
    facts = files.copy_written_file(
        assembly.work_root / output_directory,
        origin.output,
        max_sizes_bytes[origin.output],
        target,
        check.observe_chunk,
    )
    judge_written_file(origin.output, facts, max_sizes_bytes[origin.output], CONTAINER_USER_ID)
    copy = check.finish(candidate_file.name, candidate_file.file_format)
    if copy.size_bytes != facts.size_bytes:
        raise CandidateError(
            f"{origin.output} of {origin.step} held {facts.size_bytes} bytes, and "
            f"{copy.size_bytes} were copied"
        )
    return copy


def judge_inspected_candidate(
    request: CandidateRequest, report: InspectionReport, sizes_bytes: Mapping[str, int]
) -> None:
    inspections = {inspection.name: inspection for inspection in report.files}
    external_data_files = {
        candidate_file.external_data_of: candidate_file.name
        for candidate_file in request.candidate
        if candidate_file.external_data_of is not None
    }
    for candidate_file in request.candidate:
        if candidate_file.file_format is not FileFormat.ONNX:
            continue
        judge_graph(
            inspections[candidate_file.name],
            find_graph_source(candidate_file.origin, request.conversion_input, request.conversions),
            external_data_files.get(candidate_file.name),
        )
    tensors_by_onnx_file = {
        name: inspection.external_tensors for name, inspection in inspections.items()
    }
    for candidate_file in request.candidate:
        if candidate_file.external_data_of is not None:
            judge_external_data(
                candidate_file.name,
                sizes_bytes[candidate_file.name],
                candidate_file.external_data_of,
                tensors_by_onnx_file,
            )


# The inspection judged the copies as they were when it ran: a copy judged again as the host left it
# is the file that the inspection read.
def judge_inspected_copies(request: CandidateRequest, files: StepFiles) -> None:
    outputs = read_step_outputs(
        INSPECT_ROW,
        request.inspection_directory,
        request.work_root,
        request.host.user_id,
        find_parsed_names(INSPECT_ROW, INSPECTION_PARSED_NAMES),
        files,
    )
    report = parse_inspection_report(
        outputs.contents[INSPECTION_REPORT_FILE_NAME], list_onnx_names(request.candidate)
    )
    judge_inspected_candidate(request, report, judge_candidate_copies(request, files))


def judge_candidate_copies(request: CandidateRequest, files: StepFiles) -> Mapping[str, int]:
    candidate_directory = locate_candidate_directory(request, files)
    sizes_bytes = {}
    for candidate_file in request.candidate:
        facts = files.inspect_components(
            request.work_root, candidate_directory / candidate_file.name
        )[-1].facts
        judge_candidate_file(
            candidate_file.name,
            facts,
            find_candidate_file_max_size_bytes(candidate_file.origin, request.rows),
            request.host.user_id,
        )
        sizes_bytes[candidate_file.name] = facts.size_bytes
    return sizes_bytes


def read_candidate_texts(request: CandidateRequest, files: CandidateFiles) -> Mapping[str, bytes]:
    candidate_directory = locate_candidate_directory(request, files)
    texts = {}
    for candidate_file in request.candidate:
        if candidate_file.file_format is not FileFormat.TEXT:
            continue
        max_size_bytes = find_candidate_file_max_size_bytes(candidate_file.origin, request.rows)
        written = files.read_host_file(
            request.work_root / candidate_directory, candidate_file.name, max_size_bytes
        )
        judge_candidate_file(
            candidate_file.name, written.facts, max_size_bytes, request.host.user_id
        )
        if written.content is None:
            raise StepError(f"the candidate file {candidate_file.name} was not read")
        texts[candidate_file.name] = written.content
    return texts


def digest_candidate(request: CandidateRequest, files: DigestFiles) -> tuple[CandidateDigest, ...]:
    candidate_directory = locate_candidate_directory(request, files)
    digests = []
    for candidate_file in request.candidate:
        max_size_bytes = find_candidate_file_max_size_bytes(candidate_file.origin, request.rows)
        hashed = files.digest_host_file(
            request.work_root / candidate_directory, candidate_file.name, max_size_bytes
        )
        judge_candidate_file(
            candidate_file.name, hashed.facts, max_size_bytes, request.host.user_id
        )
        if hashed.sha256 is None:
            raise StepError(f"the candidate file {candidate_file.name} was not hashed")
        digests.append(
            CandidateDigest(
                name=candidate_file.name,
                file_format=candidate_file.file_format,
                variants=candidate_file.variants,
                size_bytes=hashed.facts.size_bytes,
                sha256=hashed.sha256,
            )
        )
    return tuple(digests)


def locate_candidate_directory(request: CandidateRequest, files: StepFiles) -> PurePosixPath:
    judge_mount(
        files.inspect_components(request.work_root, request.candidate_directory),
        is_file=False,
        user_id=request.host.user_id,
    )
    return request.candidate_directory


def judge_candidate_file(name: str, facts: FileFacts, max_size_bytes: int, user_id: int) -> None:
    try:
        judge_candidate_copy(name, facts, max_size_bytes, user_id)
    except MissingCandidateFileError as error:
        raise CandidateError(f"{error}: run it with {INSPECT_COMMAND}") from error
