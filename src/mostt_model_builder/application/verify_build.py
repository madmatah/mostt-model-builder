import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Protocol

from mostt_model_builder.application.assemble_candidate import (
    CandidateFiles,
    CandidateRequest,
    DigestFiles,
    digest_candidate,
    judge_inspected_copies,
    read_candidate_texts,
)
from mostt_model_builder.application.run_recipe_step import (
    RunShard,
    read_hypotheses,
    read_placement,
)
from mostt_model_builder.application.run_step import (
    StepOutputs,
    StepRunRequest,
    find_parsed_names,
    read_outputs,
)
from mostt_model_builder.domain.acceptance.acceptance import Comparison
from mostt_model_builder.domain.acceptance.parity_reports import parse_parity_report
from mostt_model_builder.domain.acceptance.verification import (
    VERIFICATION_REPORT_FILE_NAME,
    ParityMeasure,
    RunTranscripts,
    Verification,
    VerificationError,
    collect_reference_words,
    list_failures,
    measure_parity,
    render_verification_report,
    verify_transcripts,
)
from mostt_model_builder.domain.bench.fleurs_index import (
    BENCH_LANGUAGES,
    FLEURS_REPOSITORY,
    FleursIndexError,
    locate_bench_files,
    parse_fleurs_index,
)
from mostt_model_builder.domain.bench.hypotheses import Hypothesis
from mostt_model_builder.domain.bench.run_lines import RUN_LINES_FILE_NAME
from mostt_model_builder.domain.containers.container_files import judge_mount
from mostt_model_builder.domain.containers.container_steps import StepRow
from mostt_model_builder.domain.containers.work_directory import (
    locate_run,
    locate_source,
    place_step,
)
from mostt_model_builder.domain.contract.derived_rows import DerivedRow
from mostt_model_builder.domain.contract.recipe_contract import JudgedRun, Parity
from mostt_model_builder.domain.onnx.node_placement import PLACEMENT_FILE_NAME
from mostt_model_builder.domain.sources.source_pins import SourcePin, is_pinned_content

INDEX_ENCODING = "utf-8"
FETCH_COMMAND = "./model-build.sh --work <dir> --recipe <name> fetch"


class BuildRejectedError(Exception):
    pass


class VerificationFiles(CandidateFiles, DigestFiles, Protocol):
    def replace_file(self, path: PurePosixPath, content: bytes) -> None: ...


@dataclass(frozen=True, slots=True)
class VerificationRequest:
    candidate: CandidateRequest
    pins: Sequence[SourcePin]
    runs: Sequence[JudgedRun]
    derived_rows: Sequence[DerivedRow]
    parities: Sequence[Parity]
    comparisons: Sequence[Comparison]


@dataclass(frozen=True, slots=True)
class VerifiedBuild:
    verification: Verification
    report_path: PurePosixPath


def verify_build(request: VerificationRequest, files: VerificationFiles) -> VerifiedBuild:
    candidate = request.candidate
    judge_inspected_copies(candidate, files)
    candidate_digests = digest_candidate(candidate, files)
    candidate_texts = read_candidate_texts(candidate, files)
    references = collect_reference_words(
        read_bench_transcriptions(request.pins, candidate.work_root, candidate.host.user_id, files)
    )
    outputs = read_verified_outputs(request, files)
    runs = [
        read_run_transcripts(run, request.derived_rows, outputs, candidate_texts)
        for run in request.runs
    ]
    verification = verify_transcripts(
        runs,
        references,
        request.comparisons,
        read_parity_measures(request.parities, outputs),
        candidate_digests,
    )
    run_directory = locate_run(candidate.recipe_name, candidate.commit)
    judge_mount(
        files.inspect_components(candidate.work_root, run_directory),
        is_file=False,
        user_id=candidate.host.user_id,
    )
    report_path = run_directory / VERIFICATION_REPORT_FILE_NAME
    files.replace_file(candidate.work_root / report_path, render_verification_report(verification))
    return VerifiedBuild(verification, report_path)


def refuse_rejected_build(verified: VerifiedBuild) -> None:
    failures = list_failures(verified.verification)
    if failures:
        raise BuildRejectedError(
            f"the build fails {len(failures)} of its rules, as {verified.report_path} records: "
            + "; ".join(failures)
        )


def read_bench_transcriptions(
    pins: Sequence[SourcePin],
    work_root: PurePosixPath,
    user_id: int,
    files: VerificationFiles,
) -> Mapping[str, Mapping[str, str]]:
    pins_by_path = {(pin.repository, pin.path): pin for pin in pins}
    transcriptions = {}
    for language in BENCH_LANGUAGES:
        index_path, _ = locate_bench_files(language)
        pin = pins_by_path.get((FLEURS_REPOSITORY, index_path))
        if pin is None:
            raise VerificationError(f"no pin gives {index_path} of {FLEURS_REPOSITORY}")
        content = read_pinned_file(pin, work_root, user_id, files)
        try:
            utterances = parse_fleurs_index(content.decode(INDEX_ENCODING))
        except (UnicodeDecodeError, FleursIndexError) as error:
            raise VerificationError(f"the index of {language}: {error}") from error
        transcriptions[language] = {
            utterance.recording_name: utterance.transcription for utterance in utterances
        }
    return transcriptions


def read_pinned_file(
    pin: SourcePin, work_root: PurePosixPath, user_id: int, files: VerificationFiles
) -> bytes:
    source_path = locate_source(pin)
    judge_mount(files.inspect_components(work_root, source_path), is_file=True, user_id=user_id)
    written = files.read_host_file(work_root / source_path.parent, source_path.name, pin.size_bytes)
    content = written.content
    if content is None or not is_pinned_content(
        pin, len(content), hashlib.sha256(content).hexdigest()
    ):
        raise VerificationError(
            f"{source_path} no longer matches its pin: fetch it again with {FETCH_COMMAND}"
        )
    return content


def read_verified_outputs(
    request: VerificationRequest, files: VerificationFiles
) -> Mapping[str, StepOutputs]:
    rows = {row.name: row for row in request.candidate.rows}
    parsed_names: dict[str, set[str]] = {}
    for derived_row in request.derived_rows:
        declared = {output.name for output in derived_row.row.outputs}
        parsed_names.setdefault(derived_row.row.name, set()).update(
            {RUN_LINES_FILE_NAME, PLACEMENT_FILE_NAME} & declared
        )
    for parity in request.parities:
        parsed_names.setdefault(parity.step, set()).add(parity.output)
    return {
        row_name: read_row_outputs(rows[row_name], frozenset(names), request.candidate, files)
        for row_name, names in parsed_names.items()
    }


def read_row_outputs(
    row: StepRow,
    parsed_names: frozenset[str],
    candidate: CandidateRequest,
    files: VerificationFiles,
) -> StepOutputs:
    return read_outputs(
        StepRunRequest(
            row=row,
            placement=place_step(candidate.recipe_name, candidate.commit, row.name),
            request_files={},
            source_paths={},
            work_root=candidate.work_root,
            head_tree=candidate.head_tree,
            host=candidate.host,
        ),
        find_parsed_names(row, parsed_names),
        files,
    )


def read_run_transcripts(
    run: JudgedRun,
    derived_rows: Sequence[DerivedRow],
    outputs: Mapping[str, StepOutputs],
    candidate_texts: Mapping[str, bytes],
) -> RunTranscripts:
    hypotheses: list[Hypothesis] = []
    for derived_row in derived_rows:
        if derived_row.run != run.name:
            continue
        contents = outputs[derived_row.row.name].contents
        hypotheses.extend(
            read_hypotheses(
                RunShard(derived_row.request, run.reader, candidate_texts),
                contents[RUN_LINES_FILE_NAME],
            )
        )
        read_placement(contents)
    return RunTranscripts(run.name, run.languages, tuple(hypotheses))


def read_parity_measures(
    parities: Sequence[Parity], outputs: Mapping[str, StepOutputs]
) -> tuple[ParityMeasure, ...]:
    return tuple(
        measure_parity(
            parse_parity_report(outputs[parity.step].contents[parity.output], parity.name),
            parity.absolute_difference_tolerance,
        )
        for parity in parities
    )
