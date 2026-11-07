from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import PurePosixPath

from mostt_model_builder.application.assemble_candidate import (
    CandidateError,
    CandidateRequest,
    judge_inspected_copies,
)
from mostt_model_builder.application.run_step import (
    ContainerEngine,
    HostIdentity,
    StepFiles,
    StepOutputs,
    StepRunRequest,
    run_step,
)
from mostt_model_builder.domain.acceptance.parity_reports import ParityReport, parse_parity_report
from mostt_model_builder.domain.bench.hypotheses import Hypothesis, RunReader, judge_hypotheses
from mostt_model_builder.domain.bench.run_lines import RUN_LINES_FILE_NAME
from mostt_model_builder.domain.containers.container_steps import StepPlacement, StepRow
from mostt_model_builder.domain.containers.work_directory import locate_referenced_sources
from mostt_model_builder.domain.contract.recipe_contract import Parity
from mostt_model_builder.domain.contract.shard_request import (
    SHARD_REQUEST_FILE_NAME,
    ShardRequest,
    render_shard_request,
)
from mostt_model_builder.domain.onnx.node_placement import (
    PLACEMENT_FILE_NAME,
    PlacementSummary,
    judge_placement,
    parse_placement_summary,
)
from mostt_model_builder.domain.sources.source_pins import SourcePin


@dataclass(frozen=True, slots=True)
class RunShard:
    request: ShardRequest
    reader: RunReader
    candidate_texts: Mapping[str, bytes]


@dataclass(frozen=True, slots=True)
class RecipeStepRun:
    row: StepRow
    pins: Sequence[SourcePin]
    placement: StepPlacement
    earlier_output_directories: Mapping[str, PurePosixPath]
    work_root: PurePosixPath
    head_tree: str
    host: HostIdentity
    parities: Sequence[Parity] = ()
    shard: RunShard | None = None
    candidate: CandidateRequest | None = None


@dataclass(frozen=True, slots=True)
class MeasuredParity:
    report: ParityReport
    absolute_difference_tolerance: float


@dataclass(frozen=True, slots=True)
class RecipeStepOutcome:
    outputs: StepOutputs
    parities: tuple[MeasuredParity, ...]
    hypotheses: tuple[Hypothesis, ...] = ()
    placement: PlacementSummary | None = None


def run_recipe_step(
    run: RecipeStepRun, engine: ContainerEngine, files: StepFiles
) -> RecipeStepOutcome:
    row = run.row
    if row.binds_candidate or row.candidate_files:
        if run.candidate is None:
            raise CandidateError(
                f"the step {row.name} reads the candidate, and was given no request of it"
            )
        judge_inspected_copies(run.candidate, files)
    placement = run.placement
    if run.shard is None:
        placement = replace(placement, request_directory=None)
    outputs = run_step(
        StepRunRequest(
            row=row,
            placement=placement,
            request_files=list_request_files(run),
            source_paths=locate_referenced_sources(row.sources, run.pins),
            work_root=run.work_root,
            head_tree=run.head_tree,
            host=run.host,
            earlier_output_directories=run.earlier_output_directories,
            parsed_output_names=list_parsed_names(run),
        ),
        engine,
        files,
    )
    return read_recipe_step(run, outputs)


def list_request_files(run: RecipeStepRun) -> dict[str, bytes]:
    if run.shard is None:
        return {}
    return {SHARD_REQUEST_FILE_NAME: render_shard_request(run.shard.request)}


def list_parsed_names(run: RecipeStepRun) -> frozenset[str]:
    parsed_names = {parity.output for parity in run.parities if parity.step == run.row.name}
    if run.shard is not None:
        declared_names = {output.name for output in run.row.outputs}
        parsed_names |= {RUN_LINES_FILE_NAME, PLACEMENT_FILE_NAME} & declared_names
    return frozenset(parsed_names)


def read_recipe_step(run: RecipeStepRun, outputs: StepOutputs) -> RecipeStepOutcome:
    parities = read_parities(run.parities, run.row.name, outputs.contents)
    if run.shard is None:
        return RecipeStepOutcome(outputs, parities)
    return RecipeStepOutcome(
        outputs,
        parities,
        read_hypotheses(run.shard, outputs.contents[RUN_LINES_FILE_NAME]),
        read_placement(outputs.contents),
    )


def read_hypotheses(shard: RunShard, lines: bytes) -> tuple[Hypothesis, ...]:
    return judge_hypotheses(shard.reader(lines, shard.candidate_texts), shard.request.languages)


def read_placement(contents: Mapping[str, bytes]) -> PlacementSummary | None:
    content = contents.get(PLACEMENT_FILE_NAME)
    if content is None:
        return None
    summary = parse_placement_summary(content)
    judge_placement(summary)
    return summary


def read_parities(
    parities: Sequence[Parity], step_name: str, contents: Mapping[str, bytes]
) -> tuple[MeasuredParity, ...]:
    return tuple(
        MeasuredParity(
            parse_parity_report(contents[parity.output], parity.name),
            parity.absolute_difference_tolerance,
        )
        for parity in parities
        if parity.step == step_name
    )
