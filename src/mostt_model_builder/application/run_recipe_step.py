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
from mostt_model_builder.domain.containers.container_steps import StepRow
from mostt_model_builder.domain.containers.work_directory import (
    locate_referenced_sources,
    place_step,
)
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
    recipe_name: str
    pins: Sequence[SourcePin]
    commit: str
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
    if row.binds_candidate:
        if run.candidate is None:
            raise CandidateError(
                f"the step {row.name} reads the candidate, and was given no request of it"
            )
        judge_inspected_copies(run.candidate, files)
    placement = place_step(run.recipe_name, run.commit, row.name)
    declared_names = {output.name for output in row.outputs}
    parsed_names = {parity.output for parity in run.parities if parity.step == row.name}
    request_files = {}
    if run.shard is not None:
        request_files[SHARD_REQUEST_FILE_NAME] = render_shard_request(run.shard.request)
        parsed_names |= {RUN_LINES_FILE_NAME, PLACEMENT_FILE_NAME} & declared_names
    else:
        placement = replace(placement, request_directory=None)
    outputs = run_step(
        StepRunRequest(
            row=row,
            placement=placement,
            request_files=request_files,
            source_paths=locate_referenced_sources(row.sources, run.pins),
            work_root=run.work_root,
            head_tree=run.head_tree,
            host=run.host,
            earlier_output_directories={
                earlier_output.step: place_step(
                    run.recipe_name, run.commit, earlier_output.step
                ).output_directory
                for earlier_output in row.earlier_outputs
            },
            parsed_output_names=frozenset(parsed_names),
        ),
        engine,
        files,
    )
    parities = read_parities(run.parities, row.name, outputs.contents)
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
