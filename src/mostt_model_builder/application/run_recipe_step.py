from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import PurePosixPath

from mostt_model_builder.application.run_step import (
    ContainerEngine,
    HostIdentity,
    StepFiles,
    StepOutputs,
    StepRunRequest,
    run_step,
)
from mostt_model_builder.domain.acceptance.parity_reports import ParityReport, parse_parity_report
from mostt_model_builder.domain.containers.container_steps import StepRow
from mostt_model_builder.domain.containers.work_directory import (
    locate_referenced_sources,
    place_step,
)
from mostt_model_builder.domain.contract.recipe_contract import Parity
from mostt_model_builder.domain.sources.source_pins import SourcePin


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


@dataclass(frozen=True, slots=True)
class MeasuredParity:
    report: ParityReport
    absolute_difference_tolerance: float


@dataclass(frozen=True, slots=True)
class RecipeStepOutcome:
    outputs: StepOutputs
    parities: tuple[MeasuredParity, ...]


def run_recipe_step(
    run: RecipeStepRun, engine: ContainerEngine, files: StepFiles
) -> RecipeStepOutcome:
    row = run.row
    placement = place_step(run.recipe_name, run.commit, row.name)
    outputs = run_step(
        StepRunRequest(
            row=row,
            placement=replace(placement, request_directory=None),
            request_files={},
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
            parsed_output_names=frozenset(
                parity.output for parity in run.parities if parity.step == row.name
            ),
        ),
        engine,
        files,
    )
    return RecipeStepOutcome(outputs, read_parities(run.parities, row.name, outputs.contents))


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
