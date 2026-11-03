from collections.abc import Sequence
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
from mostt_model_builder.domain.containers.container_steps import StepRow
from mostt_model_builder.domain.containers.work_directory import (
    locate_referenced_sources,
    place_step,
)
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


def run_recipe_step(run: RecipeStepRun, engine: ContainerEngine, files: StepFiles) -> StepOutputs:
    row = run.row
    placement = place_step(run.recipe_name, run.commit, row.name)
    return run_step(
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
        ),
        engine,
        files,
    )
