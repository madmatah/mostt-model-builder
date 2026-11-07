from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum, auto

from mostt_model_builder.domain.bench.bench_languages import BENCH_LANGUAGES
from mostt_model_builder.domain.bench.fleurs_check_files import render_fleurs_check_files
from mostt_model_builder.domain.containers.container_steps import (
    CONVERT_STEP_NAME,
    FLEURS_CHECK_ROW,
    FLEURS_CHECK_STEP_NAME,
    INSPECT_ROW,
    INSPECT_STEP_NAME,
    StepRow,
)
from mostt_model_builder.domain.containers.work_directory import (
    CANDIDATE_ENTRY_NAME,
)
from mostt_model_builder.domain.contract.derived_rows import DerivedRow
from mostt_model_builder.domain.contract.recipe_contract import (
    ConversionRow,
    HostContentOrigin,
    JudgedRecipe,
    find_runnable_row,
    list_onnx_names,
)
from mostt_model_builder.domain.contract.shard_request import (
    SHARD_REQUEST_FILE_NAME,
    render_shard_request,
)
from mostt_model_builder.domain.onnx.conversion_request import (
    CONVERSION_REQUEST_FILE_NAME,
    render_conversion_request,
)
from mostt_model_builder.domain.onnx.inspection_report import (
    INSPECTION_REQUEST_FILE_NAME,
    render_inspection_request,
)


class BuildUnitError(ValueError):
    pass


class UnitKind(Enum):
    FLEURS_CHECK = auto()
    CONVERSION = auto()
    CANDIDATE = auto()
    INSPECTION = auto()
    PLAIN = auto()
    RUN = auto()


@dataclass(frozen=True, slots=True)
class BuildUnit:
    step: str
    name: str
    kind: UnitKind
    row: StepRow | None = None
    request_files: Mapping[str, bytes] = field(default_factory=dict)
    derived_row: DerivedRow | None = None
    conversion_row: ConversionRow | None = None
    dependencies: tuple[str, ...] = ()


def list_row_dependencies(row: StepRow, reads_the_inspection: bool) -> tuple[str, ...]:
    dependencies = [earlier_output.step for earlier_output in row.earlier_outputs]
    if row.binds_candidate or row.candidate_files:
        dependencies.append(CANDIDATE_ENTRY_NAME)
        if reads_the_inspection:
            dependencies.append(INSPECT_ROW.name)
    return tuple(dict.fromkeys(dependencies))


def plan_row(
    step: str, kind: UnitKind, row: StepRow, request_files: Mapping[str, bytes]
) -> BuildUnit:
    return BuildUnit(
        step=step,
        name=row.name,
        kind=kind,
        row=row,
        request_files=request_files,
        dependencies=list_row_dependencies(row, kind is not UnitKind.INSPECTION),
    )


def plan_candidate(recipe: JudgedRecipe) -> BuildUnit:
    origins = [
        candidate_file.origin.step
        for candidate_file in recipe.candidate
        if not isinstance(candidate_file.origin, HostContentOrigin)
    ]
    return BuildUnit(
        step=INSPECT_STEP_NAME,
        name=CANDIDATE_ENTRY_NAME,
        kind=UnitKind.CANDIDATE,
        dependencies=tuple(dict.fromkeys(origins)),
    )


def plan_conversion(conversion_row: ConversionRow) -> BuildUnit:
    return BuildUnit(
        step=CONVERT_STEP_NAME,
        name=conversion_row.row.name,
        kind=UnitKind.CONVERSION,
        row=conversion_row.row,
        request_files={
            CONVERSION_REQUEST_FILE_NAME: render_conversion_request(conversion_row.request)
        },
        conversion_row=conversion_row,
        dependencies=list_row_dependencies(conversion_row.row, reads_the_inspection=True),
    )


# A run waits for the candidate and its inspection even when it binds none of its files, since its
# reader reads the candidate's texts, such as the vocabulary.
def plan_derived_row(derived_row: DerivedRow) -> BuildUnit:
    return BuildUnit(
        step=derived_row.step,
        name=derived_row.row.name,
        kind=UnitKind.RUN,
        row=derived_row.row,
        request_files={SHARD_REQUEST_FILE_NAME: render_shard_request(derived_row.request)},
        derived_row=derived_row,
        dependencies=tuple(
            dict.fromkeys(
                (
                    *list_row_dependencies(derived_row.row, reads_the_inspection=True),
                    CANDIDATE_ENTRY_NAME,
                    INSPECT_ROW.name,
                )
            )
        ),
    )


def list_key_dependencies(unit: BuildUnit) -> tuple[str, ...]:
    if unit.row is None:
        return unit.dependencies
    return list_row_dependencies(unit.row, reads_the_inspection=False)


def plan_step(recipe: JudgedRecipe, step: str) -> tuple[BuildUnit, ...]:
    if step == FLEURS_CHECK_STEP_NAME:
        return (
            plan_row(
                step,
                UnitKind.FLEURS_CHECK,
                FLEURS_CHECK_ROW,
                render_fleurs_check_files(FLEURS_CHECK_ROW, BENCH_LANGUAGES, recipe.pins),
            ),
        )
    if step == CONVERT_STEP_NAME:
        if not recipe.conversion_rows:
            raise BuildUnitError(f"the recipe {recipe.name} converts none of its variants")
        return tuple(plan_conversion(conversion_row) for conversion_row in recipe.conversion_rows)
    if step == INSPECT_STEP_NAME:
        return (
            plan_candidate(recipe),
            plan_row(
                step,
                UnitKind.INSPECTION,
                INSPECT_ROW,
                {
                    INSPECTION_REQUEST_FILE_NAME: render_inspection_request(
                        list_onnx_names(recipe.candidate)
                    )
                },
            ),
        )
    derived_rows = [derived_row for derived_row in recipe.derived_rows if derived_row.step == step]
    if derived_rows:
        return tuple(plan_derived_row(derived_row) for derived_row in derived_rows)
    return (plan_row(step, UnitKind.PLAIN, find_runnable_row(recipe, step), {}),)


def list_recipe_steps(recipe: JudgedRecipe) -> tuple[str, ...]:
    derived_steps = {derived_row.row.name: derived_row.step for derived_row in recipe.derived_rows}
    conversion_rows = {conversion_row.row.name for conversion_row in recipe.conversion_rows}
    return tuple(
        dict.fromkeys(
            derived_steps.get(row.name, row.name)
            for row in recipe.rows
            if row.name not in conversion_rows
        )
    )


# The steps that the build leaves out, such as a description of the model's names, come after it,
# so that their keys can draw on any of its outputs.
def plan_units(recipe: JudgedRecipe, steps: Sequence[str]) -> tuple[BuildUnit, ...]:
    units = tuple(unit for step in steps for unit in plan_step(recipe, step))
    names = [unit.name for unit in units]
    if len(set(names)) != len(names):
        raise BuildUnitError(f"the recipe {recipe.name} plans an entry twice")
    return units


def list_every_step(recipe: JudgedRecipe) -> tuple[str, ...]:
    built = recipe.build_order
    return (*built, *(step for step in list_recipe_steps(recipe) if step not in built))


def select_units(units: Sequence[BuildUnit], steps: Sequence[str]) -> tuple[BuildUnit, ...]:
    by_name = {unit.name: unit for unit in units}
    needed = {unit.name for unit in units if unit.step in steps}
    pending = list(needed)
    while pending:
        unit = by_name[pending.pop()]
        for dependency in unit.dependencies:
            if dependency not in by_name:
                raise BuildUnitError(
                    f"{unit.name} reads the outputs of {dependency}, which the build does not plan"
                )
            if dependency not in needed:
                needed.add(dependency)
                pending.append(dependency)
    order = {unit.name: position for position, unit in enumerate(units)}
    for name in needed:
        for dependency in by_name[name].dependencies:
            if order[dependency] > order[name]:
                raise BuildUnitError(
                    f"{name} reads the outputs of {dependency}, which the build plans after it"
                )
    return tuple(unit for unit in units if unit.name in needed)


# A published revision is checked against the candidate and its inspection alone, so its check needs
# neither the runs' entries nor a GPU to key them.
def plan_inspected_units(recipe: JudgedRecipe) -> tuple[BuildUnit, ...]:
    return select_units(plan_units(recipe, recipe.build_order), (INSPECT_STEP_NAME,))
