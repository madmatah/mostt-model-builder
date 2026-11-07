from collections.abc import Callable
from dataclasses import replace
from functools import partial

import pytest

from mostt_model_builder.domain.cache.build_units import (
    BuildUnit,
    BuildUnitError,
    UnitKind,
    list_every_step,
    list_key_dependencies,
    plan_inspected_units,
    plan_units,
    select_units,
)
from mostt_model_builder.domain.containers.base_images import BASE_PYTHON
from mostt_model_builder.domain.containers.container_steps import (
    PEAKS_OUTPUT,
    ContainerImage,
    DeclaredOutput,
    StepLimits,
    StepRow,
)
from mostt_model_builder.domain.containers.devices import Device
from mostt_model_builder.domain.contract.derived_rows import (
    DeviceRow,
    ShardedRun,
    StepTemplate,
    derive_rows,
)
from mostt_model_builder.domain.contract.recipe_contract import (
    JudgedCandidateFile,
    JudgedRecipe,
    JudgedRun,
    StepOutputOrigin,
)
from mostt_model_builder.domain.contract.shard_request import (
    SHARD_REQUEST_FILE_NAME,
    render_shard_request,
)
from mostt_model_builder.domain.formats.file_formats import FileFormat
from mostt_model_builder.domain.onnx.inspection_report import (
    INSPECTION_REQUEST_FILE_NAME,
    render_inspection_request,
)

LIMITS = StepLimits(memory_mib=1024, process_count=64, cpu_count=2, temporary_directory_mib=64)
EXPORT_IMAGE = ContainerImage("family", "export", Device.CPU, BASE_PYTHON)
CPU_IMAGE = ContainerImage("family", "transcribe-cpu", Device.CPU, BASE_PYTHON)
EXPORT_ROW = StepRow(
    name="export",
    image=EXPORT_IMAGE,
    entry_module="mostt_recipe_family.steps.export",
    sources=(),
    has_request=False,
    outputs=(DeclaredOutput("encoder.onnx", 4096), PEAKS_OUTPUT),
    limits=LIMITS,
)
DESCRIBE_ROW = replace(EXPORT_ROW, name="describe", entry_module="mostt_recipe_family.steps.names")
TEMPLATE = StepTemplate(
    name="transcribe",
    entry_module="mostt_recipe_family.steps.transcribe",
    sources=(),
    earlier_outputs=(),
    binds_candidate=True,
    outputs=(PEAKS_OUTPUT,),
    devices={Device.CPU: DeviceRow(CPU_IMAGE, LIMITS, ())},
)
SHARDED_RUN = ShardedRun("float32-cpu", "float32", Device.CPU, ("fr_fr",), 1, ("encoder.onnx",))
REFERENCE_RUN = ShardedRun("reference-cpu", "reference", Device.CPU, ("fr_fr",), 1)
DERIVED_ROWS = derive_rows(TEMPLATE, (SHARDED_RUN, REFERENCE_RUN))
CANDIDATE = (
    JudgedCandidateFile(
        "encoder.onnx", FileFormat.ONNX, ("float32",), StepOutputOrigin("export", "encoder.onnx")
    ),
)
RECIPE = JudgedRecipe(
    name="family",
    pins=(),
    images=(EXPORT_IMAGE, CPU_IMAGE),
    rows=(EXPORT_ROW, DESCRIBE_ROW, *(derived_row.row for derived_row in DERIVED_ROWS)),
    candidate=CANDIDATE,
    parities=(),
    card_template=None,
    runs=(
        JudgedRun("float32-cpu", "transcribe", Device.CPU, "float32", ("fr_fr",), 1, lambda *_: ()),
    ),
    derived_rows=DERIVED_ROWS,
    build_order=("export", "inspect", "transcribe"),
)
RUN_ROW_NAME = "transcribe-float32-cpu-fr_fr-0"
REFERENCE_ROW_NAME = "transcribe-reference-cpu-fr_fr-0"


def describe_units(units: tuple[BuildUnit, ...]) -> list[tuple[str, str, UnitKind]]:
    return [(unit.step, unit.name, unit.kind) for unit in units]


class TestTheBuildUnits:
    class TestThePlanUnitsFunction:
        class TestWhenTheBuildOrderIsPlanned:
            @pytest.fixture
            def units(self) -> tuple[BuildUnit, ...]:
                return plan_units(RECIPE, RECIPE.build_order)

            def test_should_plan_each_row_and_the_candidate_in_the_build_order(
                self, units: tuple[BuildUnit, ...]
            ) -> None:
                assert describe_units(units) == [
                    ("export", "export", UnitKind.PLAIN),
                    ("inspect", "candidate", UnitKind.CANDIDATE),
                    ("inspect", "inspect", UnitKind.INSPECTION),
                    ("transcribe", RUN_ROW_NAME, UnitKind.RUN),
                    ("transcribe", REFERENCE_ROW_NAME, UnitKind.RUN),
                ]

            def test_should_make_the_candidate_depend_on_the_rows_its_files_come_from(
                self, units: tuple[BuildUnit, ...]
            ) -> None:
                assert units[1].dependencies == ("export",)

            def test_should_ask_the_inspection_for_the_candidates_onnx_files(
                self, units: tuple[BuildUnit, ...]
            ) -> None:
                assert units[2].request_files == {
                    INSPECTION_REQUEST_FILE_NAME: render_inspection_request(["encoder.onnx"])
                }

            def test_should_give_a_run_its_shard_request(
                self, units: tuple[BuildUnit, ...]
            ) -> None:
                assert units[3].request_files == {
                    SHARD_REQUEST_FILE_NAME: render_shard_request(DERIVED_ROWS[0].request)
                }

            def test_should_make_a_run_wait_for_the_candidate_and_its_inspection(
                self, units: tuple[BuildUnit, ...]
            ) -> None:
                assert units[3].dependencies == ("candidate", "inspect")

            def test_should_key_a_run_on_the_candidate_alone_of_those(
                self, units: tuple[BuildUnit, ...]
            ) -> None:
                assert list_key_dependencies(units[3]) == ("candidate",)

            def test_should_make_a_run_that_binds_no_candidate_file_wait_for_them_for_its_reader(
                self, units: tuple[BuildUnit, ...]
            ) -> None:
                assert (units[4].dependencies, list_key_dependencies(units[4])) == (
                    ("candidate", "inspect"),
                    (),
                )

    class TestTheListEveryStepFunction:
        @pytest.fixture
        def steps(self) -> tuple[str, ...]:
            return list_every_step(RECIPE)

        def test_should_place_the_steps_the_build_leaves_out_after_it(
            self, steps: tuple[str, ...]
        ) -> None:
            assert steps == ("export", "inspect", "transcribe", "describe")

    class TestThePlanInspectedUnitsFunction:
        class TestWhenTheBuildRunsStepsAfterTheInspection:
            @pytest.fixture
            def inspected_units(self) -> list[str]:
                return [unit.name for unit in plan_inspected_units(RECIPE)]

            def test_should_end_at_the_inspection_of_the_candidate(
                self, inspected_units: list[str]
            ) -> None:
                assert inspected_units == ["export", "candidate", "inspect"]

    class TestTheSelectUnitsFunction:
        class TestWhenAStepIsAskedFor:
            @pytest.fixture
            def selected(self) -> list[str]:
                units = plan_units(RECIPE, list_every_step(RECIPE))
                return [unit.name for unit in select_units(units, ("transcribe",))]

            def test_should_take_its_rows_and_every_entry_they_draw_on_in_order(
                self, selected: list[str]
            ) -> None:
                assert selected == [
                    "export",
                    "candidate",
                    "inspect",
                    RUN_ROW_NAME,
                    REFERENCE_ROW_NAME,
                ]

        class TestWhenADependencyIsPlannedAfterTheStepThatReadsIt:
            @pytest.fixture
            def select_act(self) -> Callable[[], object]:
                units = plan_units(RECIPE, ("inspect", "export", "transcribe"))
                return partial(select_units, units, ("inspect",))

            def test_should_refuse_the_order(self, select_act: Callable[[], object]) -> None:
                with pytest.raises(
                    BuildUnitError,
                    match="candidate reads the outputs of export, which the build plans after it",
                ):
                    select_act()

        class TestWhenADependencyIsNotPlanned:
            @pytest.fixture
            def select_act(self) -> Callable[[], object]:
                units = plan_units(RECIPE, ("inspect",))
                return partial(select_units, units, ("inspect",))

            def test_should_refuse_it(self, select_act: Callable[[], object]) -> None:
                with pytest.raises(
                    BuildUnitError,
                    match="candidate reads the outputs of export, which the build does not plan",
                ):
                    select_act()
