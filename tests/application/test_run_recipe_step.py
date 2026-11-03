import re
from collections.abc import Callable
from dataclasses import dataclass, replace
from functools import partial
from pathlib import PurePosixPath

import pytest
from run_step_fakes import (
    GROUP_ID,
    USER_ID,
    FakeContainerEngine,
    FakeStepFiles,
    build_directory_facts,
    build_file_facts,
    build_step_output,
)

from mostt_model_builder.application.run_recipe_step import (
    MeasuredParity,
    RecipeStepOutcome,
    RecipeStepRun,
    run_recipe_step,
)
from mostt_model_builder.application.run_step import HostIdentity, ImageFacts
from mostt_model_builder.domain.acceptance.parity_reports import (
    ParityReport,
    ParityReportError,
    render_parity_report,
)
from mostt_model_builder.domain.containers.account_ranges import HostAccounts
from mostt_model_builder.domain.containers.base_images import ENVIRONMENT_PYTHON
from mostt_model_builder.domain.containers.container_steps import (
    PEAKS_FILE_NAME,
    PEAKS_OUTPUT,
    BindMount,
    ContainerImage,
    ContainerRun,
    DeclaredOutput,
    Device,
    EarlierOutput,
    SourceReference,
    StepLimits,
    StepRow,
)
from mostt_model_builder.domain.containers.step_peaks import StepPeaks, render_step_peaks
from mostt_model_builder.domain.containers.work_directory import WorkDirectoryError
from mostt_model_builder.domain.contract.recipe_contract import Parity
from mostt_model_builder.domain.sources.source_pins import RepositoryType, SourcePin

WORK_ROOT = PurePosixPath("/data/builds/work")
HEAD_TREE = "3f1d1ffe79bc3336eb583331cfbf9b189fd45f01"
COMMIT = "0123456789abcdef0123456789abcdef01234567"
REVISION = "73175eb7aeb0d82f1e2a6b53b3aabc10a90bcd0b"
IMAGE_ID = "sha256:" + "e" * 64
RECIPE_NAME = "family"
RUN_DIRECTORY = PurePosixPath("runs/family", COMMIT)
EXPORT_IMAGE = ContainerImage(RECIPE_NAME, "export", Device.CPU, ENVIRONMENT_PYTHON)
WEIGHTS = SourceReference("maker/weights", "model.safetensors")
WEIGHTS_PIN = SourcePin(
    RepositoryType.MODEL, "maker/weights", REVISION, "model.safetensors", 1, "0" * 64
)
WEIGHTS_PATH = PurePosixPath("sources/maker/weights", REVISION, "model.safetensors")
NAMES_OUTPUT = DeclaredOutput("names", 1024)
PEAKS = StepPeaks(memory_peak_bytes=64 * 2**20, process_count_peak=6)
ENCODER_DATA_OUTPUT = DeclaredOutput("encoder-model.onnx.data", 3 * 2**30)
ENCODER_DATA = bytes(4096)
PARITY_OUTPUT = DeclaredOutput("encoder-parity.json", 4096)
ENCODER_PARITY = Parity("encoder-against-reference", "export", PARITY_OUTPUT.name, 5e-5)
DESCRIBE_PARITY = Parity("names-against-reference", "describe", NAMES_OUTPUT.name, 1e-3)
ENCODER_PARITY_REPORT = ParityReport("encoder-against-reference", 4.18e-6, 20)
EXPORT_ROW = StepRow(
    name="export",
    image=EXPORT_IMAGE,
    entry_module="mostt_recipe_family.steps.export",
    sources=(WEIGHTS,),
    has_request=False,
    outputs=(ENCODER_DATA_OUTPUT, PEAKS_OUTPUT),
    limits=StepLimits(memory_mib=1024, process_count=64, cpu_count=2, temporary_directory_mib=64),
    earlier_outputs=(
        EarlierOutput("describe", NAMES_OUTPUT, (NAMES_OUTPUT.name, PEAKS_FILE_NAME)),
    ),
)
HOST = HostIdentity(USER_ID, GROUP_ID, 20, HostAccounts(False, (), ()))
RUN = RecipeStepRun(
    row=EXPORT_ROW,
    recipe_name=RECIPE_NAME,
    pins=(WEIGHTS_PIN,),
    commit=COMMIT,
    work_root=WORK_ROOT,
    head_tree=HEAD_TREE,
    host=HOST,
)


@dataclass(frozen=True)
class CompletedRecipeStep:
    outcome: RecipeStepOutcome
    runs: list[ContainerRun]
    files: FakeStepFiles


def build_files() -> FakeStepFiles:
    files = FakeStepFiles()
    files.add_directory(WORK_ROOT)
    files.add_file(WORK_ROOT / WEIGHTS_PATH, build_file_facts(1, mode=0o444), b"x")
    describe_directory = WORK_ROOT / RUN_DIRECTORY / "describe"
    files.add_directory(describe_directory, build_directory_facts(0o2770))
    for name, content in ((NAMES_OUTPUT.name, b"names\n"), (PEAKS_FILE_NAME, b"{}")):
        files.add_file(describe_directory / name, *build_step_output(content))
    return files


def build_engine(files: FakeStepFiles, parity_report: bytes | None = None) -> FakeContainerEngine:
    step_outputs = {
        ENCODER_DATA_OUTPUT.name: build_step_output(ENCODER_DATA),
        PEAKS_FILE_NAME: build_step_output(render_step_peaks(PEAKS)),
    }
    if parity_report is not None:
        step_outputs[PARITY_OUTPUT.name] = build_step_output(parity_report)
    return FakeContainerEngine(
        files=files,
        images={EXPORT_IMAGE.tag: ImageFacts(IMAGE_ID, HEAD_TREE)},
        step_outputs=step_outputs,
    )


def build_parity_run() -> RecipeStepRun:
    return replace(
        RUN,
        row=replace(EXPORT_ROW, outputs=(ENCODER_DATA_OUTPUT, PARITY_OUTPUT, PEAKS_OUTPUT)),
        parities=(DESCRIBE_PARITY, ENCODER_PARITY),
    )


class TestTheRecipeStepRun:
    class TestTheRunRecipeStepFunction:
        class TestWhenAStepOfTheRecipeRuns:
            @pytest.fixture
            def recipe_step(self) -> CompletedRecipeStep:
                files = build_files()
                engine = build_engine(files)
                return CompletedRecipeStep(run_recipe_step(RUN, engine, files), engine.runs, files)

            def test_should_give_back_its_peaks(self, recipe_step: CompletedRecipeStep) -> None:
                assert recipe_step.outcome.outputs.peaks == PEAKS

            def test_should_measure_no_parity_since_its_recipe_declares_none(
                self, recipe_step: CompletedRecipeStep
            ) -> None:
                assert recipe_step.outcome.parities == ()

            def test_should_read_its_peaks_alone_since_no_parser_of_the_host_reads_the_rest(
                self, recipe_step: CompletedRecipeStep
            ) -> None:
                assert recipe_step.files.read_paths == [
                    WORK_ROOT / RUN_DIRECTORY / "export" / PEAKS_FILE_NAME
                ]

            def test_should_give_back_the_size_of_the_outputs_it_does_not_read(
                self, recipe_step: CompletedRecipeStep
            ) -> None:
                assert recipe_step.outcome.outputs.sizes_bytes == {
                    ENCODER_DATA_OUTPUT.name: len(ENCODER_DATA)
                }

            def test_should_write_its_outputs_under_the_run_of_the_recipe_at_the_commit(
                self, recipe_step: CompletedRecipeStep
            ) -> None:
                assert [
                    mount.host_path
                    for run in recipe_step.runs
                    for mount in run.mounts
                    if not mount.is_read_only
                ] == [RUN_DIRECTORY / "export"]

            def test_should_log_beside_the_outputs_of_the_run(
                self, recipe_step: CompletedRecipeStep
            ) -> None:
                assert WORK_ROOT / RUN_DIRECTORY / "logs/export.log" in recipe_step.files.entries

            def test_should_write_no_request_since_its_row_has_none(
                self, recipe_step: CompletedRecipeStep
            ) -> None:
                assert WORK_ROOT / RUN_DIRECTORY / "requests" not in recipe_step.files.entries

            def test_should_bind_each_source_at_the_layout_of_its_pin(
                self, recipe_step: CompletedRecipeStep
            ) -> None:
                assert BindMount(WEIGHTS_PATH, PurePosixPath("/") / WEIGHTS_PATH, True, True) in [
                    mount for run in recipe_step.runs for mount in run.mounts
                ]

            def test_should_bind_the_earlier_output_from_the_same_run(
                self, recipe_step: CompletedRecipeStep
            ) -> None:
                assert BindMount(
                    RUN_DIRECTORY / "describe/names",
                    PurePosixPath("/inputs/describe/names"),
                    is_read_only=True,
                    is_file=True,
                    is_written_by_a_step=True,
                ) in [mount for run in recipe_step.runs for mount in run.mounts]

        class TestWhenTheStepWritesTheReportOfAParity:
            @pytest.fixture
            def recipe_step(self) -> CompletedRecipeStep:
                files = build_files()
                engine = build_engine(files, render_parity_report(ENCODER_PARITY_REPORT))
                outcome = run_recipe_step(build_parity_run(), engine, files)
                return CompletedRecipeStep(outcome, engine.runs, files)

            def test_should_read_the_parities_of_that_step_alone_with_their_tolerance(
                self, recipe_step: CompletedRecipeStep
            ) -> None:
                assert recipe_step.outcome.parities == (
                    MeasuredParity(ENCODER_PARITY_REPORT, 5e-5),
                )

            def test_should_read_the_report_and_the_peaks_alone_of_its_outputs(
                self, recipe_step: CompletedRecipeStep
            ) -> None:
                assert recipe_step.files.read_paths == [
                    WORK_ROOT / RUN_DIRECTORY / "export" / PARITY_OUTPUT.name,
                    WORK_ROOT / RUN_DIRECTORY / "export" / PEAKS_FILE_NAME,
                ]

        class TestWhenTheReportOfAParityNamesAnotherParity:
            @pytest.fixture
            def run_act(self) -> Callable[[], RecipeStepOutcome]:
                files = build_files()
                report = render_parity_report(replace(ENCODER_PARITY_REPORT, name="other"))
                return partial(
                    run_recipe_step, build_parity_run(), build_engine(files, report), files
                )

            def test_should_refuse_it_since_the_host_trusts_no_report_it_did_not_ask_for(
                self, run_act: Callable[[], RecipeStepOutcome]
            ) -> None:
                with pytest.raises(
                    ParityReportError,
                    match=re.escape(
                        "the parity report names the parity 'other', not encoder-against-reference"
                    ),
                ):
                    run_act()

        class TestWhenNoPinGivesASourceOfTheRow:
            @pytest.fixture
            def run_act(self) -> Callable[[], RecipeStepOutcome]:
                files = build_files()
                return partial(run_recipe_step, replace(RUN, pins=()), build_engine(files), files)

            def test_should_refuse_it_and_name_the_source(
                self, run_act: Callable[[], RecipeStepOutcome]
            ) -> None:
                with pytest.raises(
                    WorkDirectoryError,
                    match=re.escape(
                        "no pin gives model.safetensors of maker/weights, which a step reads"
                    ),
                ):
                    run_act()
