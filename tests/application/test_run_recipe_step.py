import re
from collections.abc import Callable, Mapping
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

from mostt_model_builder.application.assemble_candidate import CandidateError, CandidateRequest
from mostt_model_builder.application.run_recipe_step import (
    MeasuredParity,
    RecipeStepOutcome,
    RecipeStepRun,
    RunShard,
    run_recipe_step,
)
from mostt_model_builder.application.run_step import HostIdentity, ImageFacts
from mostt_model_builder.domain.acceptance.parity_reports import (
    ParityReport,
    ParityReportError,
    render_parity_report,
)
from mostt_model_builder.domain.bench.hypotheses import (
    RUN_LINES_FILE_NAME,
    RUN_LINES_MAX_SIZE_BYTES,
    Hypothesis,
    HypothesisError,
    parse_run_lines,
)
from mostt_model_builder.domain.containers.account_ranges import HostAccounts
from mostt_model_builder.domain.containers.base_images import ENVIRONMENT_PYTHON
from mostt_model_builder.domain.containers.container_files import ContainerFileError
from mostt_model_builder.domain.containers.container_steps import (
    INSPECT_STEP_NAME,
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
from mostt_model_builder.domain.contract.derived_rows import (
    SHARD_REQUEST_FILE_NAME,
    DeviceRow,
    ShardedRun,
    StepTemplate,
    derive_rows,
    render_shard_request,
)
from mostt_model_builder.domain.contract.recipe_contract import (
    JudgedCandidateFile,
    Parity,
    StepOutputOrigin,
)
from mostt_model_builder.domain.formats.file_formats import FileFormat
from mostt_model_builder.domain.onnx.graph_rules import GraphRuleError
from mostt_model_builder.domain.onnx.inspection_report import (
    INSPECTION_REPORT_FILE_NAME,
    FileInspection,
    InspectionReport,
    NodeCount,
    NodeScope,
    Opset,
    render_inspection_report,
)
from mostt_model_builder.domain.onnx.node_placement import (
    PLACEMENT_FILE_NAME,
    PLACEMENT_MAX_SIZE_BYTES,
    NodePlacementError,
    PlacedNodes,
    PlacementSummary,
    render_placement_summary,
)
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

GPU_IMAGE = ContainerImage(RECIPE_NAME, "bench-gpu", Device.GPU, ENVIRONMENT_PYTHON)
PROFILE_OUTPUT = DeclaredOutput("profile.json", 4096)
TRANSCRIBE_TEMPLATE = StepTemplate(
    name="transcribe",
    entry_module="mostt_recipe_family.steps.transcribe",
    sources=(),
    earlier_outputs=(),
    binds_candidate=True,
    outputs=(
        DeclaredOutput(RUN_LINES_FILE_NAME, RUN_LINES_MAX_SIZE_BYTES),
        PROFILE_OUTPUT,
        PEAKS_OUTPUT,
    ),
    devices={
        Device.GPU: DeviceRow(
            image=GPU_IMAGE,
            limits=StepLimits(
                memory_mib=4096, process_count=64, cpu_count=2, temporary_directory_mib=64
            ),
            outputs=(DeclaredOutput(PLACEMENT_FILE_NAME, PLACEMENT_MAX_SIZE_BYTES),),
        )
    },
)
FLEURS_RUN = ShardedRun("fleurs", "int8", Device.GPU, ("fr_fr",), shard_count=2)
SHARD_ROW = derive_rows(TRANSCRIBE_TEMPLATE, (FLEURS_RUN,))[1]
SHARD_DIRECTORY = WORK_ROOT / RUN_DIRECTORY / SHARD_ROW.row.name
VOCABULARY_NAME = "vocab.txt"
VOCABULARY = b"<blk>\nbon\njour\n"
ENCODER_NAME = "encoder.onnx"
ENCODER = b"\x08\x07\x3a\x00"
CANDIDATE_ROW = replace(
    EXPORT_ROW,
    outputs=(
        DeclaredOutput(ENCODER_NAME, 4096),
        DeclaredOutput(VOCABULARY_NAME, 4096),
        PEAKS_OUTPUT,
    ),
)
CANDIDATE_REQUEST = CandidateRequest(
    recipe_name=RECIPE_NAME,
    commit=COMMIT,
    work_root=WORK_ROOT,
    head_tree=HEAD_TREE,
    host=HOST,
    candidate=(
        JudgedCandidateFile(
            ENCODER_NAME, FileFormat.ONNX, ("int8",), StepOutputOrigin("export", ENCODER_NAME)
        ),
        JudgedCandidateFile(
            VOCABULARY_NAME,
            FileFormat.TEXT,
            ("int8",),
            StepOutputOrigin("export", VOCABULARY_NAME),
        ),
    ),
    rows=(CANDIDATE_ROW,),
    conversion_input=None,
)
ENCODER_INSPECTION = FileInspection(
    name=ENCODER_NAME,
    opsets=(Opset("", 17),),
    functions=(),
    nodes=(NodeCount(NodeScope.GRAPH, "", "MatMul", 1),),
    metadata_props_node_count=0,
    products=(),
    external_tensors=(),
    training_graph_count=0,
    sparse_initializer_count=0,
)
INSPECTION_REPORT = InspectionReport((ENCODER_INSPECTION,))
REFUSED_INSPECTION_REPORT = InspectionReport(
    (replace(ENCODER_INSPECTION, metadata_props_node_count=3),)
)
FRENCH_LINES = (
    b'{"language": "fr_fr", "utterance": "1001.wav", "token": 1, "decoding_time_ms": 12.5}\n'
    b'{"language": "fr_fr", "utterance": "1002.wav", "token": 2, "decoding_time_ms": 9.0}\n'
)
ENGLISH_LINES = (
    b'{"language": "en_us", "utterance": "1001.wav", "token": 1, "decoding_time_ms": 12.5}\n'
)
FRENCH_HYPOTHESES = (
    Hypothesis("fr_fr", "1001.wav", (1,), "bon", "bon", 0, 12.5),
    Hypothesis("fr_fr", "1002.wav", (2,), "jour", "jour", 0, 9.0),
)
GPU_PLACEMENT = PlacementSummary(
    (
        PlacedNodes("CPUExecutionProvider", "Shape", ("int64",), 4),
        PlacedNodes("CUDAExecutionProvider", "MatMul", ("float32",), 10),
    )
)
FALLEN_BACK_PLACEMENT = PlacementSummary(
    (PlacedNodes("CPUExecutionProvider", "MatMul", ("float32",), 3),)
)


def read_tokens_through_the_vocabulary(
    lines: bytes, candidate_texts: Mapping[str, bytes]
) -> tuple[Hypothesis, ...]:
    vocabulary = candidate_texts[VOCABULARY_NAME].decode().splitlines()
    return tuple(
        Hypothesis(
            language=str(line["language"]),
            utterance_id=str(line["utterance"]),
            token_ids=(int(str(line["token"])),),
            text=vocabulary[int(str(line["token"]))],
            text_without_unknown_tokens=vocabulary[int(str(line["token"]))],
            unknown_token_count=0,
            decoding_time_ms=float(str(line["decoding_time_ms"])),
        )
        for line in parse_run_lines(lines)
    )


SHARD_RUN = RecipeStepRun(
    row=SHARD_ROW.row,
    recipe_name=RECIPE_NAME,
    pins=(),
    commit=COMMIT,
    work_root=WORK_ROOT,
    head_tree=HEAD_TREE,
    host=HOST,
    shard=RunShard(
        SHARD_ROW.request,
        read_tokens_through_the_vocabulary,
        {VOCABULARY_NAME: VOCABULARY},
    ),
    candidate=CANDIDATE_REQUEST,
)


@dataclass(frozen=True)
class CompletedRecipeStep:
    outcome: RecipeStepOutcome
    runs: list[ContainerRun]
    files: FakeStepFiles


@dataclass(frozen=True)
class RefusedRecipeStep:
    run_act: Callable[[], RecipeStepOutcome]
    engine: FakeContainerEngine


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


def build_shard_engine(
    files: FakeStepFiles, lines: bytes, placement: PlacementSummary
) -> FakeContainerEngine:
    return FakeContainerEngine(
        files=files,
        images={GPU_IMAGE.tag: ImageFacts(IMAGE_ID, HEAD_TREE)},
        step_outputs={
            PLACEMENT_FILE_NAME: build_step_output(render_placement_summary(placement)),
            RUN_LINES_FILE_NAME: build_step_output(lines),
            PROFILE_OUTPUT.name: build_step_output(b"{}"),
            PEAKS_FILE_NAME: build_step_output(render_step_peaks(PEAKS)),
        },
    )


def build_inspected_files(
    inspection_report: InspectionReport | None = INSPECTION_REPORT,
) -> FakeStepFiles:
    files = FakeStepFiles()
    files.add_directory(WORK_ROOT)
    candidate_directory = WORK_ROOT / RUN_DIRECTORY / "candidate"
    files.add_directory(candidate_directory, build_directory_facts(0o750))
    for name, content in ((ENCODER_NAME, ENCODER), (VOCABULARY_NAME, VOCABULARY)):
        files.add_file(
            candidate_directory / name, build_file_facts(len(content), mode=0o444), content
        )
    if inspection_report is not None:
        inspect_directory = WORK_ROOT / RUN_DIRECTORY / INSPECT_STEP_NAME
        files.add_directory(inspect_directory, build_directory_facts(0o2770))
        files.add_file(
            inspect_directory / INSPECTION_REPORT_FILE_NAME,
            *build_step_output(render_inspection_report(inspection_report)),
        )
        files.add_file(
            inspect_directory / PEAKS_FILE_NAME, *build_step_output(render_step_peaks(PEAKS))
        )
    return files


def build_shard_act(
    lines: bytes,
    placement: PlacementSummary,
    files: FakeStepFiles | None = None,
    shard_run: RecipeStepRun = SHARD_RUN,
) -> Callable[[], RecipeStepOutcome]:
    files = files or build_inspected_files()
    return partial(run_recipe_step, shard_run, build_shard_engine(files, lines, placement), files)


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

        class TestWhenTheRowRunsAShardOfARun:
            @pytest.fixture
            def recipe_step(self) -> CompletedRecipeStep:
                files = build_inspected_files()
                engine = build_shard_engine(files, FRENCH_LINES, GPU_PLACEMENT)
                outcome = run_recipe_step(SHARD_RUN, engine, files)
                return CompletedRecipeStep(outcome, engine.runs, files)

            def test_should_write_the_request_of_the_shard_in_the_request_of_its_row(
                self, recipe_step: CompletedRecipeStep
            ) -> None:
                assert recipe_step.files.contents[
                    WORK_ROOT
                    / RUN_DIRECTORY
                    / "requests"
                    / SHARD_ROW.row.name
                    / SHARD_REQUEST_FILE_NAME
                ] == render_shard_request(SHARD_ROW.request)

            def test_should_give_back_the_hypotheses_that_the_reader_of_the_run_gives(
                self, recipe_step: CompletedRecipeStep
            ) -> None:
                assert recipe_step.outcome.hypotheses == FRENCH_HYPOTHESES

            def test_should_give_back_the_placement_of_the_nodes(
                self, recipe_step: CompletedRecipeStep
            ) -> None:
                assert recipe_step.outcome.placement == GPU_PLACEMENT

            def test_should_read_the_inspection_then_the_placement_the_lines_and_the_peaks_alone(
                self, recipe_step: CompletedRecipeStep
            ) -> None:
                assert recipe_step.files.read_paths == [
                    WORK_ROOT / RUN_DIRECTORY / INSPECT_STEP_NAME / INSPECTION_REPORT_FILE_NAME,
                    WORK_ROOT / RUN_DIRECTORY / INSPECT_STEP_NAME / PEAKS_FILE_NAME,
                    SHARD_DIRECTORY / PLACEMENT_FILE_NAME,
                    SHARD_DIRECTORY / RUN_LINES_FILE_NAME,
                    SHARD_DIRECTORY / PEAKS_FILE_NAME,
                ]

        class TestWhenTheShardPlacesAFloatingPointNodeOnTheCpu:
            @pytest.fixture
            def run_act(self) -> Callable[[], RecipeStepOutcome]:
                return build_shard_act(FRENCH_LINES, FALLEN_BACK_PLACEMENT)

            def test_should_refuse_it_since_a_part_of_the_model_fell_back_to_the_cpu(
                self, run_act: Callable[[], RecipeStepOutcome]
            ) -> None:
                with pytest.raises(
                    NodePlacementError,
                    match=re.escape("nodes MatMul run on the CPU with float32 tensors, 3 of them"),
                ):
                    run_act()

        class TestWhenTheReaderGivesAHypothesisInALanguageOutsideTheRun:
            @pytest.fixture
            def run_act(self) -> Callable[[], RecipeStepOutcome]:
                return build_shard_act(ENGLISH_LINES, GPU_PLACEMENT)

            def test_should_refuse_it_since_the_run_transcribes_its_own_languages_alone(
                self, run_act: Callable[[], RecipeStepOutcome]
            ) -> None:
                with pytest.raises(
                    HypothesisError,
                    match=re.escape(
                        "the hypothesis of '1001.wav' is in 'en_us', not a language of its run"
                    ),
                ):
                    run_act()

        class TestWhenTheRowBindsACandidateThatItsInspectionRefused:
            @pytest.fixture
            def refused_step(self) -> RefusedRecipeStep:
                files = build_inspected_files(REFUSED_INSPECTION_REPORT)
                engine = build_shard_engine(files, FRENCH_LINES, GPU_PLACEMENT)
                return RefusedRecipeStep(partial(run_recipe_step, SHARD_RUN, engine, files), engine)

            def test_should_refuse_it_before_any_container_starts(
                self, refused_step: RefusedRecipeStep
            ) -> None:
                with pytest.raises(
                    GraphRuleError,
                    match=re.escape(
                        f"the graph of {ENCODER_NAME} has 3 nodes that hold metadata_props"
                    ),
                ):
                    refused_step.run_act()
                assert refused_step.engine.runs == []

        class TestWhenTheRowBindsACandidateThatWasNeverInspected:
            @pytest.fixture
            def run_act(self) -> Callable[[], RecipeStepOutcome]:
                return build_shard_act(
                    FRENCH_LINES, GPU_PLACEMENT, build_inspected_files(inspection_report=None)
                )

            def test_should_refuse_it_and_name_step_inspect(
                self, run_act: Callable[[], RecipeStepOutcome]
            ) -> None:
                with pytest.raises(
                    ContainerFileError,
                    match=re.escape(f"{RUN_DIRECTORY}/{INSPECT_STEP_NAME} is missing"),
                ):
                    run_act()

        class TestWhenTheRunGivesNoCandidateForARowThatBindsIt:
            @pytest.fixture
            def run_act(self) -> Callable[[], RecipeStepOutcome]:
                return build_shard_act(
                    FRENCH_LINES, GPU_PLACEMENT, shard_run=replace(SHARD_RUN, candidate=None)
                )

            def test_should_refuse_it_since_only_an_inspected_candidate_reaches_a_run(
                self, run_act: Callable[[], RecipeStepOutcome]
            ) -> None:
                with pytest.raises(
                    CandidateError,
                    match=re.escape(
                        f"the step {SHARD_ROW.row.name} reads the candidate, and was given no "
                        "request of it"
                    ),
                ):
                    run_act()
