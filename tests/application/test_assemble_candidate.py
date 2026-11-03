import contextlib
import hashlib
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

from mostt_model_builder.application.assemble_candidate import (
    AssembledCandidate,
    CandidateCopy,
    CandidateError,
    CandidateRequest,
    assemble_candidate,
    read_candidate_texts,
)
from mostt_model_builder.application.run_step import HostIdentity, ImageFacts
from mostt_model_builder.domain.containers.account_ranges import HostAccounts
from mostt_model_builder.domain.containers.base_images import ENVIRONMENT_PYTHON
from mostt_model_builder.domain.containers.container_files import (
    ContainerFileError,
    FileFacts,
    FileKind,
)
from mostt_model_builder.domain.containers.container_steps import (
    CANDIDATE_IN_THE_CONTAINER,
    CONTAINER_USER_ID,
    INSPECT_ROW,
    ONNX_INSPECT_IMAGE,
    PEAKS_FILE_NAME,
    PEAKS_OUTPUT,
    BindMount,
    ContainerImage,
    ContainerRun,
    DeclaredOutput,
    Device,
    StepLimits,
    StepRow,
)
from mostt_model_builder.domain.containers.step_peaks import StepPeaks, render_step_peaks
from mostt_model_builder.domain.contract.recipe_contract import (
    HostContentOrigin,
    JudgedCandidateFile,
    StepOutputOrigin,
)
from mostt_model_builder.domain.formats.external_data import ExternalDataError
from mostt_model_builder.domain.formats.file_formats import FileFormat
from mostt_model_builder.domain.onnx.graph_rules import GraphRuleError
from mostt_model_builder.domain.onnx.inspection_report import (
    INSPECTION_REPORT_FILE_NAME,
    INSPECTION_REQUEST_FILE_NAME,
    ExternalTensor,
    FileInspection,
    InspectionReport,
    MatrixProduct,
    NodeCount,
    NodeScope,
    Opset,
    WeightSource,
    parse_inspection_request,
    render_inspection_report,
)

WORK_ROOT = PurePosixPath("/data/builds/work")
HEAD_TREE = "3f1d1ffe79bc3336eb583331cfbf9b189fd45f01"
COMMIT = "0123456789abcdef0123456789abcdef01234567"
IMAGE_ID = "sha256:" + "a" * 64
RECIPE_NAME = "family"
RUN_DIRECTORY = PurePosixPath("runs/family", COMMIT)
CANDIDATE_DIRECTORY = WORK_ROOT / RUN_DIRECTORY / "candidate"
INSPECTION_REQUEST_PATH = (
    WORK_ROOT / RUN_DIRECTORY / "requests/inspect" / INSPECTION_REQUEST_FILE_NAME
)
EXPORT_STEP = "export"
DESCRIBE_STEP = "describe"
ENCODER_NAME = "encoder-model.onnx"
ENCODER_DATA_NAME = "encoder-model.onnx.data"
DECODER_NAME = "decoder-model.onnx"
VOCABULARY_NAME = "vocab.txt"
CONFIGURATION_NAME = "config.json"
STRAY_NAME = "stray.bin"
# The smallest models the structural check accepts: an ir_version, then an empty graph.
ENCODER = b"\x08\x07\x3a\x00"
DECODER = b"\x08\x08\x3a\x00"
TENSOR_SIZE_BYTES = 2048
ENCODER_DATA = bytes(2 * TENSOR_SIZE_BYTES)
VOCABULARY = b"<blk>\nbon\njour\n"
CONFIGURATION = b'{"features": 128}\n'
OUTPUT_BOUND_BYTES = 1024
ENCODER_DATA_BOUND_BYTES = 8192
VARIANTS = ("float32",)
PEAKS = StepPeaks(memory_peak_bytes=96 * 2**20, process_count_peak=4)
LIMITS = StepLimits(memory_mib=1024, process_count=64, cpu_count=2, temporary_directory_mib=64)
EXPORT_IMAGE = ContainerImage(RECIPE_NAME, "export", Device.CPU, ENVIRONMENT_PYTHON)
EXPORT_ROW = StepRow(
    name=EXPORT_STEP,
    image=EXPORT_IMAGE,
    entry_module="mostt_recipe_family.steps.export",
    sources=(),
    has_request=False,
    outputs=(
        DeclaredOutput(ENCODER_NAME, OUTPUT_BOUND_BYTES),
        DeclaredOutput(ENCODER_DATA_NAME, ENCODER_DATA_BOUND_BYTES),
        DeclaredOutput(DECODER_NAME, OUTPUT_BOUND_BYTES),
        PEAKS_OUTPUT,
    ),
    limits=LIMITS,
)
DESCRIBE_ROW = StepRow(
    name=DESCRIBE_STEP,
    image=EXPORT_IMAGE,
    entry_module="mostt_recipe_family.steps.describe",
    sources=(),
    has_request=False,
    outputs=(DeclaredOutput(VOCABULARY_NAME, OUTPUT_BOUND_BYTES), PEAKS_OUTPUT),
    limits=LIMITS,
)
CANDIDATE = (
    JudgedCandidateFile(
        ENCODER_NAME, FileFormat.ONNX, VARIANTS, StepOutputOrigin(EXPORT_STEP, ENCODER_NAME)
    ),
    JudgedCandidateFile(
        ENCODER_DATA_NAME,
        FileFormat.ONNX_EXTERNAL_DATA,
        VARIANTS,
        StepOutputOrigin(EXPORT_STEP, ENCODER_DATA_NAME),
        external_data_of=ENCODER_NAME,
    ),
    JudgedCandidateFile(
        DECODER_NAME, FileFormat.ONNX, VARIANTS, StepOutputOrigin(EXPORT_STEP, DECODER_NAME)
    ),
    JudgedCandidateFile(
        VOCABULARY_NAME,
        FileFormat.TEXT,
        VARIANTS,
        StepOutputOrigin(DESCRIBE_STEP, VOCABULARY_NAME),
    ),
    JudgedCandidateFile(
        CONFIGURATION_NAME, FileFormat.TEXT, VARIANTS, HostContentOrigin(CONFIGURATION)
    ),
)
CANDIDATE_CONTENTS = {
    ENCODER_NAME: ENCODER,
    ENCODER_DATA_NAME: ENCODER_DATA,
    DECODER_NAME: DECODER,
    VOCABULARY_NAME: VOCABULARY,
    CONFIGURATION_NAME: CONFIGURATION,
}
HOST = HostIdentity(USER_ID, GROUP_ID, 20, HostAccounts(False, (), ()))
REQUEST = CandidateRequest(
    recipe_name=RECIPE_NAME,
    commit=COMMIT,
    work_root=WORK_ROOT,
    head_tree=HEAD_TREE,
    host=HOST,
    candidate=CANDIDATE,
    rows=(EXPORT_ROW, DESCRIBE_ROW),
    conversion_input=StepOutputOrigin(EXPORT_STEP, ENCODER_NAME),
)
STEP_OUTPUTS = {
    EXPORT_STEP: {
        ENCODER_NAME: build_step_output(ENCODER),
        ENCODER_DATA_NAME: build_step_output(ENCODER_DATA),
        DECODER_NAME: build_step_output(DECODER),
        PEAKS_FILE_NAME: build_step_output(render_step_peaks(PEAKS)),
    },
    DESCRIBE_STEP: {
        VOCABULARY_NAME: build_step_output(VOCABULARY),
        PEAKS_FILE_NAME: build_step_output(render_step_peaks(PEAKS)),
    },
}
PLAIN_PRODUCT = MatrixProduct(NodeScope.GRAPH, "MatMul", (), WeightSource.INITIALIZER, 2)
TRANSPOSED_PRODUCT = MatrixProduct(
    NodeScope.GRAPH, "MatMul", ("Transpose",), WeightSource.INITIALIZER, 2
)
ENCODER_INSPECTION = FileInspection(
    name=ENCODER_NAME,
    opsets=(Opset("", 17),),
    functions=(),
    nodes=(NodeCount(NodeScope.GRAPH, "", "MatMul", 1),),
    metadata_props_node_count=0,
    products=(PLAIN_PRODUCT,),
    external_tensors=(
        ExternalTensor(ENCODER_DATA_NAME, 0, TENSOR_SIZE_BYTES),
        ExternalTensor(ENCODER_DATA_NAME, TENSOR_SIZE_BYTES, TENSOR_SIZE_BYTES),
    ),
    training_graph_count=0,
    sparse_initializer_count=0,
)
DECODER_INSPECTION = FileInspection(
    name=DECODER_NAME,
    opsets=(Opset("", 17),),
    functions=(),
    nodes=(NodeCount(NodeScope.GRAPH, "", "MatMul", 1),),
    metadata_props_node_count=0,
    products=(TRANSPOSED_PRODUCT,),
    external_tensors=(),
    training_graph_count=0,
    sparse_initializer_count=0,
)
REPORT = InspectionReport((ENCODER_INSPECTION, DECODER_INSPECTION))
NOT_THE_HOSTS_COPY = (
    "the candidate file vocab.txt is not the host's own copy: a regular file of the user's, in "
    "mode 0444, with a single link, of at most 1024 bytes"
)


@dataclass(frozen=True)
class AssemblyRecord:
    assembled: AssembledCandidate
    runs: list[ContainerRun]
    files: FakeStepFiles


@dataclass(frozen=True)
class RefusedAssembly:
    assemble: Callable[[], AssembledCandidate]
    error: type[Exception]
    message: str


@dataclass(frozen=True)
class ReplacedOutput:
    step: str
    name: str
    facts: FileFacts
    content: bytes
    error: type[Exception]
    message: str


@dataclass(frozen=True)
class BrokenReport:
    report: InspectionReport
    error: type[Exception]
    message: str


@dataclass(frozen=True)
class TextReading:
    texts: Mapping[str, bytes]
    read_paths: list[PurePosixPath]


@dataclass(frozen=True)
class RefusedRead:
    read: Callable[[], Mapping[str, bytes]]
    message: str


@dataclass(frozen=True)
class EarlierInspection:
    assemble_act: Callable[[], AssembledCandidate]
    files: FakeStepFiles


def build_files(
    step_outputs: Mapping[str, Mapping[str, tuple[FileFacts, bytes]]] = STEP_OUTPUTS,
) -> FakeStepFiles:
    files = FakeStepFiles()
    files.add_directory(WORK_ROOT)
    for step, outputs in step_outputs.items():
        output_directory = WORK_ROOT / RUN_DIRECTORY / step
        files.add_directory(output_directory, build_directory_facts(0o2770))
        for name, (facts, content) in outputs.items():
            files.add_file(output_directory / name, facts, content)
    return files


def replace_output(
    step: str, name: str, facts: FileFacts, content: bytes
) -> Mapping[str, Mapping[str, tuple[FileFacts, bytes]]]:
    return {**STEP_OUTPUTS, step: {**STEP_OUTPUTS[step], name: (facts, content)}}


def build_engine(files: FakeStepFiles, report: InspectionReport = REPORT) -> FakeContainerEngine:
    return FakeContainerEngine(
        files=files,
        images={ONNX_INSPECT_IMAGE.tag: ImageFacts(IMAGE_ID, HEAD_TREE)},
        step_outputs={
            INSPECTION_REPORT_FILE_NAME: build_step_output(render_inspection_report(report)),
            PEAKS_FILE_NAME: build_step_output(render_step_peaks(PEAKS)),
        },
    )


def build_assembly_act(
    step_outputs: Mapping[str, Mapping[str, tuple[FileFacts, bytes]]] = STEP_OUTPUTS,
    report: InspectionReport = REPORT,
) -> Callable[[], AssembledCandidate]:
    files = build_files(step_outputs)
    return partial(assemble_candidate, REQUEST, build_engine(files, report), files)


def describe_copy(name: str, file_format: FileFormat) -> CandidateCopy:
    content = CANDIDATE_CONTENTS[name]
    return CandidateCopy(name, file_format, len(content), hashlib.sha256(content).hexdigest())


def build_candidate_files(vocabulary_facts: FileFacts | None) -> FakeStepFiles:
    files = FakeStepFiles()
    files.add_directory(WORK_ROOT)
    files.add_directory(CANDIDATE_DIRECTORY, build_directory_facts(0o750))
    for name in (ENCODER_NAME, CONFIGURATION_NAME):
        content = CANDIDATE_CONTENTS[name]
        files.add_file(CANDIDATE_DIRECTORY / name, build_file_facts(len(content), 0o444), content)
    if vocabulary_facts is not None:
        files.add_file(CANDIDATE_DIRECTORY / VOCABULARY_NAME, vocabulary_facts, VOCABULARY)
    return files


class TestTheCandidateAssembly:
    class TestTheAssembleCandidateFunction:
        class TestWhenEveryFileOfTheCandidateHoldsWhatItDeclares:
            @pytest.fixture
            def assembly(self) -> AssemblyRecord:
                files = build_files()
                engine = build_engine(files)
                return AssemblyRecord(
                    assemble_candidate(REQUEST, engine, files), engine.runs, files
                )

            def test_should_give_back_each_copy_with_its_format_size_and_sha256(
                self, assembly: AssemblyRecord
            ) -> None:
                assert assembly.assembled.files == tuple(
                    describe_copy(candidate_file.name, candidate_file.file_format)
                    for candidate_file in CANDIDATE
                )

            def test_should_copy_each_file_whole_into_the_candidate_of_the_run(
                self, assembly: AssemblyRecord
            ) -> None:
                assert {
                    name: assembly.files.contents.get(CANDIDATE_DIRECTORY / name)
                    for name in CANDIDATE_CONTENTS
                } == CANDIDATE_CONTENTS

            def test_should_leave_every_file_of_the_candidate_readable_and_writable_by_no_one(
                self, assembly: AssemblyRecord
            ) -> None:
                assert {
                    name: assembly.files.entries[CANDIDATE_DIRECTORY / name].permission_bits
                    for name in CANDIDATE_CONTENTS
                } == dict.fromkeys(CANDIDATE_CONTENTS, 0o444)

            def test_should_make_the_candidate_directory_closed_to_others(
                self, assembly: AssemblyRecord
            ) -> None:
                assert assembly.files.entries[CANDIDATE_DIRECTORY].permission_bits == 0o750

            def test_should_ask_the_inspection_for_the_onnx_files_in_the_candidates_order(
                self, assembly: AssemblyRecord
            ) -> None:
                assert parse_inspection_request(
                    assembly.files.contents[INSPECTION_REQUEST_PATH]
                ) == (ENCODER_NAME, DECODER_NAME)

            def test_should_run_the_inspection_alone(self, assembly: AssemblyRecord) -> None:
                assert [run.entry_module for run in assembly.runs] == [INSPECT_ROW.entry_module]

            def test_should_bind_the_candidate_read_only_into_the_inspection(
                self, assembly: AssemblyRecord
            ) -> None:
                assert BindMount(
                    RUN_DIRECTORY / "candidate",
                    CANDIDATE_IN_THE_CONTAINER,
                    is_read_only=True,
                    is_file=False,
                ) in [mount for run in assembly.runs for mount in run.mounts]

            def test_should_give_back_the_report_of_the_inspection(
                self, assembly: AssemblyRecord
            ) -> None:
                assert assembly.assembled.report == REPORT

            def test_should_keep_the_products_of_a_file_that_no_conversion_reads_as_they_are(
                self, assembly: AssemblyRecord
            ) -> None:
                assert assembly.assembled.report.files[1].products == (TRANSPOSED_PRODUCT,)

            def test_should_give_back_the_peaks_of_the_inspection(
                self, assembly: AssemblyRecord
            ) -> None:
                assert assembly.assembled.peaks == PEAKS

        class TestWhenTheCandidateAlreadyExists:
            @pytest.fixture
            def assemble_act(self) -> Callable[[], AssembledCandidate]:
                files = build_files()
                files.add_directory(CANDIDATE_DIRECTORY, build_directory_facts(0o750))
                return partial(assemble_candidate, REQUEST, build_engine(files), files)

            def test_should_refuse_to_overwrite_it_and_name_the_command_that_removes_it(
                self, assemble_act: Callable[[], AssembledCandidate]
            ) -> None:
                with pytest.raises(
                    CandidateError,
                    match=re.escape(
                        f"the candidate already exists in {RUN_DIRECTORY}/candidate: remove it "
                        "with ./model-build.sh --work <dir> --recipe <name> clean inspect"
                    ),
                ):
                    assemble_act()

        class TestWhenTheInspectionsOutputsRemainFromAnEarlierRun:
            @pytest.fixture
            def earlier_inspection(self) -> EarlierInspection:
                files = build_files()
                files.add_directory(
                    WORK_ROOT / RUN_DIRECTORY / "inspect", build_directory_facts(0o2770)
                )
                return EarlierInspection(
                    partial(assemble_candidate, REQUEST, build_engine(files), files), files
                )

            def test_should_refuse_it_before_copying_a_file(
                self, earlier_inspection: EarlierInspection
            ) -> None:
                with pytest.raises(
                    CandidateError,
                    match=re.escape(
                        f"the outputs of an earlier inspection remain in {RUN_DIRECTORY}/inspect: "
                        "remove them with ./model-build.sh --work <dir> --recipe <name> clean "
                        "inspect"
                    ),
                ):
                    earlier_inspection.assemble_act()
                assert CANDIDATE_DIRECTORY not in earlier_inspection.files.entries

        class TestWhenAnOutputOfAStepIsRefused:
            @pytest.fixture(
                params=[
                    pytest.param(
                        ReplacedOutput(
                            DESCRIBE_STEP,
                            VOCABULARY_NAME,
                            *build_step_output(b"PK\x03\x04" + VOCABULARY),
                            CandidateError,
                            "the candidate file vocab.txt begins as a zip archive",
                        ),
                        id="a file that begins as a zip archive, whatever its format declares",
                    ),
                    pytest.param(
                        ReplacedOutput(
                            EXPORT_STEP,
                            ENCODER_DATA_NAME,
                            *build_step_output(b"\x80\x04\x95" + ENCODER_DATA[3:]),
                            CandidateError,
                            "the candidate file encoder-model.onnx.data begins as a binary pickle",
                        ),
                        id="external data that begins as a pickle, though no format checks it",
                    ),
                    pytest.param(
                        ReplacedOutput(
                            EXPORT_STEP,
                            ENCODER_NAME,
                            *build_step_output(b"\x3a\x00"),
                            CandidateError,
                            "the candidate file encoder-model.onnx is not onnx: the model has "
                            "no ir_version (field 1)",
                        ),
                        id="an onnx file that fails the structural check",
                    ),
                    pytest.param(
                        ReplacedOutput(
                            DESCRIBE_STEP,
                            VOCABULARY_NAME,
                            *build_step_output(b"bon\xffjour\n"),
                            CandidateError,
                            "the candidate file vocab.txt is not text: the text is not valid UTF-8",
                        ),
                        id="a text file that is not UTF-8",
                    ),
                    pytest.param(
                        ReplacedOutput(
                            DESCRIBE_STEP,
                            VOCABULARY_NAME,
                            FileFacts(FileKind.LINK, CONTAINER_USER_ID, 0o777, 9, 1),
                            b"",
                            ContainerFileError,
                            "the output vocab.txt is a link, not a regular file",
                        ),
                        id="an output that is a link, which the copy does not follow",
                    ),
                    pytest.param(
                        ReplacedOutput(
                            DESCRIBE_STEP,
                            VOCABULARY_NAME,
                            build_file_facts(len(VOCABULARY), 0o640, USER_ID),
                            VOCABULARY,
                            ContainerFileError,
                            f"the output vocab.txt is owned by the user id {USER_ID}, not by "
                            f"{CONTAINER_USER_ID}",
                        ),
                        id="an output that the container's user did not write",
                    ),
                    pytest.param(
                        ReplacedOutput(
                            DESCRIBE_STEP,
                            VOCABULARY_NAME,
                            build_file_facts(len(VOCABULARY), 0o644, CONTAINER_USER_ID),
                            VOCABULARY,
                            ContainerFileError,
                            "the output vocab.txt has the mode 0644, not 0640",
                        ),
                        id="an output in another mode than the steps leave",
                    ),
                    pytest.param(
                        ReplacedOutput(
                            DESCRIBE_STEP,
                            VOCABULARY_NAME,
                            build_file_facts(OUTPUT_BOUND_BYTES + 1, 0o640, CONTAINER_USER_ID),
                            VOCABULARY,
                            ContainerFileError,
                            f"the output vocab.txt holds {OUTPUT_BOUND_BYTES + 1} bytes, beyond "
                            f"its bound of {OUTPUT_BOUND_BYTES}",
                        ),
                        id="an output beyond its bound, which the copy does not read",
                    ),
                    pytest.param(
                        ReplacedOutput(
                            DESCRIBE_STEP,
                            STRAY_NAME,
                            *build_step_output(b"stray"),
                            ContainerFileError,
                            "the step wrote 'stray.bin', which its row does not declare",
                        ),
                        id="an entry of the step that its row does not declare",
                    ),
                    pytest.param(
                        ReplacedOutput(
                            DESCRIBE_STEP,
                            VOCABULARY_NAME,
                            build_file_facts(len(VOCABULARY) + 4, 0o640, CONTAINER_USER_ID),
                            VOCABULARY,
                            CandidateError,
                            f"vocab.txt of describe held {len(VOCABULARY) + 4} bytes, and "
                            f"{len(VOCABULARY)} were copied",
                        ),
                        id="an output that shrinks as it is copied",
                    ),
                ]
            )
            def refused_assembly(self, request: pytest.FixtureRequest) -> RefusedAssembly:
                replaced: ReplacedOutput = request.param
                assemble = build_assembly_act(
                    replace_output(replaced.step, replaced.name, replaced.facts, replaced.content)
                )
                return RefusedAssembly(assemble, replaced.error, replaced.message)

            def test_should_refuse_the_candidate_and_say_why(
                self, refused_assembly: RefusedAssembly
            ) -> None:
                with pytest.raises(
                    refused_assembly.error, match=re.escape(refused_assembly.message)
                ):
                    refused_assembly.assemble()

        class TestWhenAnOutputOfAnotherOwnerIsRefused:
            @pytest.fixture
            def candidate_entries(self) -> list[PurePosixPath]:
                files = build_files(
                    replace_output(
                        EXPORT_STEP,
                        ENCODER_NAME,
                        build_file_facts(len(ENCODER), 0o640, USER_ID + 1),
                        ENCODER,
                    )
                )
                with contextlib.suppress(ContainerFileError):
                    assemble_candidate(REQUEST, build_engine(files), files)
                return [path for path in files.entries if path.parent == CANDIDATE_DIRECTORY]

            def test_should_copy_none_of_it_since_its_facts_are_judged_before_its_bytes(
                self, candidate_entries: list[PurePosixPath]
            ) -> None:
                assert CANDIDATE_DIRECTORY / ENCODER_NAME not in candidate_entries

        class TestWhenTheInspectionReportsAGraphThatBreaksARule:
            @pytest.fixture(
                params=[
                    pytest.param(
                        BrokenReport(
                            InspectionReport(
                                (
                                    replace(ENCODER_INSPECTION, metadata_props_node_count=1),
                                    DECODER_INSPECTION,
                                )
                            ),
                            GraphRuleError,
                            "the graph of encoder-model.onnx has 1 nodes that hold metadata_props",
                        ),
                        id="a node that holds metadata_props",
                    ),
                    pytest.param(
                        BrokenReport(
                            InspectionReport(
                                (
                                    replace(ENCODER_INSPECTION, products=(TRANSPOSED_PRODUCT,)),
                                    DECODER_INSPECTION,
                                )
                            ),
                            GraphRuleError,
                            "the graph of encoder-model.onnx, the conversions' input, has a "
                            "MatMul whose constant weight is not a plain 2-D initializer: it reads "
                            "an initializer of rank 2 through Transpose",
                        ),
                        id="a product of the conversions' input that is not plain",
                    ),
                    pytest.param(
                        BrokenReport(
                            InspectionReport(
                                (
                                    ENCODER_INSPECTION,
                                    replace(
                                        DECODER_INSPECTION,
                                        external_tensors=(
                                            ExternalTensor(ENCODER_DATA_NAME, 0, TENSOR_SIZE_BYTES),
                                        ),
                                    ),
                                )
                            ),
                            GraphRuleError,
                            "the graph of decoder-model.onnx reads a tensor from "
                            "encoder-model.onnx.data, which keeps every tensor inside it",
                        ),
                        id="external data that another onnx file names",
                    ),
                    pytest.param(
                        BrokenReport(
                            InspectionReport(
                                (
                                    replace(
                                        ENCODER_INSPECTION,
                                        external_tensors=(
                                            ExternalTensor(ENCODER_DATA_NAME, 0, 1024),
                                            ExternalTensor(
                                                ENCODER_DATA_NAME, TENSOR_SIZE_BYTES, 2048
                                            ),
                                        ),
                                    ),
                                    DECODER_INSPECTION,
                                )
                            ),
                            ExternalDataError,
                            "the external data encoder-model.onnx.data has a gap of 1024 bytes "
                            "at byte 1024",
                        ),
                        id="external data with a gap between two tensors",
                    ),
                    pytest.param(
                        BrokenReport(
                            InspectionReport(
                                (
                                    replace(
                                        ENCODER_INSPECTION,
                                        external_tensors=(
                                            ExternalTensor(ENCODER_DATA_NAME, 0, TENSOR_SIZE_BYTES),
                                        ),
                                    ),
                                    DECODER_INSPECTION,
                                )
                            ),
                            ExternalDataError,
                            "the external data encoder-model.onnx.data holds 2048 bytes after "
                            "its last tensor",
                        ),
                        id="external data padded after its last tensor",
                    ),
                ]
            )
            def refused_assembly(self, request: pytest.FixtureRequest) -> RefusedAssembly:
                broken: BrokenReport = request.param
                return RefusedAssembly(
                    build_assembly_act(report=broken.report), broken.error, broken.message
                )

            def test_should_refuse_the_candidate_and_say_why(
                self, refused_assembly: RefusedAssembly
            ) -> None:
                with pytest.raises(
                    refused_assembly.error, match=re.escape(refused_assembly.message)
                ):
                    refused_assembly.assemble()

    class TestTheReadCandidateTextsFunction:
        class TestWhenTheCandidateHoldsTheHostsCopies:
            @pytest.fixture
            def text_reading(self) -> TextReading:
                files = build_candidate_files(build_file_facts(len(VOCABULARY), 0o444))
                return TextReading(read_candidate_texts(REQUEST, files), files.read_paths)

            def test_should_give_the_text_files_by_name(self, text_reading: TextReading) -> None:
                assert text_reading.texts == {
                    VOCABULARY_NAME: VOCABULARY,
                    CONFIGURATION_NAME: CONFIGURATION,
                }

            def test_should_read_the_text_files_alone(self, text_reading: TextReading) -> None:
                assert text_reading.read_paths == [
                    CANDIDATE_DIRECTORY / VOCABULARY_NAME,
                    CANDIDATE_DIRECTORY / CONFIGURATION_NAME,
                ]

        class TestWhenATextFileIsMissing:
            @pytest.fixture
            def read_act(self) -> Callable[[], Mapping[str, bytes]]:
                return partial(read_candidate_texts, REQUEST, build_candidate_files(None))

            def test_should_refuse_it_and_name_the_command_that_assembles_the_candidate(
                self, read_act: Callable[[], Mapping[str, bytes]]
            ) -> None:
                with pytest.raises(
                    CandidateError,
                    match=re.escape(
                        "the candidate file vocab.txt is missing from the candidate, which the "
                        "step inspect assembles: run it with ./model-build.sh --work <dir> "
                        "--recipe <name> step inspect"
                    ),
                ):
                    read_act()

        class TestWhenATextFileIsNotTheHostsCopy:
            @pytest.fixture(
                params=[
                    pytest.param(
                        (build_file_facts(len(VOCABULARY), 0o644), NOT_THE_HOSTS_COPY),
                        id="a file in another mode than the copies",
                    ),
                    pytest.param(
                        (
                            build_file_facts(len(VOCABULARY), 0o444, CONTAINER_USER_ID),
                            NOT_THE_HOSTS_COPY,
                        ),
                        id="a file of another user",
                    ),
                    pytest.param(
                        (
                            FileFacts(FileKind.REGULAR, USER_ID, 0o444, len(VOCABULARY), 2),
                            NOT_THE_HOSTS_COPY,
                        ),
                        id="a file with two links",
                    ),
                    pytest.param(
                        (build_file_facts(OUTPUT_BOUND_BYTES + 1, 0o444), NOT_THE_HOSTS_COPY),
                        id="a file beyond the bound of the output it was copied from",
                    ),
                ]
            )
            def refused_read(self, request: pytest.FixtureRequest) -> RefusedRead:
                vocabulary_facts, message = request.param
                files = build_candidate_files(vocabulary_facts)
                return RefusedRead(partial(read_candidate_texts, REQUEST, files), message)

            def test_should_refuse_it_and_say_why(self, refused_read: RefusedRead) -> None:
                with pytest.raises(ContainerFileError, match=re.escape(refused_read.message)):
                    refused_read.read()
