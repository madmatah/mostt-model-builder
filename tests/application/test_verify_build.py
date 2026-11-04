import hashlib
import json
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from functools import partial
from pathlib import PurePosixPath

import pytest
from run_step_fakes import (
    GROUP_ID,
    USER_ID,
    FakeStepFiles,
    build_directory_facts,
    build_file_facts,
    build_step_output,
)

from mostt_model_builder.application.assemble_candidate import CandidateRequest
from mostt_model_builder.application.run_step import HostIdentity
from mostt_model_builder.application.verify_build import (
    BuildRejectedError,
    VerificationRequest,
    VerifiedBuild,
    refuse_rejected_build,
    verify_build,
)
from mostt_model_builder.domain.acceptance.acceptance import Comparison, Equivalence
from mostt_model_builder.domain.acceptance.parity_reports import (
    ParityReport,
    render_parity_report,
)
from mostt_model_builder.domain.acceptance.verification import (
    CandidateDigest,
    VerificationError,
)
from mostt_model_builder.domain.bench.hypotheses import (
    RUN_LINES_FILE_NAME,
    RUN_LINES_MAX_SIZE_BYTES,
    Hypothesis,
    parse_run_lines,
)
from mostt_model_builder.domain.containers.account_ranges import HostAccounts
from mostt_model_builder.domain.containers.base_images import ENVIRONMENT_PYTHON
from mostt_model_builder.domain.containers.container_files import (
    ContainerFileError,
    FileFacts,
    FileKind,
)
from mostt_model_builder.domain.containers.container_steps import (
    INSPECT_STEP_NAME,
    PEAKS_FILE_NAME,
    PEAKS_OUTPUT,
    ContainerImage,
    DeclaredOutput,
    Device,
    StepLimits,
    StepRow,
)
from mostt_model_builder.domain.containers.step_peaks import StepPeaks, render_step_peaks
from mostt_model_builder.domain.contract.derived_rows import (
    DeviceRow,
    ShardedRun,
    StepTemplate,
    derive_rows,
)
from mostt_model_builder.domain.contract.recipe_contract import (
    JudgedCandidateFile,
    JudgedRun,
    Parity,
    StepOutputOrigin,
)
from mostt_model_builder.domain.formats.file_formats import FileFormat
from mostt_model_builder.domain.onnx.graph_rules import GraphRuleError
from mostt_model_builder.domain.onnx.inspection_report import (
    INSPECTION_REPORT_FILE_NAME,
    FileInspection,
    InspectionReport,
    MatrixProduct,
    NodeCount,
    NodeScope,
    Opset,
    WeightSource,
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
FLEURS_REVISION = "70bb2e84b976b7e960aa89f1c648e09c59f894dd"
RECIPE_NAME = "family"
RUN_DIRECTORY = PurePosixPath("runs/family", COMMIT)
REPORT_PATH = RUN_DIRECTORY / "verification.json"
HOST = HostIdentity(USER_ID, GROUP_ID, 20, HostAccounts(False, (), ()))
LIMITS = StepLimits(memory_mib=1024, process_count=64, cpu_count=2, temporary_directory_mib=64)
PEAKS = render_step_peaks(StepPeaks(memory_peak_bytes=64 * 2**20, process_count_peak=6))
VOCABULARY = b"le\nchat\ndort\nici\nchat\n"
VOCABULARY_FILE = JudgedCandidateFile(
    "vocab.txt", FileFormat.TEXT, ("float32",), StepOutputOrigin("export", "vocab.txt")
)
ENCODER_NAME = "encoder-model.onnx"
ENCODER = b"\x08\x07\x3a\x00"
ENCODER_FILE = JudgedCandidateFile(
    ENCODER_NAME, FileFormat.ONNX, ("float32",), StepOutputOrigin("export", ENCODER_NAME)
)
ENCODER_INSPECTION = FileInspection(
    name=ENCODER_NAME,
    opsets=(Opset("", 17),),
    functions=(),
    nodes=(NodeCount(NodeScope.GRAPH, "", "MatMul", 1),),
    metadata_props_node_count=0,
    products=(MatrixProduct(NodeScope.GRAPH, "MatMul", (), WeightSource.INITIALIZER, 2),),
    external_tensors=(),
    training_graph_count=0,
    sparse_initializer_count=0,
)
INSPECTION_REPORT = InspectionReport((ENCODER_INSPECTION,))
REFUSED_INSPECTION_REPORT = InspectionReport(
    (replace(ENCODER_INSPECTION, metadata_props_node_count=3),)
)
INDEXES = {
    "fr_fr": b"1\t1.wav\tLe chat dort.\tle chat dort\tx\t16000\tF\n"
    b"2\t2.wav\tIci.\tici\tx\t16000\tM\n",
    "en_us": b"3\t3.wav\tThe cat.\tthe cat\tx\t16000\tF\n",
}
PARITY_OUTPUT = DeclaredOutput("encoder-parity.json", 4096)
ENCODER_PARITY = Parity("encoder-against-reference", "export", PARITY_OUTPUT.name, 5e-5)
EXPORT_ROW = StepRow(
    name="export",
    image=ContainerImage(RECIPE_NAME, "export", Device.CPU, ENVIRONMENT_PYTHON),
    entry_module="mostt_recipe_family.steps.export",
    sources=(),
    has_request=False,
    outputs=(
        DeclaredOutput("vocab.txt", 4096),
        DeclaredOutput(ENCODER_NAME, 4096),
        PARITY_OUTPUT,
        PEAKS_OUTPUT,
    ),
    limits=LIMITS,
)
LINES_OUTPUT = DeclaredOutput(RUN_LINES_FILE_NAME, RUN_LINES_MAX_SIZE_BYTES)
TEMPLATE = StepTemplate(
    name="transcribe",
    entry_module="mostt_recipe_family.steps.transcribe",
    sources=(),
    earlier_outputs=(),
    binds_candidate=True,
    outputs=(PEAKS_OUTPUT,),
    devices={
        Device.CPU: DeviceRow(
            ContainerImage(RECIPE_NAME, "transcribe-cpu", Device.CPU, ENVIRONMENT_PYTHON),
            LIMITS,
            (LINES_OUTPUT,),
        ),
        Device.GPU: DeviceRow(
            ContainerImage(RECIPE_NAME, "transcribe-gpu", Device.GPU, ENVIRONMENT_PYTHON),
            LIMITS,
            (LINES_OUTPUT, DeclaredOutput(PLACEMENT_FILE_NAME, PLACEMENT_MAX_SIZE_BYTES)),
        ),
    },
)
LANGUAGES = ("fr_fr", "en_us")
DERIVED_ROWS = derive_rows(
    TEMPLATE,
    (
        ShardedRun("float32-cpu", "float32", Device.CPU, LANGUAGES, shard_count=2),
        ShardedRun("float32-gpu", "float32", Device.GPU, LANGUAGES, shard_count=1),
    ),
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


def read_vocabulary_lines(
    lines: bytes, candidate_texts: Mapping[str, bytes]
) -> tuple[Hypothesis, ...]:
    words = candidate_texts["vocab.txt"].decode().splitlines()
    hypotheses = []
    for line in parse_run_lines(lines):
        token_ids = tuple(int(str(token_id)) for token_id in str(line["tokens"]).split())
        text = " ".join(words[token_id] for token_id in token_ids)
        hypotheses.append(
            Hypothesis(
                str(line["language"]), str(line["utterance"]), token_ids, text, text, 0, 10.0
            )
        )
    return tuple(hypotheses)


RUNS = tuple(
    JudgedRun(name, "transcribe", device, "float32", LANGUAGES, shard_count, read_vocabulary_lines)
    for name, device, shard_count in (
        ("float32-cpu", Device.CPU, 2),
        ("float32-gpu", Device.GPU, 1),
    )
)
CROSS_DEVICE = Comparison("float32-gpu", "float32-cpu", Equivalence(0.5, 0.10))
SAME_TOKENS_EVERYWHERE = Comparison("float32-gpu", "float32-cpu", Equivalence(1.0, 0.10))


def render_lines(*records: tuple[str, str, str]) -> bytes:
    return b"".join(
        json.dumps({"language": language, "utterance": utterance, "tokens": tokens}).encode()
        + b"\n"
        for language, utterance, tokens in records
    )


SHARD_LINES = {
    "transcribe-float32-cpu-0": render_lines(("fr_fr", "1.wav", "0 1 2"), ("en_us", "3.wav", "1")),
    "transcribe-float32-cpu-1": render_lines(("fr_fr", "2.wav", "3")),
    "transcribe-float32-gpu-0": render_lines(
        ("fr_fr", "1.wav", "0 4 2"), ("fr_fr", "2.wav", "3"), ("en_us", "3.wav", "1")
    ),
}


def build_index_pin(language: str, content: bytes) -> SourcePin:
    return SourcePin(
        RepositoryType.DATASET,
        "google/fleurs",
        FLEURS_REVISION,
        f"data/{language}/test.tsv",
        len(content),
        hashlib.sha256(content).hexdigest(),
    )


PINS = tuple(build_index_pin(language, content) for language, content in INDEXES.items())


def build_files(
    placement: PlacementSummary = GPU_PLACEMENT,
    indexes: Mapping[str, bytes] = INDEXES,
    shard_lines: Mapping[str, bytes] = SHARD_LINES,
    inspection_report: InspectionReport | None = INSPECTION_REPORT,
    encoder_copy_mode: int = 0o444,
) -> FakeStepFiles:
    files = FakeStepFiles()
    files.add_directory(WORK_ROOT)
    for language, content in indexes.items():
        files.add_file(
            WORK_ROOT / f"sources/google/fleurs/{FLEURS_REVISION}/data/{language}/test.tsv",
            build_file_facts(len(content), mode=0o444),
            content,
        )
    candidate_directory = WORK_ROOT / RUN_DIRECTORY / "candidate"
    files.add_directory(candidate_directory, build_directory_facts(0o750))
    files.add_file(
        candidate_directory / "vocab.txt", build_file_facts(len(VOCABULARY), mode=0o444), VOCABULARY
    )
    files.add_file(
        candidate_directory / ENCODER_NAME,
        build_file_facts(len(ENCODER), mode=encoder_copy_mode),
        ENCODER,
    )
    if inspection_report is not None:
        inspect_directory = WORK_ROOT / RUN_DIRECTORY / INSPECT_STEP_NAME
        files.add_directory(inspect_directory, build_directory_facts(0o2770))
        files.add_file(
            inspect_directory / INSPECTION_REPORT_FILE_NAME,
            *build_step_output(render_inspection_report(inspection_report)),
        )
        files.add_file(inspect_directory / PEAKS_FILE_NAME, *build_step_output(PEAKS))
    export_directory = WORK_ROOT / RUN_DIRECTORY / "export"
    files.add_directory(export_directory, build_directory_facts(0o2770))
    for name, content in (
        ("vocab.txt", VOCABULARY),
        (ENCODER_NAME, ENCODER),
        (PARITY_OUTPUT.name, render_parity_report(ParityReport(ENCODER_PARITY.name, 4e-6, 20))),
        (PEAKS_FILE_NAME, PEAKS),
    ):
        files.add_file(export_directory / name, *build_step_output(content))
    for row_name, lines in shard_lines.items():
        shard_directory = WORK_ROOT / RUN_DIRECTORY / row_name
        files.add_directory(shard_directory, build_directory_facts(0o2770))
        files.add_file(shard_directory / RUN_LINES_FILE_NAME, *build_step_output(lines))
        files.add_file(shard_directory / PEAKS_FILE_NAME, *build_step_output(PEAKS))
        if "gpu" in row_name:
            files.add_file(
                shard_directory / PLACEMENT_FILE_NAME,
                *build_step_output(render_placement_summary(placement)),
            )
    return files


def build_request(comparison: Comparison = CROSS_DEVICE) -> VerificationRequest:
    return VerificationRequest(
        candidate=CandidateRequest(
            recipe_name=RECIPE_NAME,
            commit=COMMIT,
            work_root=WORK_ROOT,
            head_tree=HEAD_TREE,
            host=HOST,
            candidate=(ENCODER_FILE, VOCABULARY_FILE),
            rows=(EXPORT_ROW, *(derived_row.row for derived_row in DERIVED_ROWS)),
            conversion_input=None,
        ),
        pins=PINS,
        runs=RUNS,
        derived_rows=DERIVED_ROWS,
        parities=(ENCODER_PARITY,),
        comparisons=(comparison,),
    )


@dataclass(frozen=True)
class CompletedVerification:
    verified: VerifiedBuild
    files: FakeStepFiles


@dataclass(frozen=True)
class RefusedVerification:
    verify: Callable[[], VerifiedBuild]
    message: str


CANDIDATE_DIRECTORY = WORK_ROOT / RUN_DIRECTORY / "candidate"
NOT_THE_HOSTS_COPY = (
    f"the candidate file {ENCODER_NAME} is not the host's own copy: a regular file of the user's, "
    "in mode 0444, with a single link, of at most 4096 bytes"
)


class TestTheBuildVerification:
    class TestTheVerifyBuildFunction:
        class TestWhenEveryRunAndParityIsInPlace:
            @pytest.fixture
            def verification(self) -> CompletedVerification:
                files = build_files()
                return CompletedVerification(verify_build(build_request(), files), files)

            def test_should_measure_each_run_across_its_shards_on_each_language(
                self, verification: CompletedVerification
            ) -> None:
                assert [
                    (measure.run, measure.language, measure.utterance_count, measure.error_count)
                    for measure in verification.verified.verification.runs
                ] == [
                    ("float32-cpu", "fr_fr", 2, 0),
                    ("float32-cpu", "en_us", 1, 2),
                    ("float32-gpu", "fr_fr", 2, 0),
                    ("float32-gpu", "en_us", 1, 2),
                ]

            def test_should_compare_the_runs_by_their_token_ids(
                self, verification: CompletedVerification
            ) -> None:
                assert [
                    (measure.language, measure.identical_count, measure.compared_count)
                    for measure in verification.verified.verification.comparisons
                ] == [("fr_fr", 1, 2), ("en_us", 1, 1)]

            def test_should_hold_each_parity_of_the_recipe(
                self, verification: CompletedVerification
            ) -> None:
                assert [
                    (measure.report.name, measure.failure)
                    for measure in verification.verified.verification.parities
                ] == [(ENCODER_PARITY.name, None)]

            def test_should_write_its_report_beside_the_outputs_of_the_run(
                self, verification: CompletedVerification
            ) -> None:
                assert verification.verified.report_path == REPORT_PATH

            def test_should_record_the_size_and_sha256_of_each_candidate_file_it_accepts(
                self, verification: CompletedVerification
            ) -> None:
                assert verification.verified.verification.candidate == (
                    CandidateDigest(
                        ENCODER_NAME,
                        FileFormat.ONNX,
                        ("float32",),
                        len(ENCODER),
                        hashlib.sha256(ENCODER).hexdigest(),
                    ),
                    CandidateDigest(
                        "vocab.txt",
                        FileFormat.TEXT,
                        ("float32",),
                        len(VOCABULARY),
                        hashlib.sha256(VOCABULARY).hexdigest(),
                    ),
                )

            def test_should_hash_each_file_through_the_candidate_directory_it_judged(
                self, verification: CompletedVerification
            ) -> None:
                assert verification.files.digested_paths == [
                    CANDIDATE_DIRECTORY / ENCODER_NAME,
                    CANDIDATE_DIRECTORY / "vocab.txt",
                ]

            def test_should_write_the_hashes_into_its_report(
                self, verification: CompletedVerification
            ) -> None:
                report = json.loads(verification.files.contents[WORK_ROOT / REPORT_PATH])
                assert [(digest["name"], digest["sha256"]) for digest in report["candidate"]] == [
                    (ENCODER_NAME, hashlib.sha256(ENCODER).hexdigest()),
                    ("vocab.txt", hashlib.sha256(VOCABULARY).hexdigest()),
                ]

            def test_should_record_in_its_report_that_the_build_is_accepted(
                self, verification: CompletedVerification
            ) -> None:
                report = json.loads(verification.files.contents[WORK_ROOT / REPORT_PATH])
                assert report["is_accepted"] is True

            def test_should_read_the_inspection_the_lines_the_placements_and_the_parity(
                self, verification: CompletedVerification
            ) -> None:
                assert sorted(
                    str(path.relative_to(WORK_ROOT / RUN_DIRECTORY))
                    for path in verification.files.read_paths
                    if path.name != PEAKS_FILE_NAME
                    and not path.is_relative_to(WORK_ROOT / RUN_DIRECTORY / "candidate")
                    and not path.is_relative_to(WORK_ROOT / "sources")
                ) == [
                    "export/encoder-parity.json",
                    "inspect/inspection.json",
                    "transcribe-float32-cpu-0/lines.jsonl",
                    "transcribe-float32-cpu-1/lines.jsonl",
                    "transcribe-float32-gpu-0/lines.jsonl",
                    "transcribe-float32-gpu-0/placement.json",
                ]

        class TestWhenTheInspectionRefusedAGraphOfTheCandidate:
            @pytest.fixture
            def verify_act(self) -> Callable[[], VerifiedBuild]:
                files = build_files(inspection_report=REFUSED_INSPECTION_REPORT)
                return partial(verify_build, build_request(), files)

            def test_should_refuse_the_build_since_its_inspection_refused_the_candidate(
                self, verify_act: Callable[[], VerifiedBuild]
            ) -> None:
                with pytest.raises(
                    GraphRuleError,
                    match=re.escape(
                        f"the graph of {ENCODER_NAME} has 3 nodes that hold metadata_props, which "
                        "may name paths of the container that exported it"
                    ),
                ):
                    verify_act()

        class TestWhenTheCandidateWasNeverInspected:
            @pytest.fixture
            def verify_act(self) -> Callable[[], VerifiedBuild]:
                return partial(verify_build, build_request(), build_files(inspection_report=None))

            def test_should_refuse_the_build_and_name_step_inspect(
                self, verify_act: Callable[[], VerifiedBuild]
            ) -> None:
                with pytest.raises(
                    ContainerFileError,
                    match=re.escape(f"{RUN_DIRECTORY}/{INSPECT_STEP_NAME} is missing"),
                ):
                    verify_act()

        class TestWhenAFileOfTheCandidateIsNotTheHostsCopy:
            @pytest.fixture
            def verify_act(self) -> Callable[[], VerifiedBuild]:
                return partial(verify_build, build_request(), build_files(encoder_copy_mode=0o644))

            def test_should_refuse_it_since_it_may_have_changed_since_its_inspection(
                self, verify_act: Callable[[], VerifiedBuild]
            ) -> None:
                with pytest.raises(
                    ContainerFileError,
                    match=re.escape(
                        f"the candidate file {ENCODER_NAME} is not the host's own copy: a regular "
                        "file of the user's, in mode 0444, with a single link, of at most 4096 "
                        "bytes"
                    ),
                ):
                    verify_act()

        class TestWhenACandidateFileIsSwappedBeforeItIsHashed:
            @pytest.fixture(
                params=[
                    pytest.param(
                        (FileFacts(FileKind.LINK, USER_ID, 0o777, 12, 1), NOT_THE_HOSTS_COPY),
                        id="a link",
                    ),
                    pytest.param(
                        (build_file_facts(len(ENCODER), 0o444, owner_id=0), NOT_THE_HOSTS_COPY),
                        id="a file of another user",
                    ),
                    pytest.param(
                        (build_file_facts(len(ENCODER), 0o644), NOT_THE_HOSTS_COPY),
                        id="a file in another mode than the copies",
                    ),
                    pytest.param(
                        (
                            FileFacts(FileKind.REGULAR, USER_ID, 0o444, len(ENCODER), 2),
                            NOT_THE_HOSTS_COPY,
                        ),
                        id="a file with a second link",
                    ),
                ]
            )
            def refused_verification(self, request: pytest.FixtureRequest) -> RefusedVerification:
                hashing_facts, message = request.param
                files = build_files()
                files.hashing_facts[CANDIDATE_DIRECTORY / ENCODER_NAME] = hashing_facts
                return RefusedVerification(partial(verify_build, build_request(), files), message)

            def test_should_refuse_it_by_the_facts_of_the_descriptor_it_hashes(
                self, refused_verification: RefusedVerification
            ) -> None:
                with pytest.raises(
                    ContainerFileError, match=re.escape(refused_verification.message)
                ):
                    refused_verification.verify()

        class TestWhenAComparisonBreaksItsRule:
            @pytest.fixture
            def refuse_act(self) -> Callable[[], None]:
                verified = verify_build(build_request(SAME_TOKENS_EVERYWHERE), build_files())
                return partial(refuse_rejected_build, verified)

            def test_should_refuse_the_build_and_name_its_report(
                self, refuse_act: Callable[[], None]
            ) -> None:
                with pytest.raises(
                    BuildRejectedError,
                    match=re.escape(
                        f"the build fails 1 of its rules, as {REPORT_PATH} records: float32-gpu "
                        "against float32-cpu, fr_fr: 1 of 2 utterances have identical token ids"
                    ),
                ):
                    refuse_act()

        class TestWhenAnIndexOfTheBenchNoLongerMatchesItsPin:
            @pytest.fixture
            def verify_act(self) -> Callable[[], VerifiedBuild]:
                files = build_files(indexes={**INDEXES, "en_us": INDEXES["en_us"][:-1] + b"X"})
                return partial(verify_build, build_request(), files)

            def test_should_refuse_to_measure_against_it(
                self, verify_act: Callable[[], VerifiedBuild]
            ) -> None:
                with pytest.raises(
                    VerificationError,
                    match=re.escape(
                        f"sources/google/fleurs/{FLEURS_REVISION}/data/en_us/test.tsv no longer "
                        "matches its pin"
                    ),
                ):
                    verify_act()

        class TestWhenAShardOfARunNeverRan:
            @pytest.fixture
            def verify_act(self) -> Callable[[], VerifiedBuild]:
                shard_lines = {
                    name: lines
                    for name, lines in SHARD_LINES.items()
                    if name != "transcribe-float32-cpu-1"
                }
                return partial(verify_build, build_request(), build_files(shard_lines=shard_lines))

            def test_should_refuse_its_missing_outputs(
                self, verify_act: Callable[[], VerifiedBuild]
            ) -> None:
                with pytest.raises(
                    ContainerFileError,
                    match=re.escape("transcribe-float32-cpu-1 is missing"),
                ):
                    verify_act()

        class TestWhenARunOnTheGpuFellBackToTheCpu:
            @pytest.fixture
            def verify_act(self) -> Callable[[], VerifiedBuild]:
                return partial(
                    verify_build, build_request(), build_files(placement=FALLEN_BACK_PLACEMENT)
                )

            def test_should_refuse_its_placement(
                self, verify_act: Callable[[], VerifiedBuild]
            ) -> None:
                with pytest.raises(NodePlacementError, match="MatMul"):
                    verify_act()

        class TestWhenAParityExceedsItsTolerance:
            @pytest.fixture
            def refuse_act(self) -> Callable[[], None]:
                request = replace(
                    build_request(),
                    parities=(replace(ENCODER_PARITY, absolute_difference_tolerance=1e-6),),
                )
                return partial(refuse_rejected_build, verify_build(request, build_files()))

            def test_should_refuse_the_build(self, refuse_act: Callable[[], None]) -> None:
                with pytest.raises(
                    BuildRejectedError,
                    match=re.escape(
                        "the parity encoder-against-reference: the maximum absolute difference, "
                        "4e-06, exceeds the tolerance of 1e-06"
                    ),
                ):
                    refuse_act()
