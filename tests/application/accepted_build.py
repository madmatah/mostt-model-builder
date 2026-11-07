import hashlib
import json
from collections.abc import Mapping
from dataclasses import replace
from pathlib import PurePosixPath

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
    VerificationRequest,
)
from mostt_model_builder.domain.acceptance.acceptance import Comparison, Equivalence
from mostt_model_builder.domain.acceptance.parity_reports import (
    ParityReport,
    render_parity_report,
)
from mostt_model_builder.domain.bench.bench_languages import BENCH_LANGUAGES
from mostt_model_builder.domain.bench.hypotheses import Hypothesis, parse_run_lines
from mostt_model_builder.domain.bench.run_lines import RUN_LINES_FILE_NAME, RUN_LINES_MAX_SIZE_BYTES
from mostt_model_builder.domain.containers.account_ranges import HostAccounts
from mostt_model_builder.domain.containers.base_images import ENVIRONMENT_PYTHON
from mostt_model_builder.domain.containers.container_steps import (
    INSPECT_STEP_NAME,
    PEAKS_OUTPUT,
    ContainerImage,
    DeclaredOutput,
    StepLimits,
    StepRow,
)
from mostt_model_builder.domain.containers.devices import Device
from mostt_model_builder.domain.containers.step_peaks import (
    PEAKS_FILE_NAME,
    StepPeaks,
    render_step_peaks,
)
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
    PlacedNodes,
    PlacementSummary,
    render_placement_summary,
)
from mostt_model_builder.domain.sources.source_pins import RepositoryType, SourcePin

WORK_ROOT = PurePosixPath("/data/builds/work")
HEAD_TREE = "3f1d1ffe79bc3336eb583331cfbf9b189fd45f01"
FLEURS_REVISION = "70bb2e84b976b7e960aa89f1c648e09c59f894dd"
RECIPE_NAME = "family"
ENTRIES_DIRECTORY = PurePosixPath("cache/family")
BUILD_DIRECTORY = PurePosixPath("builds/family")
REPORT_PATH = BUILD_DIRECTORY / "verification.json"
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
INDEXES_OF_THE_RUNS = {
    "fr_fr": b"1\t1.wav\tLe chat dort.\tle chat dort\tx\t16000\tF\n"
    b"2\t2.wav\tIci.\tici\tx\t16000\tM\n",
    "en_us": b"3\t3.wav\tThe cat.\tthe cat\tx\t16000\tF\n4\t4.wav\tHere.\there\tx\t16000\tM\n",
}
INDEX_THAT_NO_RUN_READS = b"5\t5.wav\tUnread.\tunread\tx\t16000\tF\n"
INDEXES = {
    language: INDEXES_OF_THE_RUNS.get(language, INDEX_THAT_NO_RUN_READS)
    for language in BENCH_LANGUAGES
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
    "transcribe-float32-cpu-fr_fr-0": render_lines(("fr_fr", "1.wav", "0 1 2")),
    "transcribe-float32-cpu-fr_fr-1": render_lines(("fr_fr", "2.wav", "3")),
    "transcribe-float32-cpu-en_us-0": render_lines(("en_us", "3.wav", "1")),
    "transcribe-float32-cpu-en_us-1": render_lines(("en_us", "4.wav", "3")),
    "transcribe-float32-gpu-fr_fr-0": render_lines(
        ("fr_fr", "1.wav", "0 4 2"), ("fr_fr", "2.wav", "3")
    ),
    "transcribe-float32-gpu-en_us-0": render_lines(
        ("en_us", "3.wav", "1"), ("en_us", "4.wav", "3")
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
    candidate_directory = WORK_ROOT / ENTRIES_DIRECTORY / "candidate"
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
        inspect_directory = WORK_ROOT / ENTRIES_DIRECTORY / INSPECT_STEP_NAME
        files.add_directory(inspect_directory, build_directory_facts(0o2770))
        files.add_file(
            inspect_directory / INSPECTION_REPORT_FILE_NAME,
            *build_step_output(render_inspection_report(inspection_report)),
        )
        files.add_file(inspect_directory / PEAKS_FILE_NAME, *build_step_output(PEAKS))
    export_directory = WORK_ROOT / ENTRIES_DIRECTORY / "export"
    files.add_directory(export_directory, build_directory_facts(0o2770))
    for name, content in (
        ("vocab.txt", VOCABULARY),
        (ENCODER_NAME, ENCODER),
        (PARITY_OUTPUT.name, render_parity_report(ParityReport(ENCODER_PARITY.name, 4e-6, 20))),
        (PEAKS_FILE_NAME, PEAKS),
    ):
        files.add_file(export_directory / name, *build_step_output(content))
    for row_name, lines in shard_lines.items():
        shard_directory = WORK_ROOT / ENTRIES_DIRECTORY / row_name
        files.add_directory(shard_directory, build_directory_facts(0o2770))
        files.add_file(shard_directory / RUN_LINES_FILE_NAME, *build_step_output(lines))
        files.add_file(shard_directory / PEAKS_FILE_NAME, *build_step_output(PEAKS))
        if "gpu" in row_name:
            files.add_file(
                shard_directory / PLACEMENT_FILE_NAME,
                *build_step_output(render_placement_summary(placement)),
            )
    return files


ROWS = (EXPORT_ROW, *(derived_row.row for derived_row in DERIVED_ROWS))


def build_request(comparison: Comparison = CROSS_DEVICE) -> VerificationRequest:
    return VerificationRequest(
        candidate=CandidateRequest(
            work_root=WORK_ROOT,
            head_tree=HEAD_TREE,
            host=HOST,
            candidate=(ENCODER_FILE, VOCABULARY_FILE),
            rows=ROWS,
            conversion_input=None,
            candidate_directory=ENTRIES_DIRECTORY / "candidate",
            inspection_directory=ENTRIES_DIRECTORY / INSPECT_STEP_NAME,
        ),
        pins=PINS,
        runs=RUNS,
        derived_rows=DERIVED_ROWS,
        parities=(ENCODER_PARITY,),
        comparisons=(comparison,),
        output_directories={row.name: ENTRIES_DIRECTORY / row.name for row in ROWS},
        report_path=REPORT_PATH,
    )
