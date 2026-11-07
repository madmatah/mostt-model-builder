import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path, PurePosixPath

from run_step_fakes import (
    GROUP_ID,
    USER_ID,
    FakeContainerEngine,
    FakeStepFiles,
    WrittenByRun,
    build_file_facts,
    build_step_output,
)

from mostt_model_builder.application.clean_run import BuildCleaning, CleanEvent, clean_build
from mostt_model_builder.application.prune_cache import CachePruning, PrunedEntry, prune_cache
from mostt_model_builder.application.resolve_build import (
    BuildEvent,
    BuildResolutionRequest,
    EntryReused,
    ResolvedBuild,
    resolve_build,
)
from mostt_model_builder.application.run_step import HostIdentity, ImageFacts
from mostt_model_builder.domain.bench.fleurs_check_report import (
    FLEURS_CHECK_REPORT_FILE_NAME,
    FleursCheckReport,
    LanguageCheck,
    RecordingDuration,
    render_fleurs_check_report,
)
from mostt_model_builder.domain.bench.fleurs_index import FLEURS_REPOSITORY, locate_bench_files
from mostt_model_builder.domain.bench.hypotheses import Hypothesis, parse_run_lines
from mostt_model_builder.domain.bench.run_lines import RUN_LINES_FILE_NAME, RUN_LINES_MAX_SIZE_BYTES
from mostt_model_builder.domain.cache.build_units import BuildUnit, plan_units
from mostt_model_builder.domain.cache.hardware_facts import HardwareFacts
from mostt_model_builder.domain.cache.output_records import (
    OutputRecord,
    RecordedOutput,
    parse_output_record,
    render_output_record,
)
from mostt_model_builder.domain.containers.account_ranges import HostAccounts
from mostt_model_builder.domain.containers.base_images import BASE_PYTHON
from mostt_model_builder.domain.containers.container_files import FileFacts
from mostt_model_builder.domain.containers.container_paths import REQUEST_IN_THE_CONTAINER
from mostt_model_builder.domain.containers.container_steps import (
    BENCH_IMAGE,
    FLEURS_CHECK_ROW,
    INSPECT_ROW,
    ONNX_INSPECT_IMAGE,
    PEAKS_OUTPUT,
    ContainerImage,
    ContainerRun,
    DeclaredOutput,
    SourceReference,
    StepLimits,
    StepRow,
    build_clean_row,
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
    JudgedRecipe,
    JudgedRun,
    StepOutputOrigin,
)
from mostt_model_builder.domain.contract.shard_request import (
    SHARD_REQUEST_FILE_NAME,
    parse_shard_request,
)
from mostt_model_builder.domain.formats.file_formats import FileFormat
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
    PlacedNodes,
    PlacementSummary,
    render_placement_summary,
)
from mostt_model_builder.domain.sources.source_pins import RepositoryType, SourcePin

CLONE_ROOT = Path(__file__).resolve().parents[2]
WORK_ROOT = PurePosixPath("/data/builds/work")
HEAD_TREE = "3f1d1ffe79bc3336eb583331cfbf9b189fd45f01"
IMAGE_ID = "sha256:" + "c" * 64
RECIPE_NAME = "family"
HOST = HostIdentity(USER_ID, GROUP_ID, 20, HostAccounts(False, (), ()))
FLEURS_REVISION = "70bb2e84b976b7e960aa89f1c648e09c59f894dd"
WEIGHTS_REVISION = "73175eb7aeb0d82f1e2a6b53b3aabc10a90bcd0b"
LANGUAGES = ("fr_fr", "en_us")
LIMITS = StepLimits(memory_mib=1024, process_count=64, cpu_count=2, temporary_directory_mib=64)
PEAKS = render_step_peaks(StepPeaks(memory_peak_bytes=64 * 2**20, process_count_peak=6))
GPU_HARDWARE = HardwareFacts(("AMD Ryzen 9 7950X 16-Core Processor",), ("RTX 4090",), "550.54.14")
HARDWARE_WITHOUT_GPU = HardwareFacts(("AMD Ryzen 9 7950X 16-Core Processor",))
EXPORT_IMAGE = ContainerImage(RECIPE_NAME, "export", Device.CPU, BASE_PYTHON)
CPU_IMAGE = ContainerImage(RECIPE_NAME, "transcribe-cpu", Device.CPU, BASE_PYTHON)
GPU_IMAGE = ContainerImage(RECIPE_NAME, "transcribe-gpu", Device.GPU, BASE_PYTHON)
EXPORT_MODULE = "mostt_recipe_family.steps.export"
TRANSCRIBE_MODULE = "mostt_recipe_family.steps.transcribe"
ENCODER_NAME = "encoder.onnx"
VOCABULARY_NAME = "vocab.txt"
ENCODER = b"\x08\x07\x3a\x00"
VOCABULARY = b"le\nchat\ndort\n"
WEIGHTS_CONTENT = b"weights"


def build_pin(repository: str, revision: str, path: str, content: bytes) -> SourcePin:
    repository_type = (
        RepositoryType.DATASET if repository == FLEURS_REPOSITORY else (RepositoryType.MODEL)
    )
    return SourcePin(
        repository_type,
        repository,
        revision,
        path,
        len(content),
        hashlib.sha256(content).hexdigest(),
    )


BENCH_CONTENTS = {
    path: f"{path}\n".encode() for language in LANGUAGES for path in locate_bench_files(language)
}
BENCH_PINS = tuple(
    build_pin(FLEURS_REPOSITORY, FLEURS_REVISION, path, content)
    for path, content in BENCH_CONTENTS.items()
)
WEIGHTS_PIN = build_pin("maker/weights", WEIGHTS_REVISION, "model.safetensors", WEIGHTS_CONTENT)
EXPORT_ROW = StepRow(
    name="export",
    image=EXPORT_IMAGE,
    entry_module=EXPORT_MODULE,
    sources=(),
    has_request=False,
    outputs=(
        DeclaredOutput(ENCODER_NAME, 4096),
        DeclaredOutput(VOCABULARY_NAME, 4096),
        PEAKS_OUTPUT,
    ),
    limits=LIMITS,
)
LINES_OUTPUT = DeclaredOutput(RUN_LINES_FILE_NAME, RUN_LINES_MAX_SIZE_BYTES)
TEMPLATE = StepTemplate(
    name="transcribe",
    entry_module=TRANSCRIBE_MODULE,
    sources=(),
    earlier_outputs=(),
    binds_candidate=True,
    outputs=(PEAKS_OUTPUT,),
    devices={
        Device.CPU: DeviceRow(CPU_IMAGE, LIMITS, (LINES_OUTPUT,)),
        Device.GPU: DeviceRow(
            GPU_IMAGE,
            LIMITS,
            (LINES_OUTPUT, DeclaredOutput(PLACEMENT_FILE_NAME, PLACEMENT_MAX_SIZE_BYTES)),
        ),
    },
)
CANDIDATE = (
    JudgedCandidateFile(
        ENCODER_NAME, FileFormat.ONNX, ("float32",), StepOutputOrigin("export", ENCODER_NAME)
    ),
    JudgedCandidateFile(
        VOCABULARY_NAME, FileFormat.TEXT, ("float32",), StepOutputOrigin("export", VOCABULARY_NAME)
    ),
)
INSPECTION_REPORT = InspectionReport(
    (
        FileInspection(
            name=ENCODER_NAME,
            opsets=(Opset("", 17),),
            functions=(),
            nodes=(NodeCount(NodeScope.GRAPH, "", "MatMul", 1),),
            metadata_props_node_count=0,
            products=(),
            external_tensors=(),
            training_graph_count=0,
            sparse_initializer_count=0,
        ),
    )
)
FLEURS_REPORT = FleursCheckReport(
    tuple(LanguageCheck(language, (RecordingDuration("1.wav", 16000),)) for language in LANGUAGES)
)
GPU_PLACEMENT = PlacementSummary((PlacedNodes("CUDAExecutionProvider", "MatMul", ("float32",), 4),))
TREE_FILES = {
    PurePosixPath("src/mostt_model_builder/__init__.py"): b"",
    PurePosixPath("src/mostt_model_builder/bootstrap.py"): b"import sys\n",
    PurePosixPath("src/mostt_model_builder/steps/__init__.py"): b"",
    PurePosixPath("src/mostt_model_builder/steps/fleurs_check.py"): b"CHECK = 1\n",
    PurePosixPath("src/mostt_model_builder/steps/onnx_inspection.py"): b"INSPECT = 1\n",
    PurePosixPath("recipes/family/src/mostt_recipe_family/__init__.py"): b"",
    PurePosixPath("recipes/family/src/mostt_recipe_family/steps/__init__.py"): b"",
    PurePosixPath("recipes/family/src/mostt_recipe_family/steps/export.py"): b"EXPORT = 1\n",
    PurePosixPath("recipes/family/src/mostt_recipe_family/steps/transcribe.py"): (
        b"from mostt_recipe_family.domain import reader\n"
    ),
    PurePosixPath("recipes/family/src/mostt_recipe_family/domain/__init__.py"): b"",
    PurePosixPath("recipes/family/src/mostt_recipe_family/domain/reader.py"): b"READ = 1\n",
    PurePosixPath("images/bench/Containerfile"): b"FROM python\n",
    PurePosixPath("images/onnx-inspect/Containerfile"): b"FROM python\n",
    PurePosixPath("recipes/family/images/export/Containerfile"): b"FROM python\n",
    PurePosixPath("recipes/family/images/transcribe-cpu/Containerfile"): b"FROM python\n",
    PurePosixPath("recipes/family/images/transcribe-gpu/Containerfile"): b"FROM python\n",
    PurePosixPath("images/environment.lock"): (CLONE_ROOT / "images/environment.lock").read_bytes(),
}
BUILD_ORDER = ("fleurs-check", "export", "inspect", "transcribe")
ENGINE_IMAGES = (BENCH_IMAGE, ONNX_INSPECT_IMAGE, EXPORT_IMAGE, CPU_IMAGE, GPU_IMAGE)
CLEAN_MODULE = build_clean_row(BENCH_IMAGE).entry_module


def read_vocabulary_lines(
    lines: bytes, candidate_texts: Mapping[str, bytes]
) -> tuple[Hypothesis, ...]:
    words = candidate_texts[VOCABULARY_NAME].decode().splitlines()
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


@dataclass(frozen=True)
class RecipeShape:
    cpu_variant: str = "float32"
    export_sources: tuple[SourcePin, ...] = ()
    cpu_shard_count: int = 1


DEFAULT_SHAPE = RecipeShape()


def build_recipe(shape: RecipeShape = DEFAULT_SHAPE) -> JudgedRecipe:
    sharded_runs = (
        ShardedRun(
            "float32-cpu",
            shape.cpu_variant,
            Device.CPU,
            LANGUAGES,
            shape.cpu_shard_count,
            (ENCODER_NAME, VOCABULARY_NAME),
        ),
        ShardedRun(
            "float32-gpu", "float32", Device.GPU, LANGUAGES, 1, (ENCODER_NAME, VOCABULARY_NAME)
        ),
    )
    derived_rows = derive_rows(TEMPLATE, sharded_runs)
    export_row = replace(
        EXPORT_ROW,
        sources=tuple(SourceReference(pin.repository, pin.path) for pin in shape.export_sources),
    )
    return JudgedRecipe(
        name=RECIPE_NAME,
        pins=(*BENCH_PINS, *shape.export_sources),
        images=(EXPORT_IMAGE, CPU_IMAGE, GPU_IMAGE),
        rows=(export_row, *(derived_row.row for derived_row in derived_rows)),
        candidate=CANDIDATE,
        parities=(),
        card_template=None,
        runs=tuple(
            JudgedRun(
                run.name,
                "transcribe",
                run.device,
                run.variant,
                run.languages,
                run.shard_count,
                read_vocabulary_lines,
            )
            for run in sharded_runs
        ),
        derived_rows=derived_rows,
        build_order=BUILD_ORDER,
    )


def find_request_content(run: ContainerRun, files: FakeStepFiles, name: str) -> bytes:
    for mount in run.mounts:
        if mount.container_path == REQUEST_IN_THE_CONTAINER:
            return files.contents[run.work_root / mount.host_path / name]
    raise LookupError(f"the run of {run.entry_module} has no request")


def write_transcription(
    run: ContainerRun, files: FakeStepFiles
) -> dict[str, tuple[FileFacts, bytes]]:
    request = parse_shard_request(find_request_content(run, files, SHARD_REQUEST_FILE_NAME))
    language = request.languages[0]
    utterance = f"{request.shard + 1}.wav"
    lines = json.dumps({"language": language, "utterance": utterance, "tokens": "0 1"}).encode()
    outputs = {
        RUN_LINES_FILE_NAME: build_step_output(lines + b"\n"),
        PEAKS_FILE_NAME: build_step_output(PEAKS),
    }
    if request.device is Device.GPU:
        outputs[PLACEMENT_FILE_NAME] = build_step_output(render_placement_summary(GPU_PLACEMENT))
    return outputs


def write_fixed_outputs(outputs: dict[str, tuple[FileFacts, bytes]]) -> WrittenByRun:
    def write(run: ContainerRun, files: FakeStepFiles) -> dict[str, tuple[FileFacts, bytes]]:
        del run, files
        return outputs

    return write


def build_engine(files: FakeStepFiles) -> FakeContainerEngine:
    return FakeContainerEngine(
        files=files,
        images={image.tag: ImageFacts(IMAGE_ID, HEAD_TREE) for image in ENGINE_IMAGES},
        outputs_by_entry_module={
            FLEURS_CHECK_ROW.entry_module: write_fixed_outputs(
                {
                    FLEURS_CHECK_REPORT_FILE_NAME: build_step_output(
                        render_fleurs_check_report(FLEURS_REPORT)
                    ),
                    PEAKS_FILE_NAME: build_step_output(PEAKS),
                }
            ),
            EXPORT_MODULE: write_fixed_outputs(
                {
                    ENCODER_NAME: build_step_output(ENCODER),
                    VOCABULARY_NAME: build_step_output(VOCABULARY),
                    PEAKS_FILE_NAME: build_step_output(PEAKS),
                }
            ),
            INSPECT_ROW.entry_module: write_fixed_outputs(
                {
                    INSPECTION_REPORT_FILE_NAME: build_step_output(
                        render_inspection_report(INSPECTION_REPORT)
                    ),
                    PEAKS_FILE_NAME: build_step_output(PEAKS),
                }
            ),
            TRANSCRIBE_MODULE: write_transcription,
        },
    )


def build_work_files() -> FakeStepFiles:
    files = FakeStepFiles()
    files.add_directory(WORK_ROOT)
    for pin, content in (
        *zip(BENCH_PINS, BENCH_CONTENTS.values(), strict=True),
        (WEIGHTS_PIN, WEIGHTS_CONTENT),
    ):
        files.add_file(
            WORK_ROOT / "sources" / PurePosixPath(*pin.layout_segments),
            build_file_facts(len(content), mode=0o444),
            content,
        )
    return files


@dataclass
class CacheBuild:
    files: FakeStepFiles = field(default_factory=build_work_files)
    events: list[BuildEvent] = field(default_factory=list)

    def resolve(
        self,
        recipe: JudgedRecipe | None = None,
        tree_files: Mapping[PurePosixPath, bytes] = TREE_FILES,
        hardware: HardwareFacts | None = GPU_HARDWARE,
        running_steps: frozenset[str] = frozenset(BUILD_ORDER),
        units: Sequence[BuildUnit] | None = None,
        engine: FakeContainerEngine | None = None,
    ) -> tuple[ResolvedBuild, FakeContainerEngine]:
        judged = recipe or build_recipe()
        engine = engine or build_engine(self.files)
        self.events.clear()
        resolved = resolve_build(
            BuildResolutionRequest(
                recipe=judged,
                units=units if units is not None else plan_units(judged, judged.build_order),
                running_steps=running_steps,
                work_root=WORK_ROOT,
                head_tree=HEAD_TREE,
                host=HOST,
                tree_files=tree_files,
                hardware=hardware,
            ),
            engine,
            self.files,
            self.events.append,
        )
        return resolved, engine


def list_ran_rows(engine: FakeContainerEngine) -> list[str]:
    return [
        mount.host_path.parts[2]
        for run in engine.runs
        if run.entry_module != CLEAN_MODULE
        for mount in run.mounts
        if not mount.is_read_only
    ]


def list_reused_entries(events: Sequence[BuildEvent]) -> list[str]:
    return [event.name for event in events if isinstance(event, EntryReused)]


def rewrite_output(
    files: FakeStepFiles, entry: PurePosixPath, output_name: str, content: bytes
) -> None:
    output_path = WORK_ROOT / entry / "output" / output_name
    files.contents[output_path] = content
    files.entries[output_path] = build_step_output(content)[0]


def rewrite_record(files: FakeStepFiles, entry: PurePosixPath, directory_name: str) -> None:
    record_path = WORK_ROOT / entry / "record.json"
    record = parse_output_record(files.contents[record_path])
    outputs = tuple(
        RecordedOutput(
            output.name,
            len(files.contents[WORK_ROOT / entry / directory_name / output.name]),
            hashlib.sha256(
                files.contents[WORK_ROOT / entry / directory_name / output.name]
            ).hexdigest(),
        )
        for output in record.outputs
    )
    content = render_output_record(OutputRecord(record.key, record.row, record.document, outputs))
    files.contents[record_path] = content
    files.entries[record_path] = build_file_facts(len(content), 0o444)


def clean_cache(
    cache: CacheBuild,
    steps: frozenset[str] | None,
    hardware: HardwareFacts | None = GPU_HARDWARE,
) -> list[CleanEvent]:
    recipe = build_recipe()
    events: list[CleanEvent] = []
    clean_build(
        BuildCleaning(
            recipe=recipe,
            units=plan_units(recipe, recipe.build_order),
            steps=steps,
            work_root=WORK_ROOT,
            head_tree=HEAD_TREE,
            host=HOST,
            tree_files=TREE_FILES,
            hardware=hardware,
        ),
        build_engine(cache.files),
        cache.files,
        events.append,
    )
    return events


def list_entries(files: FakeStepFiles) -> list[PurePosixPath]:
    cache_directory = WORK_ROOT / "cache" / RECIPE_NAME
    return sorted(
        path.relative_to(cache_directory)
        for path in files.entries
        if path.is_relative_to(cache_directory)
        and len(path.relative_to(cache_directory).parts) == 2
    )


def prune_built_cache(
    cache: CacheBuild,
    tree_files: Mapping[PurePosixPath, bytes] = TREE_FILES,
    hardware: HardwareFacts | None = GPU_HARDWARE,
) -> tuple[PrunedEntry, ...]:
    recipe = build_recipe()
    return prune_cache(
        CachePruning(
            recipe=recipe,
            units=plan_units(recipe, recipe.build_order),
            work_root=WORK_ROOT,
            head_tree=HEAD_TREE,
            host=HOST,
            tree_files=tree_files,
            hardware=hardware,
        ),
        build_engine(cache.files),
        cache.files,
        ignore_pruned_entry,
    )


def ignore_pruned_entry(entry: PrunedEntry) -> None:
    del entry
