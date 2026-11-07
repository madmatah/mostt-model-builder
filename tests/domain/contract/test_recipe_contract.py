import re
from collections.abc import Callable, Mapping
from dataclasses import replace
from functools import partial
from pathlib import PurePosixPath

import pytest

from mostt_model_builder.domain.acceptance.acceptance import (
    REPORTED_ONLY,
    Comparison,
    Equivalence,
    NonInferiority,
)
from mostt_model_builder.domain.bench.hypotheses import Hypothesis
from mostt_model_builder.domain.containers.base_images import BASE_PYTHON, ENVIRONMENT_PYTHON
from mostt_model_builder.domain.containers.container_steps import (
    FLEURS_CHECK_ROW,
    PEAKS_OUTPUT,
    DeclaredOutput,
    EarlierOutput,
    SourceReference,
    StepLimits,
    StepRow,
)
from mostt_model_builder.domain.contract.model_description import (
    ModelDescription,
    VariantDescription,
)
from mostt_model_builder.domain.contract.recipe_contract import (
    CANDIDATE_MOUNT,
    REFERENCE_VARIANT,
    CandidateFile,
    DeviceTemplate,
    EarlierOutputMount,
    HostContentOrigin,
    JudgedRecipe,
    Parity,
    RecipeContract,
    RecipeContractError,
    RecipeFacts,
    RecipeImage,
    RecipeStep,
    Run,
    SourceMount,
    StepOutputOrigin,
    VariantConversion,
    build_conversion_request,
    find_candidate_file_max_size_bytes,
    find_graph_source,
    find_runnable_row,
    judge_recipe,
    locate_card_template,
    locate_entry_module_file,
)
from mostt_model_builder.domain.formats.file_formats import FileFormat
from mostt_model_builder.domain.onnx.conversion_request import (
    ConversionKind,
    ConversionRequest,
    EightBitBlocksConversion,
    FloatConversion,
)
from mostt_model_builder.domain.onnx.graph_rules import EightBitBlocksOutput, GraphSource
from mostt_model_builder.domain.sources.source_pins import RepositoryType, SourcePin

RECIPE_NAME = "family"
REVISION = "0123456789abcdef0123456789abcdef01234567"
SHA256 = "0" * 64
WEIGHTS_REPOSITORY = "maker/weights"
ARCHIVE_REPOSITORY = "maker/archive"
BENCH_PIN = SourcePin(RepositoryType.DATASET, "google/fleurs", REVISION, "data/test.tsv", 1, SHA256)
WEIGHTS_PIN = SourcePin(
    RepositoryType.MODEL, WEIGHTS_REPOSITORY, REVISION, "model.safetensors", 1, SHA256
)
NESTED_PIN = SourcePin(
    RepositoryType.MODEL, WEIGHTS_REPOSITORY, REVISION, "configs/model.json", 1, SHA256
)
ARCHIVE_PIN = SourcePin(
    RepositoryType.MODEL, ARCHIVE_REPOSITORY, REVISION, "model.nemo", 1, SHA256, True
)
DESCRIBE_MODULE = "mostt_recipe_family.steps.description"
TRANSCRIBE_MODULE = "mostt_recipe_family.steps.transcription"
RECIPE_MODULE = "mostt_recipe_family.recipe"
FLEURS_CHECK_MODULE = FLEURS_CHECK_ROW.entry_module
LIMITS = StepLimits(memory_mib=1024, process_count=64, cpu_count=2, temporary_directory_mib=64)
NAMES_OUTPUT = DeclaredOutput("names", 1024)
EXPORT_IMAGE = RecipeImage("export", "cpu", "family-export", reads_code_running_sources=True)
TRANSCRIBE_IMAGE = RecipeImage("transcribe", "gpu", "family-transcribe")
DESCRIBE_STEP = RecipeStep(
    name="describe",
    image="export",
    entry_module=DESCRIBE_MODULE,
    mounts=(
        SourceMount(WEIGHTS_REPOSITORY, "model.safetensors"),
        SourceMount(ARCHIVE_REPOSITORY, "model.nemo"),
    ),
    outputs=(NAMES_OUTPUT, PEAKS_OUTPUT),
    limits=LIMITS,
)
TRANSCRIBE_STEP = RecipeStep(
    name="transcribe",
    image="transcribe",
    entry_module=TRANSCRIBE_MODULE,
    mounts=(EarlierOutputMount("describe", "names"), SourceMount("google/fleurs", "data/test.tsv")),
    outputs=(PEAKS_OUTPUT,),
    limits=LIMITS,
    uses_gpu=True,
)
NAMES_FILE = CandidateFile("names.txt", "text", ("float32",), StepOutputOrigin("describe", "names"))
CONFIGURATION = b"{}"
CONFIGURATION_FILE = CandidateFile(
    "config.json", "text", ("float32", "float16"), HostContentOrigin(CONFIGURATION)
)
NAMES_PARITY = Parity("names-against-reference", "describe", "names", 1e-3)
CONTRACT = RecipeContract(
    name=RECIPE_NAME,
    images=(EXPORT_IMAGE, TRANSCRIBE_IMAGE),
    steps=(DESCRIBE_STEP, TRANSCRIBE_STEP),
    candidate=(NAMES_FILE, CONFIGURATION_FILE),
    parities=(NAMES_PARITY,),
)
FACTS = RecipeFacts(
    core_pins=(BENCH_PIN,),
    recipe_pins=(WEIGHTS_PIN, NESTED_PIN, ARCHIVE_PIN),
    present_entry_modules=frozenset(
        {DESCRIBE_MODULE, TRANSCRIBE_MODULE, RECIPE_MODULE, FLEURS_CHECK_MODULE}
    ),
    host_cpu_count=20,
    has_card_template=True,
)


# A reader takes a shard's lines and the candidate's texts; the contract only checks that a run
# has one, so this one reads nothing.
def read_no_lines(
    lines: bytes,  # noqa: ARG001
    candidate_texts: Mapping[str, bytes],  # noqa: ARG001
) -> tuple[Hypothesis, ...]:
    return ()


LINES_OUTPUT = DeclaredOutput("lines.jsonl", 2**20)
PLACEMENT_OUTPUT = DeclaredOutput("placement.json", 2**20)
SHARD_LIMITS = replace(LIMITS, cpu_count=1)
TRANSCRIBE_CPU_IMAGE = RecipeImage("transcribe-cpu", "cpu", "family-transcribe-cpu")
SHARDED_STEP = RecipeStep(
    name="transcribe-variants",
    image=None,
    entry_module=TRANSCRIBE_MODULE,
    mounts=(CANDIDATE_MOUNT, SourceMount("google/fleurs", "data/test.tsv")),
    outputs=(PEAKS_OUTPUT,),
    limits=None,
    devices=(
        DeviceTemplate("cpu", "transcribe-cpu", SHARD_LIMITS, (LINES_OUTPUT,)),
        DeviceTemplate("gpu", "transcribe", LIMITS, (LINES_OUTPUT, PLACEMENT_OUTPUT)),
    ),
)
ENCODER_FILE = CandidateFile(
    "encoder.onnx", "onnx", ("float32",), StepOutputOrigin("describe", "names")
)
ENCODER_DATA_FILE = CandidateFile(
    "encoder.onnx.data",
    "onnx-external-data",
    ("float32",),
    HostContentOrigin(b"\x00"),
    external_data_of="encoder.onnx",
)
CPU_RUN = Run(
    "float32-cpu", "transcribe-variants", "cpu", "float32", ("fr_fr", "en_us"), 3, read_no_lines
)
GPU_RUN = Run("float32-gpu", "transcribe-variants", "gpu", "float32", ("fr_fr",), 1, read_no_lines)
FLOAT32_FILES = (NAMES_FILE.name, ENCODER_FILE.name, ENCODER_DATA_FILE.name)
RUN_CONTRACT = replace(
    CONTRACT,
    images=(EXPORT_IMAGE, TRANSCRIBE_IMAGE, TRANSCRIBE_CPU_IMAGE),
    steps=(DESCRIBE_STEP, SHARDED_STEP),
    candidate=(NAMES_FILE, ENCODER_FILE, ENCODER_DATA_FILE),
    runs=(CPU_RUN, GPU_RUN),
    conversion_input=StepOutputOrigin("describe", "names"),
    build_order=("fleurs-check", "describe", "inspect", "transcribe-variants"),
)
RUN_FACTS = replace(FACTS, present_entry_modules=FACTS.present_entry_modules | {TRANSCRIBE_MODULE})
CROSS_DEVICE_COMPARISON = Comparison("float32-gpu", "float32-cpu", Equivalence(0.99, 0.10))
COMPARISON_CONTRACT = replace(RUN_CONTRACT, comparisons=(CROSS_DEVICE_COMPARISON,))


CONVERTED_SIZE_BOUND_BYTES = 2**31
EXPORT_STEP = RecipeStep(
    name="export",
    image="export",
    entry_module=DESCRIBE_MODULE,
    mounts=(),
    outputs=(
        DeclaredOutput("encoder.onnx", 2**20),
        DeclaredOutput("encoder.onnx.data", 2**31),
        PEAKS_OUTPUT,
    ),
    limits=LIMITS,
)
EXPORTED_ENCODER = StepOutputOrigin("export", "encoder.onnx")
FLOAT16_CONVERSION = VariantConversion(
    "float16",
    FloatConversion(ConversionKind.FLOAT16, "encoder.fp16.onnx"),
    CONVERTED_SIZE_BOUND_BYTES,
)
EIGHT_BIT_BLOCKS_CONVERSION = VariantConversion(
    "eight-bit-blocks",
    EightBitBlocksConversion(
        "encoder.int8.onnx",
        block_size=64,
        is_symmetric=True,
        accuracy_level=4,
        quantized_product_count=265,
    ),
    CONVERTED_SIZE_BOUND_BYTES,
)
EXPORTED_ENCODER_FILE = CandidateFile("encoder.onnx", "onnx", ("float32",), EXPORTED_ENCODER)
EXPORTED_DATA_FILE = CandidateFile(
    "encoder.onnx.data",
    "onnx-external-data",
    ("float32",),
    StepOutputOrigin("export", "encoder.onnx.data"),
    external_data_of="encoder.onnx",
)
FLOAT16_FILE = CandidateFile(
    "encoder.fp16.onnx",
    "onnx",
    ("float16",),
    StepOutputOrigin("convert", "encoder.fp16.onnx"),
)
EIGHT_BIT_BLOCKS_FILE = CandidateFile(
    "encoder.int8.onnx",
    "onnx",
    ("eight-bit-blocks",),
    StepOutputOrigin("convert", "encoder.int8.onnx"),
)
CONVERSION_CONTRACT = RecipeContract(
    name=RECIPE_NAME,
    images=(EXPORT_IMAGE,),
    steps=(EXPORT_STEP,),
    candidate=(EXPORTED_ENCODER_FILE, EXPORTED_DATA_FILE, FLOAT16_FILE, EIGHT_BIT_BLOCKS_FILE),
    conversion_input=EXPORTED_ENCODER,
    build_order=("export", "convert", "inspect"),
    conversions=(FLOAT16_CONVERSION, EIGHT_BIT_BLOCKS_CONVERSION),
)


def change_conversion(conversion_index: int, **fields: object) -> RecipeContract:
    conversions = list(CONVERSION_CONTRACT.conversions)
    conversion = conversions[conversion_index].conversion
    conversions[conversion_index] = replace(
        conversions[conversion_index],
        conversion=replace(conversion, **fields),  # type: ignore[arg-type]  # the test feeds a value outside the contract's types, on purpose
    )
    return replace(CONVERSION_CONTRACT, conversions=tuple(conversions))


def change_variant_conversion(conversion_index: int, **fields: object) -> RecipeContract:
    conversions = list(CONVERSION_CONTRACT.conversions)
    conversions[conversion_index] = replace(conversions[conversion_index], **fields)  # type: ignore[arg-type]  # the test feeds a value outside the contract's types, on purpose
    return replace(CONVERSION_CONTRACT, conversions=tuple(conversions))


def change_run(run_index: int, **fields: object) -> RecipeContract:
    runs = list(RUN_CONTRACT.runs)
    runs[run_index] = replace(runs[run_index], **fields)  # type: ignore[arg-type]  # the test feeds a value outside the contract's types, on purpose
    return replace(RUN_CONTRACT, runs=tuple(runs))


def compare(*comparisons: Comparison) -> RecipeContract:
    return replace(RUN_CONTRACT, comparisons=comparisons)


def change_sharded_step(**fields: object) -> RecipeContract:
    return replace(
        RUN_CONTRACT,
        steps=(DESCRIBE_STEP, replace(SHARDED_STEP, **fields)),  # type: ignore[arg-type]  # the test feeds a value outside the contract's types, on purpose
    )


def change_device_template(device_index: int, **fields: object) -> RecipeContract:
    devices = list(SHARDED_STEP.devices)
    devices[device_index] = replace(devices[device_index], **fields)  # type: ignore[arg-type]  # the test feeds a value outside the contract's types, on purpose
    return change_sharded_step(devices=tuple(devices))


def change_external_data(**fields: object) -> RecipeContract:
    return replace(
        RUN_CONTRACT,
        candidate=(NAMES_FILE, ENCODER_FILE, replace(ENCODER_DATA_FILE, **fields)),  # type: ignore[arg-type]  # the test feeds a value outside the contract's types, on purpose
    )


def change_step(step_index: int, **fields: object) -> RecipeContract:
    steps = list(CONTRACT.steps)
    steps[step_index] = replace(steps[step_index], **fields)  # type: ignore[arg-type]  # the test feeds a value outside the contract's types, on purpose
    return replace(CONTRACT, steps=tuple(steps))


def change_image(image_index: int, **fields: object) -> RecipeContract:
    images = list(CONTRACT.images)
    images[image_index] = replace(images[image_index], **fields)  # type: ignore[arg-type]  # the test feeds a value outside the contract's types, on purpose
    return replace(CONTRACT, images=tuple(images))


def change_candidate(**fields: object) -> RecipeContract:
    return replace(CONTRACT, candidate=(replace(NAMES_FILE, **fields),))  # type: ignore[arg-type]  # the test feeds a value outside the contract's types, on purpose


def change_parity(**fields: object) -> RecipeContract:
    return replace(CONTRACT, parities=(replace(NAMES_PARITY, **fields),))  # type: ignore[arg-type]  # the test feeds a value outside the contract's types, on purpose


MODEL = ModelDescription(
    title="Weights in ONNX",
    template_path=PurePosixPath("model-card.md"),
    measuring_library_names=("onnx-asr",),
    variants=(VariantDescription("float32", "float32"),),
)


REFUSED_CONTRACTS = {
    "a card template at an absolute path": (
        replace(CONTRACT, model=replace(MODEL, template_path=PurePosixPath("/etc/passwd"))),
        "its card template '/etc/passwd' is not a file name in the recipe's directory",
    ),
    "a card template outside the recipe's directory": (
        replace(
            CONTRACT, model=replace(MODEL, template_path=PurePosixPath("../other/model-card.md"))
        ),
        "its card template '../other/model-card.md' is not a file name in the recipe's directory",
    ),
    "a card template in a subdirectory": (
        replace(CONTRACT, model=replace(MODEL, template_path=PurePosixPath("cards/model-card.md"))),
        "its card template 'cards/model-card.md' is not a file name in the recipe's directory",
    ),
    "a card template whose name starts with a dot": (
        replace(CONTRACT, model=replace(MODEL, template_path=PurePosixPath(".model-card.md"))),
        "its card template '.model-card.md' is not a file name in the recipe's directory",
    ),
    "a step named like the directory of the publications": (
        change_step(0, name="publications"),
        "its step 'publications' bears the name of a step of the core or of a directory of its run",
    ),
    "a step named like the core's publication": (
        change_step(0, name="publish"),
        "its step 'publish' bears the name of a step of the core or of a directory of its run",
    ),
    "a candidate file named like the model card, in any case": (
        change_candidate(name="readme.MD"),
        "its candidate file 'readme.MD' bears the name of the model card, README.md, which the "
        "core writes",
    ),
    "a recipe name outside the grammar": (
        replace(CONTRACT, name="Family"),
        "the recipe 'Family' is not named under",
    ),
    "an image name outside the grammar": (
        change_image(0, name="Export"),
        "its image 'Export' is not named under",
    ),
    "a recipe image named like an image of the core": (
        change_image(0, name="bench"),
        "its image 'bench' bears the name of an image of the core",
    ),
    "an image declared twice": (
        replace(CONTRACT, images=(EXPORT_IMAGE, EXPORT_IMAGE)),
        "its image 'export' is declared twice",
    ),
    "an image without its device": (
        change_image(0, device=""),
        "its image 'export' has no device among cpu, gpu",
    ),
    "an image on an unknown device": (
        change_image(0, device="tpu"),
        "its image 'export' has no device among cpu, gpu",
    ),
    "an image that installs another image's extra": (
        change_image(0, extra="family-transcribe"),
        "installs the extra 'family-transcribe', not family-export",
    ),
    "two images that read the sources that run code": (
        change_image(1, reads_code_running_sources=True),
        "the images export, transcribe read the sources that run code when they load",
    ),
    "a step name outside the grammar": (
        change_step(0, name="Describe"),
        "its step 'Describe' is not named under",
    ),
    "a step named like a step of the core": (
        change_step(0, name="fleurs-check"),
        "its step 'fleurs-check' bears the name of a step of the core",
    ),
    "a step declared twice": (
        replace(CONTRACT, steps=(DESCRIBE_STEP, DESCRIBE_STEP)),
        "its step 'describe' is declared twice",
    ),
    "an unknown image": (
        change_step(0, image="convert"),
        "runs in the image 'convert', which neither the recipe nor the core declares",
    ),
    "an entry module that is not a dotted name": (
        change_step(0, entry_module="mostt_recipe_family.steps.de-scription"),
        "its entry module 'mostt_recipe_family.steps.de-scription' is not a dotted module name",
    ),
    "an entry module whose file does not exist": (
        change_step(0, entry_module="mostt_recipe_family.steps.missing"),
        "has no regular file at recipes/family/src/mostt_recipe_family/steps/missing.py",
    ),
    "an entry module of the core in a recipe image": (
        change_step(0, entry_module=FLEURS_CHECK_MODULE),
        "lies outside mostt_recipe_family.steps, the steps of the image",
    ),
    "an entry module of the recipe outside its steps": (
        change_step(0, entry_module=RECIPE_MODULE),
        "its entry module mostt_recipe_family.recipe lies outside mostt_recipe_family.steps, "
        "the steps of the image",
    ),
    "an entry module of the recipe in a core image": (
        change_step(0, image="bench", mounts=()),
        "lies outside mostt_model_builder.steps, the steps of the image",
    ),
    "an undeclared source": (
        change_step(0, mounts=(SourceMount(WEIGHTS_REPOSITORY, "other.safetensors"),)),
        "binds 'other.safetensors' of 'maker/weights', which no source pins",
    ),
    "a source bound as a directory": (
        change_step(0, mounts=(SourceMount(WEIGHTS_REPOSITORY, "configs"),)),
        "binds the directory 'configs' of maker/weights, where only a pinned file may be bound",
    ),
    "a source bound twice": (
        change_step(
            0,
            mounts=(
                SourceMount(WEIGHTS_REPOSITORY, "model.safetensors"),
                SourceMount(WEIGHTS_REPOSITORY, "model.safetensors"),
            ),
        ),
        "binds model.safetensors of maker/weights twice",
    ),
    "the source that runs code bound in another image": (
        change_step(1, mounts=(SourceMount(ARCHIVE_REPOSITORY, "model.nemo"),)),
        "binds model.nemo of maker/archive, which runs code when it loads, in the image "
        "mostt-model-builder-family-transcribe",
    ),
    "the source that runs code bound in a core image": (
        replace(
            CONTRACT,
            steps=(
                replace(
                    DESCRIBE_STEP,
                    image="bench",
                    entry_module=FLEURS_CHECK_MODULE,
                    mounts=(SourceMount(ARCHIVE_REPOSITORY, "model.nemo"),),
                ),
            ),
            candidate=(),
        ),
        "in the image mostt-model-builder-core-bench, which the recipe does not declare",
    ),
    "an output of a step that is not earlier": (
        replace(CONTRACT, steps=(TRANSCRIBE_STEP, DESCRIBE_STEP)),
        "binds an output of 'describe', which is not an earlier step of the recipe",
    ),
    "the outputs of an earlier step bound as a directory": (
        change_step(1, mounts=(EarlierOutputMount("describe", ""),)),
        "binds the outputs of describe as a directory, where only a declared output file",
    ),
    "an output that the earlier step does not declare": (
        change_step(1, mounts=(EarlierOutputMount("describe", "other"),)),
        "binds 'other' of describe, which that step does not declare",
    ),
    "an output without its size bound": (
        change_step(0, outputs=(DeclaredOutput("names", 0), PEAKS_OUTPUT)),
        "the size bound of names in bytes, 0, lies outside 1 to 8589934592",
    ),
    "an output beyond the ceiling": (
        change_step(0, outputs=(DeclaredOutput("names", 8 * 2**30 + 1), PEAKS_OUTPUT)),
        "the size bound of names in bytes, 8589934593, lies outside 1 to 8589934592",
    ),
    "outputs without the peaks": (
        change_step(0, outputs=(NAMES_OUTPUT,)),
        "its outputs lack peaks.json",
    ),
    "memory beyond its ceiling": (
        change_step(0, limits=replace(LIMITS, memory_mib=57_345)),
        "its memory in MiB, 57345, lies outside 6 to 57344",
    ),
    "more cpus than the host has": (
        change_step(0, limits=replace(LIMITS, cpu_count=21)),
        "its CPUs, 21, lies outside 1 to 20",
    ),
    "the gpu in a cpu image": (
        change_step(0, uses_gpu=True),
        "uses the GPU in the cpu image",
    ),
    "an unknown format": (
        change_candidate(file_format="gguf"),
        "its candidate file 'names.txt' has the format 'gguf', not one of onnx, text, "
        "onnx-external-data",
    ),
    "a candidate name outside the file-name grammar": (
        change_candidate(name="../names.txt"),
        "its candidate file '../names.txt' is not a file name",
    ),
    "a candidate name with a refused suffix": (
        change_candidate(name="weights.PT"),
        "its candidate file 'weights.PT' has the refused suffix .pt",
    ),
    "a candidate name with an inner refused suffix": (
        change_candidate(name="weights.pkl.txt"),
        "its candidate file 'weights.pkl.txt' has the refused suffix .pkl",
    ),
    "a candidate file declared twice": (
        replace(CONTRACT, candidate=(NAMES_FILE, NAMES_FILE)),
        "its candidate file 'names.txt' is declared twice",
    ),
    "a candidate file that serves no variant": (
        change_candidate(variants=()),
        "its candidate file 'names.txt' serves no variant",
    ),
    "a variant outside the grammar": (
        change_candidate(variants=("Float32",)),
        "serves the variant 'Float32', not named under",
    ),
    "a variant named twice": (
        change_candidate(variants=("float32", "float32")),
        "its candidate file 'names.txt' names a variant twice",
    ),
    "a candidate file from an undeclared output": (
        change_candidate(origin=StepOutputOrigin("describe", "other")),
        "comes from 'other' of 'describe', which no step of the recipe declares",
    ),
    "an empty content that the host writes": (
        change_candidate(origin=HostContentOrigin(b"")),
        "has a content of 0 bytes, outside 1 to 1048576",
    ),
    "a parity name outside the grammar": (
        change_parity(name="Names"),
        "its parity 'Names' is not named under",
    ),
    "a parity declared twice": (
        replace(CONTRACT, parities=(NAMES_PARITY, NAMES_PARITY)),
        "its parity 'names-against-reference' is declared twice",
    ),
    "a parity report that its step does not declare": (
        change_parity(output="parity.json"),
        "reads its report from 'parity.json' of 'describe', which no step of the recipe declares",
    ),
    "a parity report of a step the recipe does not declare": (
        change_parity(step="export"),
        "reads its report from 'names' of 'export', which no step of the recipe declares",
    ),
    "a parity report read from the peaks": (
        change_parity(output="peaks.json"),
        "reads its report from peaks.json, the peaks of its step",
    ),
    "a tolerance of zero": (
        change_parity(absolute_difference_tolerance=0.0),
        "has the tolerance 0.0, not a finite number above 0",
    ),
    "a negative tolerance": (
        change_parity(absolute_difference_tolerance=-5e-5),
        "has the tolerance -5e-05, not a finite number above 0",
    ),
    "a tolerance that is not a number": (
        change_parity(absolute_difference_tolerance=float("nan")),
        "has the tolerance nan, not a finite number above 0",
    ),
    "an infinite tolerance": (
        change_parity(absolute_difference_tolerance=float("inf")),
        "has the tolerance inf, not a finite number above 0",
    ),
    "a boolean tolerance": (
        change_parity(absolute_difference_tolerance=True),
        "has the tolerance True, not a finite number above 0",
    ),
    "a step named like a directory of its run": (
        change_step(0, name="candidate"),
        "its step 'candidate' bears the name of a step of the core or of a directory of its run",
    ),
    "a run outside the grammar": (
        change_run(0, name="Float32-cpu"),
        "its run 'Float32-cpu' is not named under",
    ),
    "a run declared twice": (
        replace(RUN_CONTRACT, runs=(CPU_RUN, CPU_RUN)),
        "its run 'float32-cpu' is declared twice",
    ),
    "a run without a producing step": (
        change_run(0, step="transcribe-other"),
        "its run 'float32-cpu' comes from 'transcribe-other', which is not a step of the recipe",
    ),
    "a run of a step without a template": (
        change_run(0, step="describe"),
        "its step 'describe' produces the run float32-cpu, and has no template of its rows",
    ),
    "a run without a reader": (
        change_run(0, reader=None),
        "its run 'float32-cpu' has no reader of its lines",
    ),
    "a run on an unknown device": (
        change_run(0, device="tpu"),
        "its run 'float32-cpu' runs on 'tpu', neither cpu nor gpu",
    ),
    "a run on a device that its template does not give": (
        change_sharded_step(devices=(SHARDED_STEP.devices[0],)),
        "its run float32-gpu runs on the gpu, which its template does not give",
    ),
    "a run that reads a variant the candidate lacks": (
        change_run(0, variant="float16"),
        "its run 'float32-cpu' reads the variant 'float16', which no file of the candidate serves",
    ),
    "two runs whose rows share a name": (
        replace(
            RUN_CONTRACT,
            steps=(
                DESCRIBE_STEP,
                replace(SHARDED_STEP, name="a"),
                replace(SHARDED_STEP, name="a-b"),
            ),
            runs=(replace(CPU_RUN, name="b-c", step="a"), replace(CPU_RUN, name="c", step="a-b")),
        ),
        "its step 'a-b' derives the row a-b-c-0, which another row already holds",
    ),
    "a run that reads no language": (
        change_run(0, languages=()),
        "its run 'float32-cpu' reads no language",
    ),
    "a run that reads a language outside the bench": (
        change_run(0, languages=("de_de",)),
        "its run 'float32-cpu' reads 'de_de', which is not a language of the bench",
    ),
    "a run that names a language twice": (
        change_run(0, languages=("fr_fr", "fr_fr")),
        "its run 'float32-cpu' names a language twice",
    ),
    "a run of no shard": (
        change_run(0, shard_count=0),
        "its run 'float32-cpu' has 0 shards, outside 1 to 16",
    ),
    "a run of 17 shards": (
        change_run(0, shard_count=17),
        "its run 'float32-cpu' has 17 shards, outside 1 to 16",
    ),
    "a template whose device does not match its image": (
        change_device_template(0, image="transcribe"),
        "its template for 'cpu' runs in the gpu image mostt-model-builder-family-transcribe",
    ),
    "a template on an unknown device": (
        change_device_template(0, device="tpu"),
        "its template for 'tpu' names no device among cpu, gpu",
    ),
    "a template whose shard writes no lines": (
        change_device_template(0, outputs=()),
        "its template for 'cpu' writes no lines.jsonl, the lines of a run",
    ),
    "a template of the gpu whose shard writes no placement": (
        change_device_template(1, outputs=(LINES_OUTPUT,)),
        "its template for 'gpu' writes no placement.json, the placement of its nodes on the GPU",
    ),
    "a template of the cpu whose shard writes a placement": (
        change_device_template(0, outputs=(LINES_OUTPUT, PLACEMENT_OUTPUT)),
        "its template for 'cpu' writes placement.json, which only a run on the GPU writes",
    ),
    "a step with a template and its own image": (
        change_sharded_step(image="transcribe"),
        "has a template of its rows, which gives each device its image, its limits and the GPU",
    ),
    "a step with a template and no run": (
        replace(RUN_CONTRACT, runs=()),
        "its step 'transcribe-variants' has a template of its rows, and no run",
    ),
    "a step with neither an image nor a template": (
        change_step(0, image=None),
        "its step 'describe' has neither an image and limits nor a template",
    ),
    "external data that names no onnx file": (
        change_external_data(external_data_of=None),
        "its candidate file encoder.onnx.data holds external data, and names no onnx file",
    ),
    "external data of a file that is not onnx": (
        change_external_data(external_data_of="names.txt"),
        "holds the tensors of 'names.txt', which is not an onnx file of the candidate",
    ),
    "external data whose onnx file serves another variant": (
        change_external_data(variants=("float16",)),
        "serves float16, and its onnx file encoder.onnx serves float32",
    ),
    "a file of another format that names an onnx file": (
        replace(RUN_CONTRACT, candidate=(replace(NAMES_FILE, external_data_of="encoder.onnx"),)),
        "is text, and names the onnx file whose tensors it holds",
    ),
    "conversions that read no onnx file of the candidate": (
        replace(RUN_CONTRACT, conversion_input=StepOutputOrigin("describe", "peaks.json")),
        "its conversions read 'peaks.json' of 'describe', which no onnx file of the candidate",
    ),
    "a build that runs an unknown step": (
        replace(RUN_CONTRACT, build_order=("fleurs-check", "publish")),
        "its build runs 'publish', which neither the core nor the recipe builds with",
    ),
    "a build that runs a step twice": (
        replace(RUN_CONTRACT, build_order=("describe", "describe")),
        "its build runs a step twice",
    ),
    "a step named like the conversions of the core": (
        change_step(0, name="convert"),
        "its step 'convert' bears the name of a step of the core",
    ),
    "a build that converts a recipe that declares no conversion": (
        replace(RUN_CONTRACT, build_order=("fleurs-check", "convert")),
        "its build runs 'convert', which neither the core nor the recipe builds with",
    ),
    "conversions without their input": (
        replace(CONVERSION_CONTRACT, conversion_input=None),
        "the recipe family converts its variants, and names no conversions' input",
    ),
    "conversions whose input is no output of an earlier step": (
        replace(CONVERSION_CONTRACT, conversion_input=StepOutputOrigin("other", "encoder.onnx")),
        "its conversions binds an output of 'other', which is not an earlier step of the recipe",
    ),
    "conversions whose input's external data comes from no step": (
        replace(
            CONVERSION_CONTRACT,
            candidate=(
                EXPORTED_ENCODER_FILE,
                replace(EXPORTED_DATA_FILE, origin=HostContentOrigin(b"\x00")),
                FLOAT16_FILE,
                EIGHT_BIT_BLOCKS_FILE,
            ),
        ),
        "the external data encoder.onnx.data of the conversions' input does not come from "
        "export, beside its onnx file",
    ),
    "a conversion to a variant outside the grammar": (
        change_variant_conversion(0, variant="Float16"),
        "its conversion to the variant 'Float16' is not named under",
    ),
    "a conversion declared twice": (
        replace(CONVERSION_CONTRACT, conversions=(FLOAT16_CONVERSION, FLOAT16_CONVERSION)),
        "its conversion to the variant float16 is declared twice",
    ),
    "a conversion that writes a refused suffix": (
        change_conversion(0, output="encoder.fp16.pt"),
        "the output of its conversion float16 has the refused suffix .pt",
    ),
    "a conversion of an unknown kind": (
        change_conversion(0, kind="float8"),
        "the conversion to 'encoder.fp16.onnx' is a 'float8', not one of bfloat16, float16",
    ),
    "an operator kept in float32 outside the grammar": (
        change_conversion(0, added_float32_operators=("Pad;Tile",)),
        "keeps 'Pad;Tile' in float32, not an operator under",
    ),
    "an operator kept in float32 twice": (
        change_conversion(0, added_float32_operators=("Floor", "Floor")),
        "the conversion to 'encoder.fp16.onnx' keeps an operator in float32 twice",
    ),
    "a keep_io_types that is not a boolean": (
        change_conversion(0, keep_io_types=1),
        "has keep_io_types 1, not a boolean",
    ),
    "blocks of a size that is not a power of two": (
        change_conversion(1, block_size=48),
        "has blocks of 48 weights, not a power of two from 16 to 256",
    ),
    "blocks under the 16 weights of MatMulNBits": (
        change_conversion(1, block_size=8),
        "has blocks of 8 weights, not a power of two from 16 to 256",
    ),
    "blocks beyond the 256 weights of ONNX Runtime's kernels": (
        change_conversion(1, block_size=512),
        "has blocks of 512 weights, not a power of two from 16 to 256",
    ),
    "an accuracy level beyond ONNX Runtime's": (
        change_conversion(1, accuracy_level=5),
        "has the accuracy level 5, outside 0 to 4",
    ),
    "a symmetry that is not a boolean": (
        change_conversion(1, is_symmetric="yes"),
        "has is_symmetric 'yes', not a boolean",
    ),
    "an 8-bit conversion that quantizes no product": (
        change_conversion(1, quantized_product_count=0),
        "quantizes 0 products, outside 1 to 100000",
    ),
    "two conversions that write one file": (
        change_conversion(1, output="encoder.fp16.onnx"),
        "two conversions write encoder.fp16.onnx",
    ),
    "a conversion that writes the conversion's report": (
        change_conversion(0, output="conversion-report.json"),
        "its output conversion-report.json is declared twice",
    ),
    "a conversion beyond the ceiling of an output": (
        change_variant_conversion(0, max_size_bytes=8 * 2**30 + 1),
        "the size bound of encoder.fp16.onnx in bytes, 8589934593, lies outside",
    ),
    "a conversion whose output no file of the candidate comes from": (
        replace(
            CONVERSION_CONTRACT,
            candidate=(EXPORTED_ENCODER_FILE, EXPORTED_DATA_FILE, EIGHT_BIT_BLOCKS_FILE),
        ),
        "its conversion to the variant float16 writes encoder.fp16.onnx, which no onnx file of "
        "the candidate serving that variant alone comes from",
    ),
    "a converted file that serves another variant": (
        replace(
            CONVERSION_CONTRACT,
            candidate=(
                EXPORTED_ENCODER_FILE,
                EXPORTED_DATA_FILE,
                replace(FLOAT16_FILE, variants=("bfloat16",)),
                EIGHT_BIT_BLOCKS_FILE,
            ),
        ),
        "its conversion to the variant float16 writes encoder.fp16.onnx, which no onnx file",
    ),
    "a comparison of a run that the recipe does not declare": (
        compare(Comparison("float16-gpu", "float32-gpu", NonInferiority(0.20))),
        "its comparison of 'float16-gpu' against 'float32-gpu' names 'float16-gpu', which is not "
        "a run of the recipe",
    ),
    "a comparison against a run that the recipe does not declare": (
        compare(Comparison("float32-gpu", "nemo", Equivalence(0.99, 0.10))),
        "names 'nemo', which is not a run of the recipe",
    ),
    "a comparison of a run with itself": (
        compare(Comparison("float32-cpu", "float32-cpu", REPORTED_ONLY)),
        "its comparison of 'float32-cpu' against 'float32-cpu' compares a run with itself",
    ),
    "a comparison declared twice": (
        compare(CROSS_DEVICE_COMPARISON, replace(CROSS_DEVICE_COMPARISON, rule=REPORTED_ONLY)),
        "its comparison of 'float32-gpu' against 'float32-cpu' is declared twice",
    ),
    "a comparison of runs that share no language": (
        replace(
            change_run(0, languages=("en_us",)),
            comparisons=(CROSS_DEVICE_COMPARISON,),
        ),
        "its comparison of 'float32-gpu' against 'float32-cpu' compares runs that share no "
        "language",
    ),
    "a comparison whose rule leaves its bounds": (
        compare(Comparison("float32-gpu", "float32-cpu", NonInferiority(0.0))),
        "its comparison of 'float32-gpu' against 'float32-cpu': a non-inferiority bounds the "
        "difference of word error rates at 0.0 points",
    ),
}


class TestTheRecipeContract:
    class TestTheJudgeRecipeFunction:
        class TestWhenTheContractDeclaresRuns:
            @pytest.fixture
            def judged_recipe(self) -> JudgedRecipe:
                return judge_recipe(RUN_CONTRACT, RUN_FACTS)

            def test_should_derive_a_row_per_shard_named_after_its_step_run_and_shard(
                self, judged_recipe: JudgedRecipe
            ) -> None:
                assert [derived_row.row.name for derived_row in judged_recipe.derived_rows] == [
                    "transcribe-variants-float32-cpu-0",
                    "transcribe-variants-float32-cpu-1",
                    "transcribe-variants-float32-cpu-2",
                    "transcribe-variants-float32-gpu-0",
                ]

            def test_should_put_the_derived_rows_among_the_rows_of_the_recipe(
                self, judged_recipe: JudgedRecipe
            ) -> None:
                assert [row.name for row in judged_recipe.rows][1:] == [
                    derived_row.row.name for derived_row in judged_recipe.derived_rows
                ]

            def test_should_run_each_row_in_the_image_of_its_device(
                self, judged_recipe: JudgedRecipe
            ) -> None:
                assert {
                    (derived_row.request.device.value, derived_row.row.image.name)
                    for derived_row in judged_recipe.derived_rows
                } == {("cpu", "transcribe-cpu"), ("gpu", "transcribe")}

            def test_should_give_the_gpu_to_the_rows_of_the_gpu_run_alone(
                self, judged_recipe: JudgedRecipe
            ) -> None:
                assert [
                    derived_row.row.name
                    for derived_row in judged_recipe.derived_rows
                    if derived_row.row.uses_gpu
                ] == ["transcribe-variants-float32-gpu-0"]

            def test_should_bind_every_row_of_the_step_to_the_files_of_its_variant_alone(
                self, judged_recipe: JudgedRecipe
            ) -> None:
                assert {
                    (derived_row.row.candidate_files, derived_row.request.files)
                    for derived_row in judged_recipe.derived_rows
                } == {(FLOAT32_FILES, FLOAT32_FILES)}

            def test_should_keep_the_runs_with_their_readers(
                self, judged_recipe: JudgedRecipe
            ) -> None:
                assert [(run.name, run.reader) for run in judged_recipe.runs] == [
                    ("float32-cpu", read_no_lines),
                    ("float32-gpu", read_no_lines),
                ]

            def test_should_keep_the_external_data_with_the_onnx_file_it_serves(
                self, judged_recipe: JudgedRecipe
            ) -> None:
                assert [
                    (candidate_file.name, candidate_file.external_data_of)
                    for candidate_file in judged_recipe.candidate
                    if candidate_file.external_data_of is not None
                ] == [("encoder.onnx.data", "encoder.onnx")]

            def test_should_keep_the_order_of_the_build(self, judged_recipe: JudgedRecipe) -> None:
                assert judged_recipe.build_order == (
                    "fleurs-check",
                    "describe",
                    "inspect",
                    "transcribe-variants",
                )

        class TestWhenTheContractDescribesItsModel:
            @pytest.fixture
            def judged_recipe(self) -> JudgedRecipe:
                return judge_recipe(replace(RUN_CONTRACT, model=MODEL), RUN_FACTS)

            def test_should_carry_the_description_for_the_card(
                self, judged_recipe: JudgedRecipe
            ) -> None:
                assert judged_recipe.model == MODEL

            def test_should_carry_the_path_of_its_template_in_the_recipes_directory(
                self, judged_recipe: JudgedRecipe
            ) -> None:
                assert judged_recipe.card_template == PurePosixPath("recipes/family/model-card.md")

        class TestWhenTheTemplateOfItsCardIsNoRegularFile:
            @pytest.fixture
            def judge_act(self) -> Callable[[], JudgedRecipe]:
                return partial(
                    judge_recipe,
                    replace(RUN_CONTRACT, model=MODEL),
                    replace(RUN_FACTS, has_card_template=False),
                )

            def test_should_refuse_it_and_name_its_path(
                self, judge_act: Callable[[], JudgedRecipe]
            ) -> None:
                with pytest.raises(
                    RecipeContractError,
                    match=re.escape(
                        "its card template model-card.md has no regular file at "
                        "recipes/family/model-card.md"
                    ),
                ):
                    judge_act()

        class TestWhenTheContractDescribesNoModel:
            @pytest.fixture
            def judged_recipe(self) -> JudgedRecipe:
                return judge_recipe(RUN_CONTRACT, replace(RUN_FACTS, has_card_template=False))

            def test_should_carry_none_so_that_the_card_names_what_is_missing(
                self, judged_recipe: JudgedRecipe
            ) -> None:
                assert judged_recipe.model is None

            def test_should_carry_no_template(self, judged_recipe: JudgedRecipe) -> None:
                assert judged_recipe.card_template is None

        class TestWhenARunReadsTheReference:
            @pytest.fixture
            def judged_recipe(self) -> JudgedRecipe:
                return judge_recipe(change_run(0, variant=REFERENCE_VARIANT), RUN_FACTS)

            def test_should_accept_it_since_a_reference_framework_reads_no_variant_of_the_candidate(
                self, judged_recipe: JudgedRecipe
            ) -> None:
                assert [run.variant for run in judged_recipe.runs] == [
                    REFERENCE_VARIANT,
                    "float32",
                ]

        class TestWhenTheContractComparesItsRuns:
            @pytest.fixture
            def judged_recipe(self) -> JudgedRecipe:
                return judge_recipe(COMPARISON_CONTRACT, RUN_FACTS)

            def test_should_keep_each_comparison_with_its_rule(
                self, judged_recipe: JudgedRecipe
            ) -> None:
                assert judged_recipe.comparisons == (CROSS_DEVICE_COMPARISON,)

        class TestWhenTheContractConvertsItsVariants:
            @pytest.fixture
            def judged_recipe(self) -> JudgedRecipe:
                return judge_recipe(CONVERSION_CONTRACT, FACTS)

            def test_should_add_the_cores_conversions_after_the_recipes_rows(
                self, judged_recipe: JudgedRecipe
            ) -> None:
                assert [row.name for row in judged_recipe.rows] == ["export", "convert"]

            def test_should_bind_the_input_and_its_external_data_each_by_itself(
                self, judged_recipe: JudgedRecipe
            ) -> None:
                assert [
                    (earlier_output.step, earlier_output.output.name)
                    for earlier_output in judged_recipe.rows[-1].earlier_outputs
                ] == [("export", "encoder.onnx"), ("export", "encoder.onnx.data")]

            def test_should_declare_each_converted_file_with_its_size_bound(
                self, judged_recipe: JudgedRecipe
            ) -> None:
                assert judged_recipe.rows[-1].outputs[:2] == (
                    DeclaredOutput("encoder.fp16.onnx", CONVERTED_SIZE_BOUND_BYTES),
                    DeclaredOutput("encoder.int8.onnx", CONVERTED_SIZE_BOUND_BYTES),
                )

            def test_should_keep_the_conversions_in_their_order(
                self, judged_recipe: JudgedRecipe
            ) -> None:
                assert judged_recipe.conversions == (
                    FLOAT16_CONVERSION,
                    EIGHT_BIT_BLOCKS_CONVERSION,
                )

            def test_should_let_the_build_convert(self, judged_recipe: JudgedRecipe) -> None:
                assert judged_recipe.build_order == ("export", "convert", "inspect")

        class TestWhenTheContractKeepsWithinItsBounds:
            @pytest.fixture
            def judged_recipe(self) -> JudgedRecipe:
                return judge_recipe(CONTRACT, FACTS)

            def test_should_tag_the_recipes_images_after_the_recipe(
                self, judged_recipe: JudgedRecipe
            ) -> None:
                assert [image.tag for image in judged_recipe.images] == [
                    "mostt-model-builder-family-export",
                    "mostt-model-builder-family-transcribe",
                ]

            def test_should_run_an_image_with_an_extra_on_its_environments_python(
                self, judged_recipe: JudgedRecipe
            ) -> None:
                assert {image.python for image in judged_recipe.images} == {ENVIRONMENT_PYTHON}

            def test_should_place_each_containerfile_under_the_recipes_images(
                self, judged_recipe: JudgedRecipe
            ) -> None:
                assert [image.containerfile for image in judged_recipe.images] == [
                    PurePosixPath("recipes/family/images/export/Containerfile"),
                    PurePosixPath("recipes/family/images/transcribe/Containerfile"),
                ]

            def test_should_keep_the_steps_in_their_order(
                self, judged_recipe: JudgedRecipe
            ) -> None:
                assert [row.name for row in judged_recipe.rows] == ["describe", "transcribe"]

            def test_should_bind_each_source_by_its_pin(self, judged_recipe: JudgedRecipe) -> None:
                assert judged_recipe.rows[0].sources == (
                    SourceReference(WEIGHTS_REPOSITORY, "model.safetensors"),
                    SourceReference(ARCHIVE_REPOSITORY, "model.nemo"),
                )

            def test_should_bind_an_earlier_output_with_every_name_its_step_declares(
                self, judged_recipe: JudgedRecipe
            ) -> None:
                assert judged_recipe.rows[1].earlier_outputs == (
                    EarlierOutput("describe", NAMES_OUTPUT, ("names", "peaks.json")),
                )

            def test_should_give_no_row_the_network(self, judged_recipe: JudgedRecipe) -> None:
                assert [row.name for row in judged_recipe.rows if row.has_network] == []

            def test_should_hold_the_cores_pins_and_the_recipes(
                self, judged_recipe: JudgedRecipe
            ) -> None:
                assert judged_recipe.pins == (BENCH_PIN, WEIGHTS_PIN, NESTED_PIN, ARCHIVE_PIN)

            def test_should_keep_each_parity_with_its_tolerance(
                self, judged_recipe: JudgedRecipe
            ) -> None:
                assert judged_recipe.parities == (NAMES_PARITY,)

            def test_should_read_the_format_of_each_candidate_file(
                self, judged_recipe: JudgedRecipe
            ) -> None:
                assert [file.file_format for file in judged_recipe.candidate] == [
                    FileFormat.TEXT,
                    FileFormat.TEXT,
                ]

        class TestWhenAnImageHasNoExtra:
            @pytest.fixture
            def judged_recipe(self) -> JudgedRecipe:
                return judge_recipe(change_image(1, extra=None), FACTS)

            def test_should_run_it_on_the_bases_python_since_it_has_no_environment(
                self, judged_recipe: JudgedRecipe
            ) -> None:
                assert judged_recipe.images[1].python == BASE_PYTHON

        class TestWhenAStepRunsInACoreImage:
            @pytest.fixture
            def judged_recipe(self) -> JudgedRecipe:
                return judge_recipe(
                    replace(
                        CONTRACT,
                        steps=(
                            replace(
                                DESCRIBE_STEP,
                                name="bench-probe",
                                image="bench",
                                entry_module=FLEURS_CHECK_MODULE,
                                mounts=(),
                            ),
                        ),
                        candidate=(),
                        parities=(),
                    ),
                    FACTS,
                )

            def test_should_accept_an_entry_module_of_the_cores_steps(
                self, judged_recipe: JudgedRecipe
            ) -> None:
                assert judged_recipe.rows[0].image.owner == "core"

        class TestWhenTheRecipePinsAFileOfTheCore:
            @pytest.fixture
            def judge_act(self) -> Callable[[], JudgedRecipe]:
                return partial(judge_recipe, CONTRACT, replace(FACTS, recipe_pins=(BENCH_PIN,)))

            def test_should_refuse_it_since_one_file_has_one_pin(
                self, judge_act: Callable[[], JudgedRecipe]
            ) -> None:
                with pytest.raises(
                    RecipeContractError,
                    match=re.escape("its sources and the core's: data/test.tsv of google/fleurs"),
                ):
                    judge_act()

        @pytest.mark.parametrize(
            ("contract", "reason"),
            list(REFUSED_CONTRACTS.values()),
            ids=list(REFUSED_CONTRACTS.keys()),
        )
        class TestWhenTheContractBreaksARule:
            @pytest.fixture
            def judge_act(self, contract: RecipeContract) -> Callable[[], JudgedRecipe]:
                return partial(judge_recipe, contract, FACTS)

            def test_should_refuse_it_and_say_why(
                self, judge_act: Callable[[], JudgedRecipe], reason: str
            ) -> None:
                with pytest.raises(RecipeContractError, match=re.escape(reason)):
                    judge_act()

    class TestTheBuildConversionRequestFunction:
        class TestWhenTheRecipeConvertsItsVariants:
            @pytest.fixture
            def conversion_request(self) -> ConversionRequest:
                return build_conversion_request(judge_recipe(CONVERSION_CONTRACT, FACTS))

            def test_should_ask_for_each_conversion_of_the_input_in_the_recipes_order(
                self, conversion_request: ConversionRequest
            ) -> None:
                assert conversion_request == ConversionRequest(
                    "export",
                    "encoder.onnx",
                    (FLOAT16_CONVERSION.conversion, EIGHT_BIT_BLOCKS_CONVERSION.conversion),
                )

        class TestWhenTheRecipeConvertsNothing:
            @pytest.fixture
            def build_act(self) -> Callable[[], ConversionRequest]:
                return partial(build_conversion_request, judge_recipe(CONTRACT, FACTS))

            def test_should_refuse_to_ask_for_a_conversion(
                self, build_act: Callable[[], ConversionRequest]
            ) -> None:
                with pytest.raises(
                    RecipeContractError,
                    match=re.escape("the recipe family converts none of its variants"),
                ):
                    build_act()

    class TestTheFindGraphSourceFunction:
        @pytest.mark.parametrize(
            ("origin", "expected_source"),
            [
                pytest.param(
                    EXPORTED_ENCODER,
                    GraphSource.CONVERSION_INPUT,
                    id="the exported encoder, the conversions' input",
                ),
                pytest.param(
                    StepOutputOrigin("convert", "encoder.int8.onnx"),
                    EightBitBlocksOutput(265),
                    id="the 8-bit conversion's output, with the count of products it declares",
                ),
                pytest.param(
                    StepOutputOrigin("convert", "encoder.fp16.onnx"),
                    GraphSource.OTHER,
                    id="the float16 conversion's output",
                ),
                pytest.param(
                    StepOutputOrigin("export", "decoder.onnx"),
                    GraphSource.OTHER,
                    id="a decoder that no conversion reads, whatever name it is published under",
                ),
                pytest.param(
                    HostContentOrigin(b"{}"), GraphSource.OTHER, id="a content the host writes"
                ),
            ],
        )
        class TestWhenGivenTheOriginOfAFile:
            @pytest.fixture
            def graph_source(
                self, origin: StepOutputOrigin | HostContentOrigin
            ) -> GraphSource | EightBitBlocksOutput:
                return find_graph_source(origin, EXPORTED_ENCODER, CONVERSION_CONTRACT.conversions)

            def test_should_hold_it_to_the_rules_of_where_it_comes_from_never_of_its_name(
                self,
                graph_source: GraphSource | EightBitBlocksOutput,
                expected_source: GraphSource | EightBitBlocksOutput,
            ) -> None:
                assert graph_source == expected_source

    class TestTheLocateEntryModuleFileFunction:
        @pytest.mark.parametrize(
            ("entry_module", "module_file"),
            [
                pytest.param(
                    DESCRIBE_MODULE,
                    PurePosixPath("recipes/family/src/mostt_recipe_family/steps/description.py"),
                    id="a module of the recipe, under the recipe's sources",
                ),
                pytest.param(
                    FLEURS_CHECK_MODULE,
                    PurePosixPath("src/mostt_model_builder/steps/fleurs_check.py"),
                    id="a module of the core, under the core's sources",
                ),
                pytest.param(
                    "mostt_recipe_other.steps.description",
                    None,
                    id="a module of another recipe, nowhere",
                ),
                pytest.param("os.path", None, id="a module of the standard library, nowhere"),
                pytest.param("mostt_recipe_family", None, id="a package alone, nowhere"),
            ],
        )
        class TestWhenGivenAnEntryModule:
            @pytest.fixture
            def located_file(self, entry_module: str) -> PurePosixPath | None:
                return locate_entry_module_file(entry_module, RECIPE_NAME)

            def test_should_place_its_file_lexically_without_importing_it(
                self, located_file: PurePosixPath | None, module_file: PurePosixPath | None
            ) -> None:
                assert located_file == module_file

    class TestTheLocateCardTemplateFunction:
        @pytest.mark.parametrize(
            ("recipe_name", "model", "card_template"),
            [
                pytest.param(
                    RECIPE_NAME,
                    MODEL,
                    PurePosixPath("recipes/family/model-card.md"),
                    id="a file name, in the recipe's directory",
                ),
                pytest.param(RECIPE_NAME, None, None, id="no model, nowhere"),
                pytest.param(
                    RECIPE_NAME,
                    replace(MODEL, template_path=PurePosixPath("../model-card.md")),
                    None,
                    id="a path outside the file-name grammar, nowhere",
                ),
                pytest.param(
                    "../family", MODEL, None, id="a recipe name outside its grammar, nowhere"
                ),
            ],
        )
        class TestWhenGivenTheModelOfARecipe:
            @pytest.fixture
            def located_template(
                self, recipe_name: str, model: ModelDescription | None
            ) -> PurePosixPath | None:
                return locate_card_template(recipe_name, model)

            def test_should_place_it_only_once_both_names_pass_their_grammars(
                self,
                located_template: PurePosixPath | None,
                card_template: PurePosixPath | None,
            ) -> None:
                assert located_template == card_template

    class TestTheFindRunnableRowFunction:
        @pytest.fixture
        def judged_recipe(self) -> JudgedRecipe:
            return judge_recipe(CONTRACT, FACTS)

        class TestWhenTheStepIsTheRecipes:
            @pytest.fixture
            def row(self, judged_recipe: JudgedRecipe) -> StepRow:
                return find_runnable_row(judged_recipe, "transcribe")

            def test_should_find_its_row(self, row: StepRow) -> None:
                assert row.entry_module == TRANSCRIBE_MODULE

        class TestWhenTheStepIsTheCores:
            @pytest.fixture
            def row(self, judged_recipe: JudgedRecipe) -> StepRow:
                return find_runnable_row(judged_recipe, "fleurs-check")

            def test_should_find_its_row(self, row: StepRow) -> None:
                assert row == FLEURS_CHECK_ROW

        class TestWhenTheStepIsTheCleaning:
            @pytest.fixture
            def find_act(self, judged_recipe: JudgedRecipe) -> Callable[[], StepRow]:
                return partial(find_runnable_row, judged_recipe, "clean")

            def test_should_refuse_it_since_clean_has_its_own_command(
                self, find_act: Callable[[], StepRow]
            ) -> None:
                with pytest.raises(
                    RecipeContractError,
                    match="neither the core nor the recipe family has a step named 'clean'",
                ):
                    find_act()

    class TestTheFindCandidateFileMaxSizeBytesFunction:
        @pytest.fixture
        def judged_recipe(self) -> JudgedRecipe:
            return judge_recipe(CONTRACT, FACTS)

        class TestWhenTheFileIsCopiedFromTheOutputOfAStep:
            @pytest.fixture
            def max_size_bytes(self, judged_recipe: JudgedRecipe) -> int:
                return find_candidate_file_max_size_bytes(NAMES_FILE.origin, judged_recipe.rows)

            def test_should_give_the_bound_its_step_declares(self, max_size_bytes: int) -> None:
                assert max_size_bytes == NAMES_OUTPUT.max_size_bytes

        class TestWhenTheFileHoldsContentOfTheHost:
            @pytest.fixture
            def max_size_bytes(self, judged_recipe: JudgedRecipe) -> int:
                return find_candidate_file_max_size_bytes(
                    CONFIGURATION_FILE.origin, judged_recipe.rows
                )

            def test_should_give_the_length_of_that_content(self, max_size_bytes: int) -> None:
                assert max_size_bytes == len(CONFIGURATION)
