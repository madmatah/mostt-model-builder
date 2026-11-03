from mostt_model_builder.domain.acceptance.parity_reports import PARITY_REPORT_MAX_SIZE_BYTES
from mostt_model_builder.domain.bench.fleurs_index import locate_bench_files
from mostt_model_builder.domain.containers.container_steps import (
    FLEURS_REPOSITORY,
    PEAKS_OUTPUT,
    DeclaredOutput,
    Device,
    StepLimits,
)
from mostt_model_builder.domain.contract.recipe_contract import (
    Parity,
    RecipeContract,
    RecipeImage,
    RecipeStep,
    SourceMount,
)
from mostt_recipe_parakeet_tdt.domain.export.tensor_fixtures import NAME_LIST_MAX_SIZE_BYTES

RECIPE_NAME = "parakeet-tdt"
ULTRA_REPOSITORY = "moondream/parakeet-ultra"
ULTRA_CONFIGURATION = "config.json"
ULTRA_TOKENIZER = "tokenizer.json"
ULTRA_WEIGHTS = "model.safetensors"
NEMO_REPOSITORY = "nvidia/parakeet-tdt-0.6b-v3"
NEMO_ARCHIVE = "parakeet-tdt-0.6b-v3.nemo"
EXPORT_IMAGE_NAME = "export"
DESCRIBE_STEP_NAME = "describe"
ULTRA_TENSORS_FILE_NAME = "moondream-tensors"
NEMO_STATE_DICT_FILE_NAME = "nemo-state-dict"
EXPORT_STEP_NAME = "export"
ENCODER_FILE_NAME = "encoder-model.onnx"
ENCODER_DATA_FILE_NAME = "encoder-model.onnx.data"
DECODER_JOINT_FILE_NAME = "decoder_joint-model.onnx"
VOCABULARY_FILE_NAME = "vocab.txt"
ENCODER_PARITY_FILE_NAME = "encoder-parity.json"
ENCODER_PARITY_NAME = "encoder-onnx-against-nemo"
PARITY_LANGUAGE = "fr_fr"

EXPORT_IMAGE = RecipeImage(
    name=EXPORT_IMAGE_NAME,
    device=Device.CPU,
    extra=f"{RECIPE_NAME}-{EXPORT_IMAGE_NAME}",
    reads_code_running_sources=True,
)
ULTRA_WEIGHTS_MOUNT = SourceMount(ULTRA_REPOSITORY, ULTRA_WEIGHTS)
ULTRA_TOKENIZER_AND_WEIGHTS_MOUNTS = (
    SourceMount(ULTRA_REPOSITORY, ULTRA_TOKENIZER),
    ULTRA_WEIGHTS_MOUNT,
)
NEMO_ARCHIVE_MOUNT = SourceMount(NEMO_REPOSITORY, NEMO_ARCHIVE)
PARITY_BENCH_MOUNTS = tuple(
    SourceMount(FLEURS_REPOSITORY, path) for path in locate_bench_files(PARITY_LANGUAGE)
)
# NeMo's build of the model, with Moondream's weights loaded, peaked at 3.4 GiB and 60 processes.
DESCRIBE_STEP = RecipeStep(
    name=DESCRIBE_STEP_NAME,
    image=EXPORT_IMAGE_NAME,
    entry_module="mostt_recipe_parakeet_tdt.steps.nemo_description",
    mounts=(ULTRA_WEIGHTS_MOUNT, NEMO_ARCHIVE_MOUNT),
    outputs=(
        DeclaredOutput(ULTRA_TENSORS_FILE_NAME, NAME_LIST_MAX_SIZE_BYTES),
        DeclaredOutput(NEMO_STATE_DICT_FILE_NAME, NAME_LIST_MAX_SIZE_BYTES),
        PEAKS_OUTPUT,
    ),
    limits=StepLimits(memory_mib=5120, process_count=128, cpu_count=8, temporary_directory_mib=512),
)
# The export peaked at 11.1 GiB, 66 processes and 2.37 GiB of /scratch in the trial build, its
# /scratch counted against its memory, and wrote a graph of the encoder under 1 MB, its external
# data of 2 476 376 064 bytes and a decoder and joint of 73 MB.
EXPORT_STEP = RecipeStep(
    name=EXPORT_STEP_NAME,
    image=EXPORT_IMAGE_NAME,
    entry_module="mostt_recipe_parakeet_tdt.steps.nemo_export",
    mounts=(*ULTRA_TOKENIZER_AND_WEIGHTS_MOUNTS, NEMO_ARCHIVE_MOUNT, *PARITY_BENCH_MOUNTS),
    outputs=(
        DeclaredOutput(ENCODER_FILE_NAME, 16 * 2**20),
        DeclaredOutput(ENCODER_DATA_FILE_NAME, 3 * 2**30),
        DeclaredOutput(DECODER_JOINT_FILE_NAME, 128 * 2**20),
        DeclaredOutput(VOCABULARY_FILE_NAME, 2**20),
        DeclaredOutput(ENCODER_PARITY_FILE_NAME, PARITY_REPORT_MAX_SIZE_BYTES),
        PEAKS_OUTPUT,
    ),
    limits=StepLimits(
        memory_mib=14_336,
        process_count=128,
        cpu_count=8,
        temporary_directory_mib=512,
        scratch_mib=3072,
    ),
)
# The trial build measured 4.18e-6 between the exported encoder on ONNX Runtime's CPU provider and
# NeMo's unpatched encoder, on 20 utterances of FLEURS fr: the tolerance is over ten times that.
ENCODER_PARITY = Parity(
    name=ENCODER_PARITY_NAME,
    step=EXPORT_STEP_NAME,
    output=ENCODER_PARITY_FILE_NAME,
    absolute_difference_tolerance=5e-5,
)
RECIPE = RecipeContract(
    name=RECIPE_NAME,
    images=(EXPORT_IMAGE,),
    steps=(DESCRIBE_STEP, EXPORT_STEP),
    candidate=(),
    parities=(ENCODER_PARITY,),
)
