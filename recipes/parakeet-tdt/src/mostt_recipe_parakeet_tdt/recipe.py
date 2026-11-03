from mostt_model_builder.domain.containers.container_steps import (
    PEAKS_OUTPUT,
    DeclaredOutput,
    Device,
    StepLimits,
)
from mostt_model_builder.domain.contract.recipe_contract import (
    RecipeContract,
    RecipeImage,
    RecipeStep,
    SourceMount,
)

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
# A name list holds one line per tensor, under a thousand of them.
NAME_LIST_MAX_SIZE_BYTES = 2**20

EXPORT_IMAGE = RecipeImage(
    name=EXPORT_IMAGE_NAME,
    device=Device.CPU,
    extra=f"{RECIPE_NAME}-{EXPORT_IMAGE_NAME}",
    reads_code_running_sources=True,
)
ULTRA_WEIGHTS_MOUNT = SourceMount(ULTRA_REPOSITORY, ULTRA_WEIGHTS)
NEMO_ARCHIVE_MOUNT = SourceMount(NEMO_REPOSITORY, NEMO_ARCHIVE)
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
RECIPE = RecipeContract(
    name=RECIPE_NAME,
    images=(EXPORT_IMAGE,),
    steps=(DESCRIBE_STEP,),
    candidate=(),
)
