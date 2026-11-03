from mostt_model_builder.domain.acceptance.parity_reports import PARITY_REPORT_MAX_SIZE_BYTES
from mostt_model_builder.domain.bench.fleurs_index import BENCH_LANGUAGES, locate_bench_files
from mostt_model_builder.domain.bench.hypotheses import (
    RUN_LINES_FILE_NAME,
    RUN_LINES_MAX_SIZE_BYTES,
)
from mostt_model_builder.domain.containers.container_steps import (
    FLEURS_REPOSITORY,
    PEAKS_OUTPUT,
    DeclaredOutput,
    Device,
    StepLimits,
)
from mostt_model_builder.domain.contract.recipe_contract import (
    CANDIDATE_MOUNT,
    REFERENCE_VARIANT,
    CandidateFile,
    DeviceTemplate,
    EarlierOutputMount,
    HostContentOrigin,
    Parity,
    RecipeContract,
    RecipeImage,
    RecipeStep,
    Run,
    SourceMount,
    StepOutputOrigin,
)
from mostt_model_builder.domain.formats.file_formats import FileFormat
from mostt_model_builder.domain.onnx.node_placement import (
    PLACEMENT_FILE_NAME,
    PLACEMENT_MAX_SIZE_BYTES,
)
from mostt_recipe_parakeet_tdt.domain.export.tensor_fixtures import NAME_LIST_MAX_SIZE_BYTES
from mostt_recipe_parakeet_tdt.domain.transcripts.run_readers import (
    read_onnx_asr_lines,
    read_token_id_lines,
)
from mostt_recipe_parakeet_tdt.domain.variants.candidate_layout import (
    CONFIGURATION_FILE_NAME,
    ONNX_ASR_CONFIGURATION,
    VOCABULARY_FILE_NAME,
    name_decoder_joint_file,
    name_encoder_file,
)

RECIPE_NAME = "parakeet-tdt"
ULTRA_REPOSITORY = "moondream/parakeet-ultra"
ULTRA_CONFIGURATION = "config.json"
ULTRA_TOKENIZER = "tokenizer.json"
ULTRA_WEIGHTS = "model.safetensors"
NEMO_REPOSITORY = "nvidia/parakeet-tdt-0.6b-v3"
NEMO_ARCHIVE = "parakeet-tdt-0.6b-v3.nemo"
PROCESSOR_CONFIGURATION = "processor_config.json"
TOKENIZER_CONFIGURATION = "tokenizer_config.json"
GENERATION_CONFIGURATION = "generation_config.json"
EXPORT_IMAGE_NAME = "export"
TRANSCRIBE_CPU_IMAGE_NAME = "transcribe-cpu"
TRANSCRIBE_GPU_IMAGE_NAME = "transcribe-gpu"
TRANSFORMERS_IMAGE_NAME = "transformers-reference"
DESCRIBE_STEP_NAME = "describe"
ULTRA_TENSORS_FILE_NAME = "moondream-tensors"
NEMO_STATE_DICT_FILE_NAME = "nemo-state-dict"
EXPORT_STEP_NAME = "export"
FLOAT32_VARIANT = "float32"
ENCODER_FILE_NAME = name_encoder_file(FLOAT32_VARIANT)
ENCODER_DATA_FILE_NAME = f"{ENCODER_FILE_NAME}.data"
DECODER_JOINT_FILE_NAME = name_decoder_joint_file(FLOAT32_VARIANT)
REFERENCE_STEP_NAME = "reference"
REFERENCE_TENSORS_FILE_NAME = "reference.safetensors"
REFERENCE_STEPS_FILE_NAME = "reference.json"
TRANSCRIBE_STEP_NAME = "transcribe"
NEMO_RUN_NAME = "nemo"
ENCODER_PARITY_FILE_NAME = "encoder-parity.json"
ENCODER_PARITY_NAME = "encoder-onnx-against-nemo"
PARITY_LANGUAGE = "fr_fr"
TRANSFORMERS_STEP_NAME = "transformers-reference"
TRANSFORMERS_RUN_NAME = "transformers"
TRANSFORMERS_ENCODER_PARITY_FILE_NAME = "transformers-encoder-parity.json"
TRANSFORMERS_ENCODER_PARITY_NAME = "encoder-transformers-against-nemo"
TRANSFORMERS_JOINT_PARITY_FILE_NAME = "transformers-joint-parity.json"
TRANSFORMERS_JOINT_PARITY_NAME = "joint-transformers-against-nemo"

EXPORT_IMAGE = RecipeImage(
    name=EXPORT_IMAGE_NAME,
    device=Device.CPU,
    extra=f"{RECIPE_NAME}-{EXPORT_IMAGE_NAME}",
    reads_code_running_sources=True,
)
TRANSCRIBE_CPU_IMAGE = RecipeImage(
    name=TRANSCRIBE_CPU_IMAGE_NAME,
    device=Device.CPU,
    extra=f"{RECIPE_NAME}-{TRANSCRIBE_CPU_IMAGE_NAME}",
)
TRANSCRIBE_GPU_IMAGE = RecipeImage(
    name=TRANSCRIBE_GPU_IMAGE_NAME,
    device=Device.GPU,
    extra=f"{RECIPE_NAME}-{TRANSCRIBE_GPU_IMAGE_NAME}",
)
TRANSFORMERS_IMAGE = RecipeImage(
    name=TRANSFORMERS_IMAGE_NAME,
    device=Device.CPU,
    extra=f"{RECIPE_NAME}-{TRANSFORMERS_IMAGE_NAME}",
)
ULTRA_WEIGHTS_MOUNT = SourceMount(ULTRA_REPOSITORY, ULTRA_WEIGHTS)
ULTRA_TOKENIZER_AND_WEIGHTS_MOUNTS = (
    SourceMount(ULTRA_REPOSITORY, ULTRA_TOKENIZER),
    ULTRA_WEIGHTS_MOUNT,
)
ULTRA_MOUNTS = (
    SourceMount(ULTRA_REPOSITORY, ULTRA_CONFIGURATION),
    *ULTRA_TOKENIZER_AND_WEIGHTS_MOUNTS,
)
NEMO_ARCHIVE_MOUNT = SourceMount(NEMO_REPOSITORY, NEMO_ARCHIVE)
PROCESSOR_MOUNTS = (
    SourceMount(NEMO_REPOSITORY, PROCESSOR_CONFIGURATION),
    SourceMount(NEMO_REPOSITORY, TOKENIZER_CONFIGURATION),
    SourceMount(NEMO_REPOSITORY, GENERATION_CONFIGURATION),
)
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
EXPORT_LIMITS = StepLimits(
    memory_mib=14_336,
    process_count=128,
    cpu_count=8,
    temporary_directory_mib=512,
    scratch_mib=3072,
)
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
    limits=EXPORT_LIMITS,
)
# The trial build measured 4.18e-6 between the exported encoder on ONNX Runtime's CPU provider and
# NeMo's unpatched encoder, on 20 utterances of FLEURS fr: the tolerance is over ten times that.
ENCODER_PARITY = Parity(
    name=ENCODER_PARITY_NAME,
    step=EXPORT_STEP_NAME,
    output=ENCODER_PARITY_FILE_NAME,
    absolute_difference_tolerance=5e-5,
)
BENCH_MOUNTS = tuple(
    SourceMount(FLEURS_REPOSITORY, path)
    for language in BENCH_LANGUAGES
    for path in locate_bench_files(language)
)
RUN_LINES_OUTPUT = DeclaredOutput(RUN_LINES_FILE_NAME, RUN_LINES_MAX_SIZE_BYTES)
# NeMo's reference peaked at 6.2 GiB, its /scratch included, and 58 processes.
REFERENCE_CPU_LIMITS = StepLimits(
    memory_mib=8192,
    process_count=128,
    cpu_count=8,
    temporary_directory_mib=512,
    scratch_mib=1024,
)
# The reference runs NeMo unpatched, on the CPU in float32, so that no TF32 enters what the export
# is measured against.
REFERENCE_STEP = RecipeStep(
    name=REFERENCE_STEP_NAME,
    image=None,
    entry_module="mostt_recipe_parakeet_tdt.steps.nemo_reference",
    mounts=(*ULTRA_TOKENIZER_AND_WEIGHTS_MOUNTS, NEMO_ARCHIVE_MOUNT, *BENCH_MOUNTS),
    outputs=(
        DeclaredOutput(REFERENCE_TENSORS_FILE_NAME, 64 * 2**20),
        DeclaredOutput(REFERENCE_STEPS_FILE_NAME, 2**20),
        PEAKS_OUTPUT,
    ),
    limits=None,
    devices=(
        DeviceTemplate(
            device=Device.CPU,
            image=EXPORT_IMAGE_NAME,
            limits=REFERENCE_CPU_LIMITS,
            outputs=(RUN_LINES_OUTPUT,),
        ),
    ),
)
# Transformers' check peaked at 5.4 GiB and 79 processes.
TRANSFORMERS_LIMITS = StepLimits(
    memory_mib=7168, process_count=128, cpu_count=8, temporary_directory_mib=512
)
# Transformers reads Moondream's files as they are published, so that a weight that the mapping
# onto NeMo's names misplaced shows as a difference against NeMo's reference.
TRANSFORMERS_STEP = RecipeStep(
    name=TRANSFORMERS_STEP_NAME,
    image=None,
    entry_module="mostt_recipe_parakeet_tdt.steps.transformers_reference",
    mounts=(
        *ULTRA_MOUNTS,
        *PROCESSOR_MOUNTS,
        *PARITY_BENCH_MOUNTS,
        EarlierOutputMount(REFERENCE_STEP_NAME, REFERENCE_TENSORS_FILE_NAME),
        EarlierOutputMount(REFERENCE_STEP_NAME, REFERENCE_STEPS_FILE_NAME),
    ),
    outputs=(
        DeclaredOutput(TRANSFORMERS_ENCODER_PARITY_FILE_NAME, PARITY_REPORT_MAX_SIZE_BYTES),
        DeclaredOutput(TRANSFORMERS_JOINT_PARITY_FILE_NAME, PARITY_REPORT_MAX_SIZE_BYTES),
        PEAKS_OUTPUT,
    ),
    limits=None,
    devices=(
        DeviceTemplate(
            device=Device.CPU,
            image=TRANSFORMERS_IMAGE_NAME,
            limits=TRANSFORMERS_LIMITS,
            outputs=(RUN_LINES_OUTPUT,),
        ),
    ),
)
# The trial build measured 1.34e-6 between Transformers' encoder and NeMo's on the same features,
# over 5 utterances, and 5.3e-5 between their joints over 25 steps once each head, the tokens and
# the durations, is log-softmaxed on its own, since NeMo log-softmaxes the two together: each
# tolerance is about ten times that.
TRANSFORMERS_ENCODER_PARITY = Parity(
    name=TRANSFORMERS_ENCODER_PARITY_NAME,
    step=TRANSFORMERS_STEP_NAME,
    output=TRANSFORMERS_ENCODER_PARITY_FILE_NAME,
    absolute_difference_tolerance=1.5e-5,
)
TRANSFORMERS_JOINT_PARITY = Parity(
    name=TRANSFORMERS_JOINT_PARITY_NAME,
    step=TRANSFORMERS_STEP_NAME,
    output=TRANSFORMERS_JOINT_PARITY_FILE_NAME,
    absolute_difference_tolerance=5.3e-4,
)
# A transcription runs one process of 8 threads on the GPU, which peaked at 3.4 GiB and 99
# processes, and on the CPU four of 2 threads, each in its own container, which peaked at 3.4 GiB
# and 33 processes each.
TRANSCRIBE_GPU_LIMITS = StepLimits(
    memory_mib=5120, process_count=128, cpu_count=8, temporary_directory_mib=256
)
TRANSCRIBE_CPU_LIMITS = StepLimits(
    memory_mib=5120, process_count=64, cpu_count=2, temporary_directory_mib=256
)
TRANSCRIBE_STEP = RecipeStep(
    name=TRANSCRIBE_STEP_NAME,
    image=None,
    entry_module="mostt_recipe_parakeet_tdt.steps.onnx_asr_transcription",
    mounts=(CANDIDATE_MOUNT, *BENCH_MOUNTS),
    outputs=(PEAKS_OUTPUT,),
    limits=None,
    devices=(
        DeviceTemplate(
            device=Device.CPU,
            image=TRANSCRIBE_CPU_IMAGE_NAME,
            limits=TRANSCRIBE_CPU_LIMITS,
            outputs=(RUN_LINES_OUTPUT,),
        ),
        DeviceTemplate(
            device=Device.GPU,
            image=TRANSCRIBE_GPU_IMAGE_NAME,
            limits=TRANSCRIBE_GPU_LIMITS,
            outputs=(
                RUN_LINES_OUTPUT,
                DeclaredOutput(PLACEMENT_FILE_NAME, PLACEMENT_MAX_SIZE_BYTES),
            ),
        ),
    ),
)
ENCODER_ORIGIN = StepOutputOrigin(EXPORT_STEP_NAME, ENCODER_FILE_NAME)
CANDIDATE = (
    CandidateFile(ENCODER_FILE_NAME, FileFormat.ONNX, (FLOAT32_VARIANT,), ENCODER_ORIGIN),
    CandidateFile(
        ENCODER_DATA_FILE_NAME,
        FileFormat.ONNX_EXTERNAL_DATA,
        (FLOAT32_VARIANT,),
        StepOutputOrigin(EXPORT_STEP_NAME, ENCODER_DATA_FILE_NAME),
        external_data_of=ENCODER_FILE_NAME,
    ),
    CandidateFile(
        DECODER_JOINT_FILE_NAME,
        FileFormat.ONNX,
        (FLOAT32_VARIANT,),
        StepOutputOrigin(EXPORT_STEP_NAME, DECODER_JOINT_FILE_NAME),
    ),
    CandidateFile(
        VOCABULARY_FILE_NAME,
        FileFormat.TEXT,
        (FLOAT32_VARIANT,),
        StepOutputOrigin(EXPORT_STEP_NAME, VOCABULARY_FILE_NAME),
    ),
    CandidateFile(
        CONFIGURATION_FILE_NAME,
        FileFormat.TEXT,
        (FLOAT32_VARIANT,),
        HostContentOrigin(ONNX_ASR_CONFIGURATION),
    ),
)
RUNS = (
    Run(
        name=NEMO_RUN_NAME,
        step=REFERENCE_STEP_NAME,
        device=Device.CPU,
        variant=REFERENCE_VARIANT,
        languages=BENCH_LANGUAGES,
        shard_count=1,
        reader=read_token_id_lines,
    ),
    Run(
        name=TRANSFORMERS_RUN_NAME,
        step=TRANSFORMERS_STEP_NAME,
        device=Device.CPU,
        variant=REFERENCE_VARIANT,
        languages=(PARITY_LANGUAGE,),
        shard_count=1,
        reader=read_token_id_lines,
    ),
    Run(
        name="float32-cpu",
        step=TRANSCRIBE_STEP_NAME,
        device=Device.CPU,
        variant=FLOAT32_VARIANT,
        languages=BENCH_LANGUAGES,
        shard_count=4,
        reader=read_onnx_asr_lines,
    ),
    Run(
        name="float32-gpu",
        step=TRANSCRIBE_STEP_NAME,
        device=Device.GPU,
        variant=FLOAT32_VARIANT,
        languages=BENCH_LANGUAGES,
        shard_count=1,
        reader=read_onnx_asr_lines,
    ),
)
RECIPE = RecipeContract(
    name=RECIPE_NAME,
    images=(EXPORT_IMAGE, TRANSFORMERS_IMAGE, TRANSCRIBE_CPU_IMAGE, TRANSCRIBE_GPU_IMAGE),
    steps=(DESCRIBE_STEP, EXPORT_STEP, REFERENCE_STEP, TRANSFORMERS_STEP, TRANSCRIBE_STEP),
    candidate=CANDIDATE,
    parities=(ENCODER_PARITY, TRANSFORMERS_ENCODER_PARITY, TRANSFORMERS_JOINT_PARITY),
    runs=RUNS,
    conversion_input=ENCODER_ORIGIN,
    build_order=(
        "fleurs-check",
        EXPORT_STEP_NAME,
        "inspect",
        REFERENCE_STEP_NAME,
        TRANSFORMERS_STEP_NAME,
        TRANSCRIBE_STEP_NAME,
    ),
)
