from pathlib import PurePosixPath

from mostt_model_builder.domain.containers.container_steps import FLEURS_REPOSITORY
from mostt_model_builder.domain.contract.model_description import (
    ModelDescription,
    PinnedRevision,
    ReferenceRunDescription,
    VariantDescription,
)
from mostt_recipe_parakeet_tdt.domain.variants.comparisons import NEMO_RUN_NAME
from mostt_recipe_parakeet_tdt.domain.variants.variants import (
    BFLOAT16_VARIANT,
    EIGHT_BIT_BLOCKS_VARIANT,
    FLOAT16_VARIANT,
    FLOAT32_VARIANT,
)

ULTRA_REPOSITORY = "moondream/parakeet-ultra"
NEMO_REPOSITORY = "nvidia/parakeet-tdt-0.6b-v3"
GPU_NOTE = "For the GPU: on the CPU, ONNX Runtime computes most of it in float32 again."

MODEL_DESCRIPTION = ModelDescription(
    title="Parakeet Ultra in ONNX",
    template_path=PurePosixPath("model-card.md"),
    measuring_library_names=("onnx-asr", "onnxruntime", "onnxruntime-gpu"),
    variants=(
        VariantDescription(FLOAT32_VARIANT, "float32"),
        VariantDescription(FLOAT16_VARIANT, "float16", GPU_NOTE),
        VariantDescription(BFLOAT16_VARIANT, "bfloat16", GPU_NOTE),
        # onnx-asr names the 8-bit blocks int8, in the suffix of their files and in its quantization
        # option, so the card shows them under that name.
        VariantDescription(EIGHT_BIT_BLOCKS_VARIANT, "int8"),
    ),
    reference_runs=(ReferenceRunDescription(NEMO_RUN_NAME, "the original model in NeMo"),),
    pinned_revisions=(
        PinnedRevision("fleurs_revision", FLEURS_REPOSITORY),
        PinnedRevision("ultra_revision", ULTRA_REPOSITORY),
        PinnedRevision("v3_revision", NEMO_REPOSITORY),
    ),
)
