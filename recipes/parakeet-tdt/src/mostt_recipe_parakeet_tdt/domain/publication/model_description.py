from pathlib import PurePosixPath

from mostt_model_builder.domain.bench.fleurs_index import FLEURS_REPOSITORY
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
GPU_NOTE = "on the CPU, ONNX Runtime computes most of it in float32 again"

MODEL_DESCRIPTION = ModelDescription(
    title="Parakeet Ultra in ONNX",
    template_path=PurePosixPath("model-card.md"),
    measuring_library_names=("onnx-asr", "onnxruntime", "onnxruntime-gpu"),
    variants=(
        VariantDescription(
            FLOAT32_VARIANT,
            "float32",
            "The reference: the exported model, unconverted, against which the other variants "
            "are measured.",
            abbreviation="FP32",
        ),
        VariantDescription(
            FLOAT16_VARIANT,
            "float16",
            "The encoder converted to float16, for the GPU.",
            note=GPU_NOTE,
            abbreviation="FP16",
        ),
        VariantDescription(
            BFLOAT16_VARIANT,
            "bfloat16",
            "The encoder converted to bfloat16, with Pad, Tile and Floor kept in float32, for the "
            "GPU.",
            note=GPU_NOTE,
            abbreviation="BF16",
        ),
        # onnx-asr names the 8-bit blocks int8, in the suffix of their files and in its quantization
        # option, so the card shows them under that name.
        VariantDescription(
            EIGHT_BIT_BLOCKS_VARIANT,
            "int8",
            "Recommended on the CPU: the smallest variant. The encoder is in 8-bit blocks of 64 "
            "symmetric weights.",
        ),
    ),
    reference_runs=(ReferenceRunDescription(NEMO_RUN_NAME, "the original model in NeMo"),),
    pinned_revisions=(
        PinnedRevision("fleurs_revision", FLEURS_REPOSITORY),
        PinnedRevision("ultra_revision", ULTRA_REPOSITORY),
        PinnedRevision("v3_revision", NEMO_REPOSITORY),
    ),
)
