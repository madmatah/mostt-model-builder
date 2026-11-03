from mostt_model_builder.domain.acceptance.acceptance import (
    REPORTED_ONLY,
    Comparison,
    Equivalence,
    NonInferiority,
)
from mostt_model_builder.domain.containers.container_steps import Device
from mostt_recipe_parakeet_tdt.domain.variants.variants import FLOAT32_VARIANT, VARIANTS, name_run

NEMO_RUN_NAME = "nemo"
TRANSFORMERS_RUN_NAME = "transformers"
FLOAT32_CPU_RUN_NAME = name_run(FLOAT32_VARIANT, Device.CPU)
FLOAT32_GPU_RUN_NAME = name_run(FLOAT32_VARIANT, Device.GPU)
# A tenth of a point of word error rate is a few words of FLEURS's test set.
EQUIVALENCE_BOUND_POINTS = 0.10
# NeMo and onnx-asr, both in float32 on the CPU, gave identical token ids on every utterance of
# FLEURS fr and en in the trial build: any difference is a fault of the export or of its patch.
SAME_DEVICE_EQUIVALENCE = Equivalence(
    min_identical_ratio=1.0, max_difference_points=EQUIVALENCE_BOUND_POINTS
)
# CUDA's kernels sum in another order than the CPU's, and the trial build found 99.4 % of identical
# token ids in fr and 99.7 % in en between the two devices.
CROSS_DEVICE_EQUIVALENCE = Equivalence(
    min_identical_ratio=0.99, max_difference_points=EQUIVALENCE_BOUND_POINTS
)
# A variant may lose a fifth of a point at most against float32 on its device: the trial build
# measured every one of them within 0.1 point, and the interval of FLEURS's test set is a few tenths
# wide.
VARIANT_NON_INFERIORITY = NonInferiority(max_upper_bound_points=0.20)

# Transformers reads its features with its own extractor, apart from NeMo's, so its transcripts are
# only reported against NeMo's, on fr, the one language it transcribes.
COMPARISONS = (
    Comparison(NEMO_RUN_NAME, FLOAT32_CPU_RUN_NAME, SAME_DEVICE_EQUIVALENCE),
    Comparison(FLOAT32_GPU_RUN_NAME, FLOAT32_CPU_RUN_NAME, CROSS_DEVICE_EQUIVALENCE),
    *(
        Comparison(
            name_run(variant.name, device),
            name_run(FLOAT32_VARIANT, device),
            VARIANT_NON_INFERIORITY,
        )
        for variant in VARIANTS
        if variant.name != FLOAT32_VARIANT
        for device in variant.devices
    ),
    Comparison(TRANSFORMERS_RUN_NAME, NEMO_RUN_NAME, REPORTED_ONLY),
)
