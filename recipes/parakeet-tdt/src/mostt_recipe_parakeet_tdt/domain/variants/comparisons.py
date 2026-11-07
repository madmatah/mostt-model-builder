from mostt_model_builder.domain.acceptance.acceptance import (
    REPORTED_ONLY,
    Comparison,
    Equivalence,
    NonInferiority,
)
from mostt_model_builder.domain.containers.devices import Device
from mostt_recipe_parakeet_tdt.domain.variants.variants import FLOAT32_VARIANT, VARIANTS, name_run

NEMO_RUN_NAME = "nemo"
TRANSFORMERS_RUN_NAME = "transformers"
FLOAT32_CPU_RUN_NAME = name_run(FLOAT32_VARIANT, Device.CPU)
FLOAT32_GPU_RUN_NAME = name_run(FLOAT32_VARIANT, Device.GPU)
# A tenth of a point of word error rate is a few words of FLEURS's test set.
EQUIVALENCE_BOUND_POINTS = 0.10
# NeMo and onnx-asr, both in float32 on the CPU, gave identical token ids on every utterance of
# en_us, es_419, fr_fr and pt_br in the five-language build, and on 861 of 862 in de_de, where they
# parted on a near tie, "élevés" or "élevé", 0.01 point apart: the bound admits such a tie on one
# utterance in two hundred, while the rates must still stay within a tenth of a point.
SAME_DEVICE_EQUIVALENCE = Equivalence(
    min_identical_ratio=0.995, max_difference_points=EQUIVALENCE_BOUND_POINTS
)
# CUDA's kernels sum in another order than the CPU's, and the five-language build found 99.4 % to
# 99.9 % of identical token ids between the two devices, the lowest in fr_fr, 672 of 676.
CROSS_DEVICE_EQUIVALENCE = Equivalence(
    min_identical_ratio=0.99, max_difference_points=EQUIVALENCE_BOUND_POINTS
)
# A variant may lose a fifth of a point at most against float32 on its device: the five-language
# build measured the upper bound of every one of them at +0.14 point at most, eight-bit-blocks-cpu
# in en_us, and the interval of FLEURS's test set is a few tenths wide.
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
