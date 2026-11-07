from mostt_model_builder.domain.acceptance.acceptance import (
    REPORTED_ONLY,
    ComparisonRule,
    Equivalence,
    NonInferiority,
)
from mostt_recipe_parakeet_tdt.domain.variants.comparisons import COMPARISONS

SAME_DEVICE = Equivalence(min_identical_ratio=0.995, max_difference_points=0.10)
CROSS_DEVICE = Equivalence(min_identical_ratio=0.99, max_difference_points=0.10)
VARIANT = NonInferiority(max_upper_bound_points=0.20)
COMPARED_RUNS: list[tuple[str, str, ComparisonRule]] = [
    ("nemo", "float32-cpu", SAME_DEVICE),
    ("float32-gpu", "float32-cpu", CROSS_DEVICE),
    ("float16-gpu", "float32-gpu", VARIANT),
    ("bfloat16-gpu", "float32-gpu", VARIANT),
    ("eight-bit-blocks-gpu", "float32-gpu", VARIANT),
    ("eight-bit-blocks-cpu", "float32-cpu", VARIANT),
    ("transformers", "nemo", REPORTED_ONLY),
]


class TestTheComparisons:
    class TestTheComparisonsConstant:
        def test_should_hold_the_export_to_nemo_each_device_and_each_variant_to_float32(
            self,
        ) -> None:
            assert [
                (comparison.run, comparison.reference_run, comparison.rule)
                for comparison in COMPARISONS
            ] == COMPARED_RUNS
