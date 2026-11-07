from collections.abc import Sequence
from pathlib import Path

import pytest

from mostt_model_builder.domain.acceptance.paired_comparison import (
    INTERVAL_HIGH_INDEX,
    INTERVAL_LOW_INDEX,
    RESAMPLE_COUNT,
)
from mostt_model_builder.domain.onnx.conversion_request import (
    Conversion,
    EightBitBlocksConversion,
)
from mostt_model_builder.domain.onnx.precision_decisions import list_kept_float32_operators
from mostt_model_builder.domain.publication.model_card import join_words
from mostt_recipe_parakeet_tdt.domain.publication.model_description import MODEL_DESCRIPTION
from mostt_recipe_parakeet_tdt.domain.variants.comparisons import VARIANT_NON_INFERIORITY
from mostt_recipe_parakeet_tdt.domain.variants.variants import FEATURE_LENGTHS_INPUT, VARIANTS

RECIPE_DIRECTORY = Path(__file__).resolve().parents[3]
CONVERTED_VARIANTS = tuple(
    (variant.name, variant.conversion) for variant in VARIANTS if variant.conversion is not None
)
INTERVAL_COVERAGE_PERCENT = (INTERVAL_HIGH_INDEX - INTERVAL_LOW_INDEX + 1) * 100 / RESAMPLE_COUNT
KEPT_INPUT_PHRASES = {FEATURE_LENGTHS_INPUT: "the computation of its lengths"}
ROLE_LINES = {
    description.variant: description.role_line for description in MODEL_DESCRIPTION.variants
}


# The role lines and the template state the conversions by hand, so that a reader gets a sentence
# rather than a list of parameters; these phrases, derived from the conversions the recipe runs,
# keep that text true when a parameter changes.
def describe_conversion(conversion: Conversion) -> str:
    if isinstance(conversion, EightBitBlocksConversion):
        symmetry = "symmetric" if conversion.is_symmetric else "asymmetric"
        return f"8-bit blocks of {conversion.block_size} {symmetry} weights"
    kept_parts = (
        *list_kept_float32_operators(conversion.kind, conversion.added_float32_operators),
        *(KEPT_INPUT_PHRASES[kept_input] for kept_input in conversion.kept_inputs),
    )
    if not kept_parts:
        return conversion.kind.value
    return f"{conversion.kind.value}, with {join_words(kept_parts)} kept in float32"


def state_the_conversions(phrases: Sequence[str]) -> str:
    return f"The encoder is converted to {'; to '.join(phrases[:-1])}; and to {phrases[-1]}."


class TestTheModelDescription:
    class TestTheModelDescriptionConstant:
        def test_should_state_its_conversion_in_the_role_line_of_each_converted_variant(
            self,
        ) -> None:
            mismatches = [
                variant
                for variant, conversion in CONVERTED_VARIANTS
                if describe_conversion(conversion) not in ROLE_LINES[variant]
            ]
            assert (mismatches, len(CONVERTED_VARIANTS)) == ([], 3)

    class TestWhenItsTemplateIsRead:
        @pytest.fixture
        def template(self) -> str:
            return (RECIPE_DIRECTORY / MODEL_DESCRIPTION.template_path).read_text(encoding="utf-8")

        def test_should_state_the_conversions_the_recipe_runs_among_its_modifications(
            self, template: str
        ) -> None:
            phrases = [describe_conversion(conversion) for _, conversion in CONVERTED_VARIANTS]
            assert f"- {state_the_conversions(phrases)} " in template

        def test_should_state_the_bound_of_the_non_inferiority_that_accepts_a_variant(
            self, template: str
        ) -> None:
            bound_points = VARIANT_NON_INFERIORITY.max_upper_bound_points
            assert f"loses at most {bound_points:g} points against its reference" in template

        def test_should_state_the_coverage_of_the_interval_of_the_paired_bootstrap(
            self, template: str
        ) -> None:
            assert f"a {INTERVAL_COVERAGE_PERCENT:g} % interval from a paired bootstrap" in template
