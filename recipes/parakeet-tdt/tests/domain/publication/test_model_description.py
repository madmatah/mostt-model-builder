from collections.abc import Sequence
from pathlib import Path

import pytest

from mostt_model_builder.domain.onnx.conversion_request import (
    Conversion,
    EightBitBlocksConversion,
)
from mostt_model_builder.domain.onnx.precision_decisions import list_kept_float32_operators
from mostt_model_builder.domain.publication.model_card import join_words
from mostt_recipe_parakeet_tdt.domain.publication.model_description import MODEL_DESCRIPTION
from mostt_recipe_parakeet_tdt.domain.variants.variants import VARIANTS

RECIPE_DIRECTORY = Path(__file__).resolve().parents[3]
CONVERTED_VARIANTS = tuple(
    (variant.name, variant.conversion) for variant in VARIANTS if variant.conversion is not None
)
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
    kept_operators = list_kept_float32_operators(
        conversion.kind, conversion.added_float32_operators
    )
    if not kept_operators:
        return conversion.kind.value
    return f"{conversion.kind.value}, with {join_words(kept_operators)} kept in float32"


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
