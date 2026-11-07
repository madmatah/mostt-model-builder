import re
from collections.abc import Callable
from functools import partial

import pytest

from mostt_recipe_parakeet_tdt.domain.variants.candidate_layout import CandidateLayoutError
from mostt_recipe_parakeet_tdt.domain.variants.quantizations import (
    find_quantization,
    name_decoder_joint_file,
    name_encoder_file,
)

LAYOUT_BY_VARIANT = {
    "float32": (None, "encoder-model.onnx", "decoder_joint-model.onnx"),
    "float16": ("fp16", "encoder-model.fp16.onnx", "decoder_joint-model.fp16.onnx"),
    "bfloat16": ("bf16", "encoder-model.bf16.onnx", "decoder_joint-model.bf16.onnx"),
    "eight-bit-blocks": ("int8", "encoder-model.int8.onnx", "decoder_joint-model.int8.onnx"),
}
UNKNOWN_VARIANT = "four-bit-blocks"


class TestTheQuantizations:
    class TestTheFindQuantizationFunction:
        @pytest.mark.parametrize(
            ("variant", "expected_quantization"),
            [(variant, layout[0]) for variant, layout in LAYOUT_BY_VARIANT.items()],
            ids=list(LAYOUT_BY_VARIANT),
        )
        class TestWhenTheVariantIsKnown:
            @pytest.fixture
            def quantization(self, variant: str) -> str | None:
                return find_quantization(variant)

            def test_should_give_onnx_asr_suffix_of_the_variant(
                self, quantization: str | None, expected_quantization: str | None
            ) -> None:
                assert quantization == expected_quantization

        class TestWhenTheVariantIsUnknown:
            @pytest.fixture
            def find_act(self) -> Callable[[], str | None]:
                return partial(find_quantization, UNKNOWN_VARIANT)

            def test_should_refuse_it_and_name_the_known_variants(
                self, find_act: Callable[[], str | None]
            ) -> None:
                with pytest.raises(
                    CandidateLayoutError,
                    match=re.escape(
                        "the variant 'four-bit-blocks' is none of float32, float16, bfloat16, "
                        "eight-bit-blocks"
                    ),
                ):
                    find_act()

    class TestTheNameEncoderFileFunction:
        @pytest.mark.parametrize(
            ("variant", "expected_file_name"),
            [(variant, layout[1]) for variant, layout in LAYOUT_BY_VARIANT.items()],
            ids=list(LAYOUT_BY_VARIANT),
        )
        class TestWhenTheVariantIsKnown:
            @pytest.fixture
            def file_name(self, variant: str) -> str:
                return name_encoder_file(variant)

            def test_should_name_the_file_onnx_asr_loads_for_the_variant(
                self, file_name: str, expected_file_name: str
            ) -> None:
                assert file_name == expected_file_name

        class TestWhenTheVariantIsUnknown:
            @pytest.fixture
            def name_act(self) -> Callable[[], str]:
                return partial(name_encoder_file, UNKNOWN_VARIANT)

            def test_should_refuse_it(self, name_act: Callable[[], str]) -> None:
                with pytest.raises(
                    CandidateLayoutError, match=re.escape("the variant 'four-bit-blocks' is none")
                ):
                    name_act()

    class TestTheNameDecoderJointFileFunction:
        @pytest.mark.parametrize(
            ("variant", "expected_file_name"),
            [(variant, layout[2]) for variant, layout in LAYOUT_BY_VARIANT.items()],
            ids=list(LAYOUT_BY_VARIANT),
        )
        class TestWhenTheVariantIsKnown:
            @pytest.fixture
            def file_name(self, variant: str) -> str:
                return name_decoder_joint_file(variant)

            def test_should_name_the_file_onnx_asr_loads_for_the_variant(
                self, file_name: str, expected_file_name: str
            ) -> None:
                assert file_name == expected_file_name

        class TestWhenTheVariantIsUnknown:
            @pytest.fixture
            def name_act(self) -> Callable[[], str]:
                return partial(name_decoder_joint_file, UNKNOWN_VARIANT)

            def test_should_refuse_it(self, name_act: Callable[[], str]) -> None:
                with pytest.raises(
                    CandidateLayoutError, match=re.escape("the variant 'four-bit-blocks' is none")
                ):
                    name_act()
