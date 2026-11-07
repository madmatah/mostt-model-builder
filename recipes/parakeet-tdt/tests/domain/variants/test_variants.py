from fnmatch import fnmatchcase

import pytest

from mostt_model_builder.domain.containers.devices import Device
from mostt_model_builder.domain.formats.file_formats import FileFormat
from mostt_model_builder.domain.onnx.conversion_request import (
    EightBitBlocksConversion,
    FloatConversion,
)
from mostt_recipe_parakeet_tdt.domain.variants.quantizations import find_quantization
from mostt_recipe_parakeet_tdt.domain.variants.variants import VARIANTS, Variant, name_run
from mostt_recipe_parakeet_tdt.recipe import RECIPE

CANDIDATE_NAMES = [candidate_file.name for candidate_file in RECIPE.candidate]
LENGTHS_KEPT_IN_FLOAT32 = ("length",)
DEVICES_BY_VARIANT = {
    "float32": (Device.CPU, Device.GPU),
    "float16": (Device.GPU,),
    "bfloat16": (Device.GPU,),
    "eight-bit-blocks": (Device.GPU, Device.CPU),
}


# onnx-asr 0.12 looks for each model file by a glob: the name alone without a quantization, or the
# stem, any one character and the quantization's suffix, and loads a quantization only when its
# glob matches one file exactly.
def list_onnx_asr_patterns(variant: str) -> tuple[str, ...]:
    quantization = find_quantization(variant)
    if quantization is None:
        return ("encoder-model.onnx", "decoder_joint-model.onnx")
    return (f"encoder-model?{quantization}.onnx", f"decoder_joint-model?{quantization}.onnx")


def list_served_onnx_files(variant: str) -> list[str]:
    return [
        candidate_file.name
        for candidate_file in RECIPE.candidate
        if candidate_file.file_format == FileFormat.ONNX and variant in candidate_file.variants
    ]


class TestTheVariants:
    class TestTheVariantsConstant:
        @pytest.mark.parametrize("variant", VARIANTS, ids=[variant.name for variant in VARIANTS])
        class TestWhenOnnxAsrLooksForTheFilesOfAVariant:
            def test_should_find_its_own_files_alone_one_per_glob(self, variant: Variant) -> None:
                assert [
                    [name for name in CANDIDATE_NAMES if fnmatchcase(name, pattern)]
                    for pattern in list_onnx_asr_patterns(variant.name)
                ] == [[name] for name in list_served_onnx_files(variant.name)]

        def test_should_give_external_data_to_float32_alone(self) -> None:
            assert [
                candidate_file.variants
                for candidate_file in RECIPE.candidate
                if candidate_file.file_format == FileFormat.ONNX_EXTERNAL_DATA
            ] == [("float32",)]

        def test_should_run_float16_and_bfloat16_on_the_gpu_and_the_others_on_both(self) -> None:
            assert {variant.name: variant.devices for variant in VARIANTS} == DEVICES_BY_VARIANT

        def test_should_compute_the_lengths_in_float32_in_float16_and_bfloat16(self) -> None:
            assert {
                variant.name: variant.conversion.kept_inputs
                for variant in VARIANTS
                if isinstance(variant.conversion, FloatConversion)
            } == {"float16": LENGTHS_KEPT_IN_FLOAT32, "bfloat16": LENGTHS_KEPT_IN_FLOAT32}

        def test_should_convert_to_8_bit_blocks_once(self) -> None:
            assert [
                variant.name
                for variant in VARIANTS
                if isinstance(variant.conversion, EightBitBlocksConversion)
            ] == ["eight-bit-blocks"]

        def test_should_quantize_every_product_of_a_constant_weight_in_blocks_of_64(self) -> None:
            assert [
                (
                    variant.conversion.block_size,
                    variant.conversion.is_symmetric,
                    variant.conversion.accuracy_level,
                    variant.conversion.quantized_product_count,
                )
                for variant in VARIANTS
                if isinstance(variant.conversion, EightBitBlocksConversion)
            ] == [(64, True, 4, 265)]

        def test_should_convert_every_variant_but_float32_which_is_the_conversions_input(
            self,
        ) -> None:
            assert [variant.name for variant in VARIANTS if variant.conversion is None] == [
                "float32"
            ]

    class TestTheNameRunFunction:
        class TestWhenAVariantRunsOnADevice:
            @pytest.fixture
            def run_name(self) -> str:
                return name_run("eight-bit-blocks", Device.CPU)

            def test_should_name_the_run_after_the_variant_then_the_device(
                self, run_name: str
            ) -> None:
                assert run_name == "eight-bit-blocks-cpu"
