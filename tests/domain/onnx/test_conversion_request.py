import json
import re
from collections.abc import Callable
from functools import partial

import pytest

from mostt_model_builder.domain.onnx.conversion_request import (
    ConversionKind,
    ConversionRequest,
    ConversionRequestError,
    EightBitBlocksConversion,
    FloatConversion,
    judge_conversion,
    parse_conversion_request,
    render_conversion_request,
)

FLOAT16_CONVERSION = FloatConversion(ConversionKind.FLOAT16, "encoder.fp16.onnx")
BFLOAT16_CONVERSION = FloatConversion(
    ConversionKind.BFLOAT16,
    "encoder.bf16.onnx",
    added_float32_operators=("Range",),
    kept_inputs=("length", "audio_lengths"),
)
EIGHT_BIT_BLOCKS_CONVERSION = EightBitBlocksConversion(
    "encoder.int8.onnx",
    block_size=64,
    is_symmetric=True,
    accuracy_level=4,
    quantized_product_count=265,
)
REQUEST = ConversionRequest(
    "export",
    "encoder.onnx",
    (FLOAT16_CONVERSION, BFLOAT16_CONVERSION, EIGHT_BIT_BLOCKS_CONVERSION),
)
FLOAT16_ENTRY: dict[str, object] = {
    "kind": "float16",
    "output": "encoder.fp16.onnx",
    "added_float32_operators": [],
    "keep_io_types": True,
    "kept_inputs": ["length"],
}
EIGHT_BIT_BLOCKS_ENTRY: dict[str, object] = {
    "kind": "eight-bit-blocks",
    "output": "encoder.int8.onnx",
    "block_size": 64,
    "is_symmetric": True,
    "accuracy_level": 4,
    "quantized_product_count": 265,
}
DOCUMENT: dict[str, object] = {
    "input_step": "export",
    "input_name": "encoder.onnx",
    "conversions": [FLOAT16_ENTRY, EIGHT_BIT_BLOCKS_ENTRY],
}


def render_with(**fields: object) -> bytes:
    return json.dumps({**DOCUMENT, **fields}).encode("ascii")


def render_float16_with(**fields: object) -> bytes:
    return render_with(conversions=[{**FLOAT16_ENTRY, **fields}])


def render_eight_bit_blocks_with(**fields: object) -> bytes:
    return render_with(conversions=[{**EIGHT_BIT_BLOCKS_ENTRY, **fields}])


REFUSED_REQUESTS = {
    "kept inputs that are not a list": (
        render_float16_with(kept_inputs="length"),
        "a conversion of the request lists no kept inputs",
    ),
    "a float conversion without its kept inputs": (
        render_with(
            conversions=[
                {name: field for name, field in FLOAT16_ENTRY.items() if name != "kept_inputs"}
            ]
        ),
        "a conversion of the request is not an object of exactly",
    ),
    "a kept input whose name is not a string": (
        render_float16_with(kept_inputs=[7]),
        "a conversion of the request holds 7, not a string",
    ),
    "a document nested 3 000 deep": (b"[" * 3000, "the document nests deeper than 32 levels"),
    "a request beyond its size bound": (
        b" " * (64 * 1024 + 1),
        "the conversion's request holds 65537 bytes, more than 65536",
    ),
    "a request that is not ASCII": (
        json.dumps({**DOCUMENT, "input_name": "encodér.onnx"}, ensure_ascii=False).encode(),
        "the conversion's request is not ASCII JSON",
    ),
    "a key given twice": (
        b'{"input_step": "export", "input_step": "export", "input_name": "encoder.onnx", '
        b'"conversions": []}',
        "a key appears twice in an object",
    ),
    "a request without its conversions": (
        json.dumps({"input_step": "export", "input_name": "encoder.onnx"}).encode("ascii"),
        "the conversion's request is not an object of exactly",
    ),
    "a request with a key of its own": (
        render_with(device="gpu"),
        "the conversion's request is not an object of exactly",
    ),
    "an input of a step outside the grammar": (
        render_with(input_step="Export"),
        "the conversion's input comes from 'Export', not a step under",
    ),
    "an input that is not a file name": (
        render_with(input_name="../encoder.onnx"),
        "the conversion's input, '../encoder.onnx', is not a file name",
    ),
    "conversions that are not a list": (
        render_with(conversions={"float16": FLOAT16_ENTRY}),
        "the conversion's request holds no list of conversions",
    ),
    "no conversion": (render_with(conversions=[]), "the request converts nothing"),
    "more conversions than the bound": (
        render_with(
            conversions=[
                {**FLOAT16_ENTRY, "output": f"encoder.{index}.onnx"} for index in range(17)
            ]
        ),
        "the request holds 17 conversions, more than 16",
    ),
    "a conversion that is not an object": (
        render_with(conversions=["float16"]),
        "a conversion of the request is not an object with a kind",
    ),
    "a conversion whose kind is a list": (
        render_float16_with(kind=["float16"]),
        "a conversion of the request is a ['float16'], an unknown kind",
    ),
    "a conversion of an unknown kind": (
        render_float16_with(kind="float8"),
        "a conversion of the request is a 'float8', an unknown kind",
    ),
    "a float conversion with a key of the 8-bit one": (
        render_float16_with(block_size=64),
        "a conversion of the request is not an object of exactly",
    ),
    "an 8-bit conversion without its accuracy level": (
        render_with(
            conversions=[
                {
                    key: field
                    for key, field in EIGHT_BIT_BLOCKS_ENTRY.items()
                    if key != "accuracy_level"
                }
            ]
        ),
        "a conversion of the request is not an object of exactly",
    ),
    "an output that is not a string": (
        render_float16_with(output=16),
        "a conversion of the request holds 16, not a string",
    ),
    "an output that is not a file name": (
        render_float16_with(output="out/encoder.onnx"),
        "the conversion to 'out/encoder.onnx' writes no file name",
    ),
    "operators that are not a list": (
        render_float16_with(added_float32_operators="Pad"),
        "a conversion of the request lists no operators",
    ),
    "an operator outside the grammar": (
        render_float16_with(added_float32_operators=["Pad Tile"]),
        "keeps 'Pad Tile' in float32, not an operator under",
    ),
    "an operator named twice": (
        render_float16_with(added_float32_operators=["Pad", "Pad"]),
        "keeps an operator in float32 twice",
    ),
    "a keep_io_types that is not a boolean": (
        render_float16_with(keep_io_types=1),
        "a conversion of the request holds 1, not a boolean",
    ),
    "a block size given as a boolean": (
        render_eight_bit_blocks_with(block_size=True),
        "a conversion of the request holds True, not a whole number",
    ),
    "a block size that is not a power of two": (
        render_eight_bit_blocks_with(block_size=96),
        "has blocks of 96 weights, not a power of two from 16 to 256",
    ),
    "an accuracy level under 0": (
        render_eight_bit_blocks_with(accuracy_level=-1),
        "has the accuracy level -1, outside 0 to 4",
    ),
    "a symmetry given as a word": (
        render_eight_bit_blocks_with(is_symmetric="true"),
        "a conversion of the request holds 'true', not a boolean",
    ),
    "a count of products beyond the bound": (
        render_eight_bit_blocks_with(quantized_product_count=100_001),
        "quantizes 100001 products, outside 1 to 100000",
    ),
    "two conversions that write one file": (
        render_with(conversions=[FLOAT16_ENTRY, {**FLOAT16_ENTRY, "kind": "bfloat16"}]),
        "two conversions write encoder.fp16.onnx",
    ),
}

REFUSED_FLOAT_CONVERSIONS = {
    "kept inputs given as one name, which would read as its letters": (
        FloatConversion(ConversionKind.FLOAT16, "encoder.fp16.onnx", kept_inputs="length"),  # type: ignore[arg-type]  # the test feeds a value outside the contract's types, on purpose
        "the conversion to 'encoder.fp16.onnx' keeps the computation from 'length', not a tuple "
        "of input names",
    ),
    "a kept input whose name is not a string": (
        FloatConversion(ConversionKind.FLOAT16, "encoder.fp16.onnx", kept_inputs=(7,)),  # type: ignore[arg-type]  # the test feeds a value outside the contract's types, on purpose
        "the conversion to 'encoder.fp16.onnx' keeps the computation from (7,), not a tuple of "
        "input names",
    ),
}


class TestTheConversionRequest:
    class TestTheParseConversionRequestFunction:
        class TestWhenTheRequestIsTheOneTheHostRendered:
            @pytest.fixture
            def parsed_request(self) -> ConversionRequest:
                return parse_conversion_request(render_conversion_request(REQUEST))

            def test_should_give_back_every_conversion_with_its_parameters_in_order(
                self, parsed_request: ConversionRequest
            ) -> None:
                assert parsed_request == REQUEST

        @pytest.mark.parametrize(
            ("content", "reason"), list(REFUSED_REQUESTS.values()), ids=list(REFUSED_REQUESTS)
        )
        class TestWhenTheRequestBreaksARule:
            @pytest.fixture
            def parse_act(self, content: bytes) -> Callable[[], ConversionRequest]:
                return partial(parse_conversion_request, content)

            def test_should_refuse_it_and_say_why(
                self, parse_act: Callable[[], ConversionRequest], reason: str
            ) -> None:
                with pytest.raises(ConversionRequestError, match=re.escape(reason)):
                    parse_act()

    class TestTheJudgeConversionFunction:
        @pytest.mark.parametrize(
            ("conversion", "reason"),
            list(REFUSED_FLOAT_CONVERSIONS.values()),
            ids=list(REFUSED_FLOAT_CONVERSIONS),
        )
        class TestWhenAFloatConversionBreaksARule:
            @pytest.fixture
            def judge_act(self, conversion: FloatConversion) -> Callable[[], None]:
                return partial(judge_conversion, conversion)

            def test_should_refuse_it_and_say_why(
                self, judge_act: Callable[[], None], reason: str
            ) -> None:
                with pytest.raises(ConversionRequestError, match=re.escape(reason)):
                    judge_act()

    class TestTheEightBitBlocksConversion:
        class TestWhenItsKindIsRead:
            def test_should_be_the_8_bit_blocks_whatever_its_parameters(self) -> None:
                assert EIGHT_BIT_BLOCKS_CONVERSION.kind is ConversionKind.EIGHT_BIT_BLOCKS

    class TestTheFloatConversion:
        class TestWhenItsRecipeSaysNothingOfTheInputsAndOutputs:
            def test_should_keep_them_in_float32_as_the_runtimes_feed_them(self) -> None:
                assert FLOAT16_CONVERSION.keep_io_types is True

        class TestWhenItsRecipeAddsNoOperator:
            def test_should_keep_none_of_its_own_in_float32(self) -> None:
                assert FLOAT16_CONVERSION.added_float32_operators == ()

        class TestWhenItsRecipeDeclaresNoInput:
            def test_should_keep_no_computation_in_float32(self) -> None:
                assert FLOAT16_CONVERSION.kept_inputs == ()
