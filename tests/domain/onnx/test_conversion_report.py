import json
import re
from collections.abc import Callable
from functools import partial

import pytest

from mostt_model_builder.domain.onnx.conversion_report import (
    ConversionReport,
    ConversionReportError,
    ConvertedFile,
    parse_conversion_report,
    render_conversion_report,
)
from mostt_model_builder.domain.onnx.conversion_request import (
    ConversionKind,
    ConversionRequest,
    EightBitBlocksConversion,
    FloatConversion,
)

FLOAT16_CONVERSION = FloatConversion(ConversionKind.FLOAT16, "encoder.fp16.onnx")
BFLOAT16_CONVERSION = FloatConversion(
    ConversionKind.BFLOAT16, "encoder.bf16.onnx", added_float32_operators=("Range",)
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
FLOAT16_FILE = ConvertedFile(
    kind=ConversionKind.FLOAT16,
    output="encoder.fp16.onnx",
    node_count=1648,
    quantized_product_count=0,
    kept_float32_operators=(),
    clamped_initializer_count=75,
    below_min_positive_count=95,
    above_max_finite_count=0,
    realigned_cast_count=5,
)
BFLOAT16_FILE = ConvertedFile(
    kind=ConversionKind.BFLOAT16,
    output="encoder.bf16.onnx",
    node_count=1660,
    quantized_product_count=0,
    kept_float32_operators=("Pad", "Tile", "Floor", "Range"),
    clamped_initializer_count=75,
    below_min_positive_count=95,
    above_max_finite_count=0,
    realigned_cast_count=5,
)
EIGHT_BIT_BLOCKS_FILE = ConvertedFile(
    kind=ConversionKind.EIGHT_BIT_BLOCKS,
    output="encoder.int8.onnx",
    node_count=1643,
    quantized_product_count=265,
    kept_float32_operators=(),
    clamped_initializer_count=0,
    below_min_positive_count=0,
    above_max_finite_count=0,
    realigned_cast_count=0,
)
REPORT = ConversionReport((FLOAT16_FILE, BFLOAT16_FILE, EIGHT_BIT_BLOCKS_FILE))
DOCUMENT = json.loads(render_conversion_report(REPORT))


def render_with_entry(entry_index: int, **fields: object) -> bytes:
    entries = [dict(entry) for entry in DOCUMENT["conversions"]]
    entries[entry_index].update(fields)
    return json.dumps({"conversions": entries}).encode("ascii")


REFUSED_REPORTS = {
    "a document nested 3 000 deep": (b"[" * 3000, "the document nests deeper than 32 levels"),
    "a report beyond its size bound": (
        b" " * (64 * 1024 + 1),
        "the conversion's report holds 65537 bytes, more than 65536",
    ),
    "a report that is not JSON": (b"{conversions", "the conversion's report is not strict JSON"),
    "a count written as a constant that JSON lacks": (
        render_with_entry(0).replace(b'"node_count": 1648', b'"node_count": NaN'),
        "the conversion's report is not strict JSON: NaN is not a JSON number",
    ),
    "a key given twice": (
        b'{"conversions": [], "conversions": []}',
        "the conversion's report is not strict JSON: a key appears twice in an object",
    ),
    "a report with a key of its own": (
        json.dumps({**DOCUMENT, "elapsed_s": 12}).encode("ascii"),
        "the conversion's report is not an object of exactly ['conversions']",
    ),
    "conversions that are not a list": (
        json.dumps({"conversions": {}}).encode("ascii"),
        "the conversion's report holds no list of at most 16 conversions",
    ),
    "a conversion fewer than asked": (
        json.dumps({"conversions": DOCUMENT["conversions"][:2]}).encode("ascii"),
        "the conversion's report describes 2 conversions, and 3 were asked",
    ),
    "a conversion of a key of its own": (
        render_with_entry(0, path="/output/encoder.fp16.onnx"),
        "the report of the conversion to encoder.fp16.onnx is not an object of exactly",
    ),
    "another output than asked": (
        render_with_entry(0, output="../encoder.onnx"),
        "the report of the conversion to encoder.fp16.onnx describes '../encoder.onnx' as "
        "'float16', not encoder.fp16.onnx as float16",
    ),
    "another kind than asked": (
        render_with_entry(1, kind="float16"),
        "describes 'encoder.bf16.onnx' as 'float16', not encoder.bf16.onnx as bfloat16",
    ),
    "a count that is negative": (
        render_with_entry(0, node_count=-1),
        "the conversion to encoder.fp16.onnx: its count of nodes, -1, is not a count",
    ),
    "a count given as a boolean": (
        render_with_entry(0, realigned_cast_count=True),
        "its count of realigned Cast nodes, True, is not a count",
    ),
    "a count given as a decimal": (
        render_with_entry(0, below_min_positive_count=95.0),
        "its count of values below the minimum, 95.0, is not a count",
    ),
    "a count beyond the bound": (
        render_with_entry(0, above_max_finite_count=2**40 + 1),
        "its count of values above the maximum, 1099511627777, is not a count up to",
    ),
    "operators that are not a list": (
        render_with_entry(1, kept_float32_operators="Pad"),
        "its operators kept in float32 are no list of at most 64",
    ),
    "an operator outside the grammar": (
        render_with_entry(1, kept_float32_operators=["Pad", "Tile\x1b"]),
        "an operator kept in float32, 'Tile\\x1b', is not under",
    ),
    "bfloat16 without an operator of the core kept in float32": (
        render_with_entry(1, kept_float32_operators=["Pad", "Tile", "Range"]),
        "the conversion to encoder.bf16.onnx kept Pad, Tile, Range in float32, not Pad, Tile, "
        "Floor, Range",
    ),
    "float16 with an operator kept that no one asked for": (
        render_with_entry(0, kept_float32_operators=["Pad"]),
        "the conversion to encoder.fp16.onnx kept Pad in float32, not no operator",
    ),
    "an 8-bit conversion that quantized fewer products than declared": (
        render_with_entry(2, quantized_product_count=217),
        "the conversion to encoder.int8.onnx quantized 217 products to 8-bit blocks, not 265",
    ),
    "a float conversion that quantized products": (
        render_with_entry(0, quantized_product_count=3),
        "the conversion to encoder.fp16.onnx quantized 3 products to 8-bit blocks, not 0",
    ),
}


class TestTheConversionReport:
    class TestTheParseConversionReportFunction:
        class TestWhenTheReportDescribesEachConversionAsked:
            @pytest.fixture
            def parsed_report(self) -> ConversionReport:
                return parse_conversion_report(render_conversion_report(REPORT), REQUEST)

            def test_should_give_back_each_converted_file_with_its_counts(
                self, parsed_report: ConversionReport
            ) -> None:
                assert parsed_report == REPORT

        @pytest.mark.parametrize(
            ("content", "reason"), list(REFUSED_REPORTS.values()), ids=list(REFUSED_REPORTS)
        )
        class TestWhenTheReportBreaksARule:
            @pytest.fixture
            def parse_act(self, content: bytes) -> Callable[[], ConversionReport]:
                return partial(parse_conversion_report, content, REQUEST)

            def test_should_refuse_it_and_say_why(
                self, parse_act: Callable[[], ConversionReport], reason: str
            ) -> None:
                with pytest.raises(ConversionReportError, match=re.escape(reason)):
                    parse_act()
