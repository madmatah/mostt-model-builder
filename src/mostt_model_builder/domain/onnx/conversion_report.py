import json
from collections.abc import Mapping
from dataclasses import dataclass

from mostt_model_builder.domain.json_documents.json_depth import MAX_JSON_DEPTH, refuse_deep_json
from mostt_model_builder.domain.onnx.conversion_request import (
    MAX_CONVERSION_COUNT,
    MAX_OPERATOR_COUNT,
    OPERATOR_GRAMMAR,
    Conversion,
    ConversionKind,
    ConversionRequest,
    EightBitBlocksConversion,
)
from mostt_model_builder.domain.onnx.precision_decisions import list_kept_float32_operators

CONVERSION_REPORT_FILE_NAME = "conversion-report.json"
CONVERSION_REPORT_MAX_SIZE_BYTES = 64 * 1024
# An encoder holds a few thousand nodes and a few billion weights; the bound only refuses a count
# that no graph of the build could hold.
MAX_COUNT = 2**40
REPORT_KEYS = frozenset({"conversions"})
CONVERTED_FILE_KEYS = frozenset(
    {
        "kind",
        "output",
        "node_count",
        "quantized_product_count",
        "kept_float32_operators",
        "clamped_initializer_count",
        "below_min_positive_count",
        "above_max_finite_count",
        "realigned_cast_count",
    }
)


class ConversionReportError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class ConvertedFile:
    kind: ConversionKind
    output: str
    node_count: int
    quantized_product_count: int
    kept_float32_operators: tuple[str, ...]
    clamped_initializer_count: int
    below_min_positive_count: int
    above_max_finite_count: int
    realigned_cast_count: int


@dataclass(frozen=True, slots=True)
class ConversionReport:
    files: tuple[ConvertedFile, ...]


def render_conversion_report(report: ConversionReport) -> bytes:
    document = {
        "conversions": [
            {
                "kind": converted.kind.value,
                "output": converted.output,
                "node_count": converted.node_count,
                "quantized_product_count": converted.quantized_product_count,
                "kept_float32_operators": list(converted.kept_float32_operators),
                "clamped_initializer_count": converted.clamped_initializer_count,
                "below_min_positive_count": converted.below_min_positive_count,
                "above_max_finite_count": converted.above_max_finite_count,
                "realigned_cast_count": converted.realigned_cast_count,
            }
            for converted in report.files
        ]
    }
    return json.dumps(document, indent=1).encode("ascii")


def parse_conversion_report(content: bytes, request: ConversionRequest) -> ConversionReport:
    if len(content) > CONVERSION_REPORT_MAX_SIZE_BYTES:
        raise ConversionReportError(
            f"the conversion's report holds {len(content)} bytes, more than "
            f"{CONVERSION_REPORT_MAX_SIZE_BYTES}"
        )
    try:
        document = json.loads(
            refuse_deep_json(content.decode("ascii"), MAX_JSON_DEPTH),
            object_pairs_hook=refuse_duplicate_keys,
            parse_constant=refuse_constant,
        )
    except (UnicodeDecodeError, ValueError) as error:
        raise ConversionReportError(
            f"the conversion's report is not strict JSON: {error}"
        ) from error
    fields = require_object(document, REPORT_KEYS, "the conversion's report")
    entries = fields["conversions"]
    if not isinstance(entries, list) or len(entries) > MAX_CONVERSION_COUNT:
        raise ConversionReportError(
            f"the conversion's report holds no list of at most {MAX_CONVERSION_COUNT} conversions"
        )
    if len(entries) != len(request.conversions):
        raise ConversionReportError(
            f"the conversion's report describes {len(entries)} conversions, and "
            f"{len(request.conversions)} were asked"
        )
    return ConversionReport(
        tuple(
            parse_converted_file(entry, conversion)
            for entry, conversion in zip(entries, request.conversions, strict=True)
        )
    )


def parse_converted_file(entry: object, conversion: Conversion) -> ConvertedFile:
    where = f"the conversion to {conversion.output}"
    fields = require_object(entry, CONVERTED_FILE_KEYS, f"the report of {where}")
    kind = fields["kind"]
    output = fields["output"]
    if kind != conversion.kind.value or output != conversion.output:
        raise ConversionReportError(
            f"the report of {where} describes {output!a} as {kind!a}, not "
            f"{conversion.output} as {conversion.kind.value}"
        )
    converted = ConvertedFile(
        kind=conversion.kind,
        output=conversion.output,
        node_count=require_count(fields["node_count"], f"{where}: its count of nodes"),
        quantized_product_count=require_count(
            fields["quantized_product_count"], f"{where}: its count of quantized products"
        ),
        kept_float32_operators=require_operators(fields["kept_float32_operators"], where),
        clamped_initializer_count=require_count(
            fields["clamped_initializer_count"], f"{where}: its count of clamped initializers"
        ),
        below_min_positive_count=require_count(
            fields["below_min_positive_count"], f"{where}: its count of values below the minimum"
        ),
        above_max_finite_count=require_count(
            fields["above_max_finite_count"], f"{where}: its count of values above the maximum"
        ),
        realigned_cast_count=require_count(
            fields["realigned_cast_count"], f"{where}: its count of realigned Cast nodes"
        ),
    )
    judge_converted_file(converted, conversion, where)
    return converted


def judge_converted_file(converted: ConvertedFile, conversion: Conversion, where: str) -> None:
    if isinstance(conversion, EightBitBlocksConversion):
        expected_operators: tuple[str, ...] = ()
        expected_product_count = conversion.quantized_product_count
    else:
        expected_operators = list_kept_float32_operators(
            conversion.kind, conversion.added_float32_operators
        )
        expected_product_count = 0
    if converted.kept_float32_operators != expected_operators:
        raise ConversionReportError(
            f"{where} kept {', '.join(converted.kept_float32_operators) or 'no operator'} in "
            f"float32, not {', '.join(expected_operators) or 'no operator'}"
        )
    if converted.quantized_product_count != expected_product_count:
        raise ConversionReportError(
            f"{where} quantized {converted.quantized_product_count} products to 8-bit blocks, "
            f"not {expected_product_count}"
        )


def require_operators(field: object, where: str) -> tuple[str, ...]:
    if not isinstance(field, list) or len(field) > MAX_OPERATOR_COUNT:
        raise ConversionReportError(
            f"{where}: its operators kept in float32 are no list of at most {MAX_OPERATOR_COUNT}"
        )
    for operator in field:
        if not isinstance(operator, str) or not OPERATOR_GRAMMAR.fullmatch(operator):
            raise ConversionReportError(
                f"{where}: an operator kept in float32, {operator!a}, is not under "
                f"{OPERATOR_GRAMMAR.pattern}"
            )
    return tuple(str(operator) for operator in field)


def require_count(field: object, what: str) -> int:
    if isinstance(field, bool) or not isinstance(field, int) or not 0 <= field <= MAX_COUNT:
        raise ConversionReportError(f"{what}, {field!a}, is not a count up to {MAX_COUNT}")
    return field


def require_object(document: object, keys: frozenset[str], what: str) -> Mapping[str, object]:
    if not isinstance(document, dict) or set(document) != keys:
        raise ConversionReportError(f"{what} is not an object of exactly {sorted(keys)}")
    return document


def refuse_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    keys = [key for key, _ in pairs]
    if len(set(keys)) != len(keys):
        raise ValueError("a key appears twice in an object")
    return dict(pairs)


def refuse_constant(constant: str) -> object:
    raise ValueError(f"{constant} is not a JSON number")
