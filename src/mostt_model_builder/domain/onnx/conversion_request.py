import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum

from mostt_model_builder.domain.formats.file_names import is_file_name
from mostt_model_builder.domain.json_documents.json_depth import MAX_JSON_DEPTH, refuse_deep_json

CONVERSION_REQUEST_FILE_NAME = "conversion-request.json"
CONVERSION_REQUEST_MAX_SIZE_BYTES = 64 * 1024
EIGHT_BIT_BLOCKS_BITS = 8
# MatMulNBits takes a block of a power of two, of at least 16 weights; ONNX Runtime's kernels go up
# to 256.
MIN_BLOCK_SIZE = 16
MAX_BLOCK_SIZE = 256
# ONNX Runtime's accuracy levels of MatMulNBits: 0 leaves the choice to the kernel, 1 to 4 compute
# in float32, float16, bfloat16 and int8.
MIN_ACCURACY_LEVEL = 0
MAX_ACCURACY_LEVEL = 4
# An encoder of the families the build targets holds a few hundred matrix products; the bound only
# refuses a count that no graph could hold.
MAX_QUANTIZED_PRODUCT_COUNT = 100_000
MAX_CONVERSION_COUNT = 16
MAX_OPERATOR_COUNT = 64
OPERATOR_GRAMMAR = re.compile(r"[A-Za-z][A-Za-z0-9_]{0,127}")
STEP_GRAMMAR = re.compile(r"[a-z0-9][a-z0-9-]{0,31}")
REQUEST_KEYS = frozenset({"input_step", "input_name", "conversions"})
FLOAT_CONVERSION_KEYS = frozenset({"kind", "output", "added_float32_operators", "keep_io_types"})
EIGHT_BIT_BLOCKS_KEYS = frozenset(
    {"kind", "output", "block_size", "is_symmetric", "accuracy_level", "quantized_product_count"}
)


class ConversionRequestError(ValueError):
    pass


class ConversionKind(StrEnum):
    FLOAT16 = "float16"
    BFLOAT16 = "bfloat16"
    EIGHT_BIT_BLOCKS = "eight-bit-blocks"


FLOAT_KINDS = frozenset({ConversionKind.FLOAT16, ConversionKind.BFLOAT16})


@dataclass(frozen=True, slots=True)
class FloatConversion:
    kind: ConversionKind
    output: str
    added_float32_operators: tuple[str, ...] = ()
    keep_io_types: bool = True


@dataclass(frozen=True, slots=True)
class EightBitBlocksConversion:
    output: str
    block_size: int
    is_symmetric: bool
    accuracy_level: int
    quantized_product_count: int

    @property
    def kind(self) -> ConversionKind:
        return ConversionKind.EIGHT_BIT_BLOCKS


type Conversion = FloatConversion | EightBitBlocksConversion


@dataclass(frozen=True, slots=True)
class ConversionRequest:
    input_step: str
    input_name: str
    conversions: tuple[Conversion, ...]


def judge_conversion(conversion: Conversion) -> None:
    where = f"the conversion to {conversion.output!a}"
    if not isinstance(conversion.output, str) or not is_file_name(conversion.output):
        raise ConversionRequestError(f"{where} writes no file name")
    if isinstance(conversion, FloatConversion):
        judge_float_conversion(conversion, where)
    else:
        judge_eight_bit_blocks_conversion(conversion, where)


def judge_float_conversion(conversion: FloatConversion, where: str) -> None:
    if conversion.kind not in FLOAT_KINDS:
        kinds = ", ".join(sorted(kind.value for kind in FLOAT_KINDS))
        raise ConversionRequestError(f"{where} is a {conversion.kind!a}, not one of {kinds}")
    operators = conversion.added_float32_operators
    if len(operators) > MAX_OPERATOR_COUNT:
        raise ConversionRequestError(
            f"{where} keeps {len(operators)} operators in float32, more than {MAX_OPERATOR_COUNT}"
        )
    for operator in operators:
        if not isinstance(operator, str) or not OPERATOR_GRAMMAR.fullmatch(operator):
            raise ConversionRequestError(
                f"{where} keeps {operator!a} in float32, not an operator under "
                f"{OPERATOR_GRAMMAR.pattern}"
            )
    if len(set(operators)) != len(operators):
        raise ConversionRequestError(f"{where} keeps an operator in float32 twice")
    if not isinstance(conversion.keep_io_types, bool):
        raise ConversionRequestError(
            f"{where} has keep_io_types {conversion.keep_io_types!a}, not a boolean"
        )


def judge_eight_bit_blocks_conversion(conversion: EightBitBlocksConversion, where: str) -> None:
    block_size = conversion.block_size
    if (
        not is_whole_number(block_size)
        or not MIN_BLOCK_SIZE <= block_size <= MAX_BLOCK_SIZE
        or block_size & (block_size - 1)
    ):
        raise ConversionRequestError(
            f"{where} has blocks of {block_size!a} weights, not a power of two from "
            f"{MIN_BLOCK_SIZE} to {MAX_BLOCK_SIZE}"
        )
    if not isinstance(conversion.is_symmetric, bool):
        raise ConversionRequestError(
            f"{where} has is_symmetric {conversion.is_symmetric!a}, not a boolean"
        )
    if (
        not is_whole_number(conversion.accuracy_level)
        or not MIN_ACCURACY_LEVEL <= conversion.accuracy_level <= MAX_ACCURACY_LEVEL
    ):
        raise ConversionRequestError(
            f"{where} has the accuracy level {conversion.accuracy_level!a}, outside "
            f"{MIN_ACCURACY_LEVEL} to {MAX_ACCURACY_LEVEL}"
        )
    if (
        not is_whole_number(conversion.quantized_product_count)
        or not 1 <= conversion.quantized_product_count <= MAX_QUANTIZED_PRODUCT_COUNT
    ):
        raise ConversionRequestError(
            f"{where} quantizes {conversion.quantized_product_count!a} products, outside 1 to "
            f"{MAX_QUANTIZED_PRODUCT_COUNT}"
        )


def judge_conversions(conversions: Sequence[Conversion]) -> None:
    if not conversions:
        raise ConversionRequestError("the request converts nothing")
    if len(conversions) > MAX_CONVERSION_COUNT:
        raise ConversionRequestError(
            f"the request holds {len(conversions)} conversions, more than {MAX_CONVERSION_COUNT}"
        )
    for conversion in conversions:
        judge_conversion(conversion)
    outputs = [conversion.output for conversion in conversions]
    duplicates = sorted({output for output in outputs if outputs.count(output) > 1})
    if duplicates:
        raise ConversionRequestError(f"two conversions write {duplicates[0]}")


def is_whole_number(field: object) -> bool:
    return isinstance(field, int) and not isinstance(field, bool)


def render_conversion_request(request: ConversionRequest) -> bytes:
    document = {
        "input_step": request.input_step,
        "input_name": request.input_name,
        "conversions": [render_conversion(conversion) for conversion in request.conversions],
    }
    return json.dumps(document, indent=1).encode("ascii")


def render_conversion(conversion: Conversion) -> Mapping[str, object]:
    if isinstance(conversion, FloatConversion):
        return {
            "kind": conversion.kind.value,
            "output": conversion.output,
            "added_float32_operators": list(conversion.added_float32_operators),
            "keep_io_types": conversion.keep_io_types,
        }
    return {
        "kind": conversion.kind.value,
        "output": conversion.output,
        "block_size": conversion.block_size,
        "is_symmetric": conversion.is_symmetric,
        "accuracy_level": conversion.accuracy_level,
        "quantized_product_count": conversion.quantized_product_count,
    }


def parse_conversion_request(content: bytes) -> ConversionRequest:
    if len(content) > CONVERSION_REQUEST_MAX_SIZE_BYTES:
        raise ConversionRequestError(
            f"the conversion's request holds {len(content)} bytes, more than "
            f"{CONVERSION_REQUEST_MAX_SIZE_BYTES}"
        )
    try:
        document = json.loads(
            refuse_deep_json(content.decode("ascii"), MAX_JSON_DEPTH),
            object_pairs_hook=refuse_duplicate_keys,
        )
    except (UnicodeDecodeError, ValueError) as error:
        raise ConversionRequestError(
            f"the conversion's request is not ASCII JSON: {error}"
        ) from error
    fields = require_object(document, REQUEST_KEYS, "the conversion's request")
    input_step = fields["input_step"]
    if not isinstance(input_step, str) or not STEP_GRAMMAR.fullmatch(input_step):
        raise ConversionRequestError(
            f"the conversion's input comes from {input_step!a}, not a step under "
            f"{STEP_GRAMMAR.pattern}"
        )
    input_name = fields["input_name"]
    if not isinstance(input_name, str) or not is_file_name(input_name):
        raise ConversionRequestError(f"the conversion's input, {input_name!a}, is not a file name")
    entries = fields["conversions"]
    if not isinstance(entries, list):
        raise ConversionRequestError("the conversion's request holds no list of conversions")
    conversions = tuple(parse_conversion(entry) for entry in entries[: MAX_CONVERSION_COUNT + 1])
    judge_conversions(conversions)
    return ConversionRequest(input_step, input_name, conversions)


def parse_conversion(entry: object) -> Conversion:
    if not isinstance(entry, dict) or "kind" not in entry:
        raise ConversionRequestError("a conversion of the request is not an object with a kind")
    kind = entry["kind"]
    if kind == ConversionKind.EIGHT_BIT_BLOCKS.value:
        fields = require_object(entry, EIGHT_BIT_BLOCKS_KEYS, "a conversion of the request")
        return EightBitBlocksConversion(
            output=require_string(fields["output"]),
            block_size=require_whole_number(fields["block_size"]),
            is_symmetric=require_boolean(fields["is_symmetric"]),
            accuracy_level=require_whole_number(fields["accuracy_level"]),
            quantized_product_count=require_whole_number(fields["quantized_product_count"]),
        )
    if not isinstance(kind, str) or kind not in {float_kind.value for float_kind in FLOAT_KINDS}:
        raise ConversionRequestError(f"a conversion of the request is a {kind!a}, an unknown kind")
    fields = require_object(entry, FLOAT_CONVERSION_KEYS, "a conversion of the request")
    operators = fields["added_float32_operators"]
    if not isinstance(operators, list):
        raise ConversionRequestError("a conversion of the request lists no operators")
    return FloatConversion(
        kind=ConversionKind(kind),
        output=require_string(fields["output"]),
        added_float32_operators=tuple(
            require_string(operator) for operator in operators[: MAX_OPERATOR_COUNT + 1]
        ),
        keep_io_types=require_boolean(fields["keep_io_types"]),
    )


def require_object(document: object, keys: frozenset[str], what: str) -> Mapping[str, object]:
    if not isinstance(document, dict) or set(document) != keys:
        raise ConversionRequestError(f"{what} is not an object of exactly {sorted(keys)}")
    return document


def require_string(field: object) -> str:
    if not isinstance(field, str):
        raise ConversionRequestError(f"a conversion of the request holds {field!a}, not a string")
    return field


def require_whole_number(field: object) -> int:
    if not isinstance(field, int) or isinstance(field, bool):
        raise ConversionRequestError(
            f"a conversion of the request holds {field!a}, not a whole number"
        )
    return field


def require_boolean(field: object) -> bool:
    if not isinstance(field, bool):
        raise ConversionRequestError(f"a conversion of the request holds {field!a}, not a boolean")
    return field


def refuse_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    keys = [key for key, _ in pairs]
    if len(set(keys)) != len(keys):
        raise ValueError("a key appears twice in an object")
    return dict(pairs)
