from collections.abc import Sequence

from mostt_model_builder.domain.onnx.conversion_request import ConversionKind
from mostt_model_builder.domain.onnx.element_types import ElementType

# ONNX Runtime 1.31's CUDA provider has no bfloat16 kernel for these operators, and refuses the
# whole graph when one of them is left in bfloat16; the Floor comes from torch 2.15's export.
CORE_FLOAT32_OPERATORS_IN_BFLOAT16 = ("Pad", "Tile", "Floor")
# onnxconverter-common's defaults: a float32 weight outside them is clamped into float16's range,
# and each such value is counted.
MIN_POSITIVE_VALUE = 1e-7
MAX_FINITE_VALUE = 1e4


def list_kept_float32_operators(
    kind: ConversionKind, added_float32_operators: Sequence[str]
) -> tuple[str, ...]:
    core_operators = CORE_FLOAT32_OPERATORS_IN_BFLOAT16 if kind is ConversionKind.BFLOAT16 else ()
    return tuple(dict.fromkeys((*core_operators, *added_float32_operators)))


# onnxconverter-common declares some Cast outputs float16 and leaves their target at float32, so
# that the graph's types contradict each other: such a Cast targets the type its output declares.
def realign_cast_target(cast_target: int, output_element_type: int | None) -> int:
    if cast_target == ElementType.FLOAT and output_element_type == ElementType.FLOAT16:
        return ElementType.FLOAT16
    return cast_target


def retype_for_bfloat16(element_type: int) -> int:
    if element_type == ElementType.FLOAT16:
        return ElementType.BFLOAT16
    return element_type
