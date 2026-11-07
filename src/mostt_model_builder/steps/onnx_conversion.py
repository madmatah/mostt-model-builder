import gc
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import ml_dtypes
import numpy
import onnx
from onnx import numpy_helper
from onnx.external_data_helper import load_external_data_for_model
from onnxconverter_common import float16
from onnxruntime.quantization.matmul_nbits_quantizer import (
    DefaultWeightOnlyQuantConfig,
    MatMulNBitsQuantizer,
)

from mostt_model_builder.domain.containers.container_paths import (
    INPUTS_IN_THE_CONTAINER,
    OUTPUT_IN_THE_CONTAINER,
    REQUEST_IN_THE_CONTAINER,
)
from mostt_model_builder.domain.onnx.conversion_report import (
    CONVERSION_REPORT_FILE_NAME,
    ConversionReport,
    ConvertedFile,
    render_conversion_report,
)
from mostt_model_builder.domain.onnx.conversion_request import (
    CONVERSION_REQUEST_FILE_NAME,
    EIGHT_BIT_BLOCKS_BITS,
    ConversionKind,
    EightBitBlocksConversion,
    FloatConversion,
    parse_conversion_request,
)
from mostt_model_builder.domain.onnx.graph_rules import (
    EIGHT_BIT_BLOCKS_OPERATOR,
    ONNX_RUNTIME_DOMAIN,
)
from mostt_model_builder.domain.onnx.precision_decisions import (
    MAX_FINITE_VALUE,
    MIN_POSITIVE_VALUE,
    ElementType,
    list_kept_float32_operators,
    realign_cast_target,
    retype_for_bfloat16,
)
from mostt_model_builder.steps.output_modes import finish_outputs

# onnx, numpy, ml_dtypes, onnxconverter-common and ONNX Runtime are installed in the conversion's
# image alone, so mypy on the host knows nothing of their types.
type OnnxModel = Any
type OnnxGraph = Any
type OnnxTensor = Any

CAST_OPERATOR = "Cast"
CAST_TARGET_ATTRIBUTE = "to"


@dataclass(frozen=True, slots=True)
class ClampCounts:
    initializer_count: int
    below_min_positive_count: int
    above_max_finite_count: int


def count_clamped_values(model: OnnxModel) -> ClampCounts:
    per_initializer = []
    for initializer in model.graph.initializer:
        if initializer.data_type != ElementType.FLOAT:
            continue
        magnitudes = numpy.abs(numpy_helper.to_array(initializer))
        below = int(((magnitudes > 0) & (magnitudes < MIN_POSITIVE_VALUE)).sum())
        above = int((magnitudes > MAX_FINITE_VALUE).sum())
        if below or above:
            per_initializer.append((below, above))
    return ClampCounts(
        initializer_count=len(per_initializer),
        below_min_positive_count=sum(below for below, _ in per_initializer),
        above_max_finite_count=sum(above for _, above in per_initializer),
    )


def realign_casts(model: OnnxModel) -> int:
    declared_types = {
        value_info.name: value_info.type.tensor_type.elem_type
        for value_info in model.graph.value_info
    }
    realigned_count = 0
    for node in model.graph.node:
        if node.op_type != CAST_OPERATOR:
            continue
        for attribute in node.attribute:
            if attribute.name != CAST_TARGET_ATTRIBUTE:
                continue
            target = realign_cast_target(attribute.i, declared_types.get(node.output[0]))
            if target != attribute.i:
                attribute.i = target
                realigned_count += 1
    return realigned_count


def retype_tensor(tensor: OnnxTensor) -> None:
    if tensor.data_type != ElementType.FLOAT16:
        return
    bfloat16_elements = (
        numpy_helper.to_array(tensor).astype(numpy.float32).astype(ml_dtypes.bfloat16)
    )
    tensor.CopyFrom(numpy_helper.from_array(bfloat16_elements, tensor.name))


def retype_graph(graph: OnnxGraph) -> None:
    for tensor in graph.initializer:
        retype_tensor(tensor)
    for value_info in (*graph.value_info, *graph.input, *graph.output):
        tensor_type = value_info.type.tensor_type
        tensor_type.elem_type = retype_for_bfloat16(tensor_type.elem_type)
    for node in graph.node:
        for attribute in node.attribute:
            if attribute.name == CAST_TARGET_ATTRIBUTE:
                attribute.i = retype_for_bfloat16(attribute.i)
            if attribute.type == onnx.AttributeProto.TENSOR:
                retype_tensor(attribute.t)
            if attribute.type == onnx.AttributeProto.GRAPH:
                retype_graph(attribute.g)


def load_with_inferred_shapes(input_path: Path) -> OnnxModel:
    model = onnx.load(str(input_path), load_external_data=False)
    model = onnx.shape_inference.infer_shapes(model, strict_mode=False, data_prop=True)
    load_external_data_for_model(model, str(input_path.parent))
    return model


def convert_float(input_path: Path, conversion: FloatConversion) -> tuple[OnnxModel, ConvertedFile]:
    model = load_with_inferred_shapes(input_path)
    clamps = count_clamped_values(model)
    kept_operators = list_kept_float32_operators(
        conversion.kind, conversion.added_float32_operators
    )
    converted = float16.convert_float_to_float16(
        model,
        min_positive_val=MIN_POSITIVE_VALUE,
        max_finite_val=MAX_FINITE_VALUE,
        keep_io_types=conversion.keep_io_types,
        disable_shape_infer=True,
        op_block_list=[*float16.DEFAULT_OP_BLOCK_LIST, *kept_operators],
    )
    realigned_count = realign_casts(converted)
    if conversion.kind is ConversionKind.BFLOAT16:
        retype_graph(converted.graph)
    return converted, ConvertedFile(
        kind=conversion.kind,
        output=conversion.output,
        node_count=len(converted.graph.node),
        quantized_product_count=count_eight_bit_block_products(converted),
        kept_float32_operators=kept_operators,
        clamped_initializer_count=clamps.initializer_count,
        below_min_positive_count=clamps.below_min_positive_count,
        above_max_finite_count=clamps.above_max_finite_count,
        realigned_cast_count=realigned_count,
    )


def convert_eight_bit_blocks(
    input_path: Path, conversion: EightBitBlocksConversion
) -> tuple[OnnxModel, ConvertedFile]:
    model = onnx.load(str(input_path))
    quantizer = MatMulNBitsQuantizer(
        model,
        bits=EIGHT_BIT_BLOCKS_BITS,
        block_size=conversion.block_size,
        is_symmetric=conversion.is_symmetric,
        accuracy_level=conversion.accuracy_level,
        algo_config=DefaultWeightOnlyQuantConfig(
            block_size=conversion.block_size,
            is_symmetric=conversion.is_symmetric,
            accuracy_level=conversion.accuracy_level,
            bits=EIGHT_BIT_BLOCKS_BITS,
        ),
    )
    quantizer.process()
    converted = quantizer.model.model
    return converted, ConvertedFile(
        kind=conversion.kind,
        output=conversion.output,
        node_count=len(converted.graph.node),
        quantized_product_count=count_eight_bit_block_products(converted),
        kept_float32_operators=(),
        clamped_initializer_count=0,
        below_min_positive_count=0,
        above_max_finite_count=0,
        realigned_cast_count=0,
    )


def count_eight_bit_block_products(model: OnnxModel) -> int:
    return sum(
        1
        for node in model.graph.node
        if node.domain == ONNX_RUNTIME_DOMAIN and node.op_type == EIGHT_BIT_BLOCKS_OPERATOR
    )


def convert_input(
    request_directory: Path, inputs_directory: Path, output_directory: Path
) -> ConversionReport:
    request = parse_conversion_request(
        (request_directory / CONVERSION_REQUEST_FILE_NAME).read_bytes()
    )
    input_path = inputs_directory / request.input_step / request.input_name
    converted_files = []
    for conversion in request.conversions:
        if isinstance(conversion, EightBitBlocksConversion):
            converted, converted_file = convert_eight_bit_blocks(input_path, conversion)
        else:
            converted, converted_file = convert_float(input_path, conversion)
        onnx.save_model(converted, str(output_directory / conversion.output))
        converted_files.append(converted_file)
        del converted
        gc.collect()
    return ConversionReport(tuple(converted_files))


def main() -> None:
    output_directory = Path(OUTPUT_IN_THE_CONTAINER)
    report = convert_input(
        Path(REQUEST_IN_THE_CONTAINER), Path(INPUTS_IN_THE_CONTAINER), output_directory
    )
    (output_directory / CONVERSION_REPORT_FILE_NAME).write_bytes(render_conversion_report(report))
    finish_outputs(output_directory)


if __name__ == "__main__":
    main()
