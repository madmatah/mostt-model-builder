"""Ported from parakeet-rs's scripts/onnx_export.py and scripts/export_parakeet_tdt.py, at
cadebd4469921ff426bf179e1cfd78fbe6fd9e05, under its MIT license, which CREDITS.md keeps."""

import tempfile
import types
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import onnx
import onnx_ir
import torch
from nemo.core.classes.common import typecheck
from nemo.core.classes.exportable import Exportable
from onnxscript import optimizer

# torch, onnx and NeMo are installed in the export image alone, so mypy on the host knows nothing
# of their types.
type TorchModule = Any
type TorchTensor = Any
type OnnxModel = Any
type TorchDimension = Any

# Opset 23 writes attention as the ONNX Attention operator, and keeps the LSTMs as LSTM.
OPSET = 23
SELF_ATTENTION_SWITCH = "use_pytorch_sdpa"
SELF_ATTENTION_BACKENDS = "use_pytorch_sdpa_backends"
POSITION_ENCODING_TYPE = "RelPositionalEncoding"
# ONNX Runtime cannot read the dynamo export's shape constants from external data, so the tensors
# under 1 KiB stay in the graph's file.
EXTERNAL_DATA_THRESHOLD_BYTES = 1024
FOLDED_TENSOR_SIZE_LIMIT_BYTES = 2**31
RAW_DECODER_JOINT_FILE_NAME = "decoder_joint-raw.onnx"

type DynamicAxes = Mapping[str, Mapping[int, object] | Sequence[int]]


class OnnxExportError(Exception):
    pass


# NeMo's attention runs through torch's scaled_dot_product_attention, so that each layer exports as
# one ONNX Attention node, with the relative positions as an additive mask.
def use_scaled_dot_product_attention(module: TorchModule) -> int:
    attention_count = 0
    for submodule in module.modules():
        if hasattr(submodule, SELF_ATTENTION_SWITCH):
            setattr(submodule, SELF_ATTENTION_SWITCH, True)
            setattr(submodule, SELF_ATTENTION_BACKENDS, None)
            attention_count += 1
    return attention_count


# NeMo's RelPositionalEncoding slices its table as pe[:, start:end], whose size torch.export
# cannot follow on a symbolic length; narrow() takes the same 2 * L - 1 rows with an exact length.
# NeMo's encoder passes x and cache_len by name.
def forward_with_exact_slices(
    position_encoding: TorchModule, x: TorchTensor, cache_len: int = 0
) -> tuple[TorchTensor, TorchTensor]:
    features = x
    if position_encoding.xscale:
        features = features * position_encoding.xscale
    input_length = features.size(1) + cache_len
    center_position = position_encoding.pe.size(1) // 2 + 1
    position_embedding = position_encoding.pe.narrow(
        1, center_position - input_length, 2 * input_length - 1
    )
    if position_encoding.dropout_emb:
        position_embedding = position_encoding.dropout_emb(position_embedding)
    return position_encoding.dropout(features), position_embedding


def use_exact_position_slices(module: TorchModule) -> int:
    encoding_count = 0
    for submodule in module.modules():
        if type(submodule).__name__ == POSITION_ENCODING_TYPE:
            submodule.forward = types.MethodType(forward_with_exact_slices, submodule)
            encoding_count += 1
    return encoding_count


def build_dynamic_shapes(
    input_names: Sequence[str], dynamic_axes: DynamicAxes
) -> tuple[dict[int, TorchDimension] | None, ...]:
    bounds: dict[str, dict[int, int | None]] = {}
    for name, axes in dynamic_axes.items():
        axis_bounds = axes.items() if isinstance(axes, Mapping) else ((axis, None) for axis in axes)
        bounds[name] = {
            int(axis): (bound if isinstance(bound, int) else None) for axis, bound in axis_bounds
        }
    return tuple(
        {
            axis: torch.export.Dim.AUTO(max=bound) if bound else torch.export.Dim.AUTO
            for axis, bound in bounds[name].items()
        }
        if name in bounds
        else None
        for name in input_names
    )


# Dim.AUTO lets torch.export fix a size that it cannot keep dynamic, so each axis that the export
# was given as dynamic is checked in the graph it wrote.
def refuse_fixed_axes(path: Path, dynamic_axes: DynamicAxes) -> None:
    graph_inputs = {
        graph_input.name: graph_input
        for graph_input in onnx.load(str(path), load_external_data=False).graph.input
    }
    for name, axes in dynamic_axes.items():
        dimensions = graph_inputs[name].type.tensor_type.shape.dim
        fixed_axes = [int(axis) for axis in axes if dimensions[int(axis)].HasField("dim_value")]
        if fixed_axes:
            raise OnnxExportError(f"the axes {fixed_axes} of {name} became fixed in the export")


def export_module(
    module: TorchModule,
    example_inputs: Sequence[TorchTensor],
    path: Path,
    input_names: Sequence[str],
    output_names: Sequence[str],
    dynamic_axes: DynamicAxes,
) -> None:
    module.eval()
    with torch.no_grad():
        torch.onnx.export(
            module,
            tuple(example_inputs),
            str(path),
            input_names=list(input_names),
            output_names=list(output_names),
            dynamic_shapes=build_dynamic_shapes(input_names, dynamic_axes),
            opset_version=OPSET,
            dynamo=True,
        )
    refuse_fixed_axes(path, dynamic_axes)


# The dynamo exporter builds an LSTM's weights from torch's at run time, reordering its gates;
# folded, they are plain initializers, as in any other ONNX LSTM.
def fold_constants(model: OnnxModel) -> OnnxModel:
    ir_model = onnx_ir.serde.deserialize_model(model)
    optimizer.fold_constants(
        ir_model,
        input_size_limit=FOLDED_TENSOR_SIZE_LIMIT_BYTES,
        output_size_limit=FOLDED_TENSOR_SIZE_LIMIT_BYTES,
    )
    onnx_ir.passes.common.RemoveUnusedNodesPass()(ir_model)
    return onnx_ir.serde.serialize_model(ir_model)


# The dynamo exporter records on each node the stack trace and the module path it came from, with
# paths of the container, which a published file must not carry.
def strip_node_metadata(model: OnnxModel) -> int:
    stripped_count = 0
    graphs = [model.graph, *model.functions]
    while graphs:
        graph = graphs.pop()
        for node in graph.node:
            stripped_count += len(node.metadata_props)
            del node.metadata_props[:]
            for attribute in node.attribute:
                if attribute.HasField("g"):
                    graphs.append(attribute.g)
                graphs.extend(attribute.graphs)
    return stripped_count


def save_model(model: OnnxModel, path: Path, external_data_name: str | None = None) -> None:
    strip_node_metadata(model)
    if external_data_name is None:
        onnx.save_model(model, str(path))
        return
    onnx.save_model(
        model,
        str(path),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=external_data_name,
        size_threshold=EXTERNAL_DATA_THRESHOLD_BYTES,
    )


# NeMo gives the decoder and joint of one prediction step its own example inputs, names and
# dynamic axes once each of its exportable modules is prepared for an export.
def export_decoder_joint(model: TorchModule, path: Path) -> None:
    decoder_joint = model.decoder_joint
    decoder_joint.eval()
    typecheck.set_typecheck_enabled(enabled=False)
    for submodule in decoder_joint.modules():
        if isinstance(submodule, Exportable) and submodule is not decoder_joint:
            submodule._prepare_for_export(noreplace=True)
    with tempfile.TemporaryDirectory() as raw_directory:
        raw_path = Path(raw_directory) / RAW_DECODER_JOINT_FILE_NAME
        export_module(
            decoder_joint,
            decoder_joint.input_example(),
            raw_path,
            input_names=decoder_joint.input_names,
            output_names=decoder_joint.output_names,
            dynamic_axes=decoder_joint.dynamic_shapes_for_export(),
        )
        save_model(fold_constants(onnx.load(str(raw_path))), path)
