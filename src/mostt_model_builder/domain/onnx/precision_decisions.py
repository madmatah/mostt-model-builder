from collections import Counter
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

from mostt_model_builder.domain.onnx.conversion_request import ConversionKind
from mostt_model_builder.domain.onnx.element_types import ElementType
from mostt_model_builder.domain.onnx.graph_paths import (
    GraphAttribute,
    NamedTensor,
    holds_subgraph,
    index_graph_readers,
    list_attribute_graphs,
    list_computation_to_integer_casts,
    list_descendants,
)

# ONNX Runtime 1.31's CUDA provider has no bfloat16 kernel for these operators, and refuses the
# whole graph when one of them is left in bfloat16; the Floor comes from torch 2.15's export.
CORE_FLOAT32_OPERATORS_IN_BFLOAT16 = ("Pad", "Tile", "Floor")
# onnxconverter-common's defaults: a float32 weight outside them is clamped into float16's range,
# and each such value is counted.
MIN_POSITIVE_VALUE = 1e-7
MAX_FINITE_VALUE = 1e4


class PrecisionDecisionError(ValueError):
    pass


class ConvertedTensorType(Protocol):
    @property
    def elem_type(self) -> int: ...


class ConvertedType(Protocol):
    @property
    def tensor_type(self) -> ConvertedTensorType: ...


class ConvertedValueInfo(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def type(self) -> ConvertedType: ...


class ConvertedNode(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def op_type(self) -> str: ...

    @property
    def domain(self) -> str: ...

    @property
    def input(self) -> Sequence[str]: ...

    @property
    def output(self) -> Sequence[str]: ...

    @property
    def attribute(self) -> Sequence[GraphAttribute["ConvertedGraph"]]: ...


class ConvertedGraph(Protocol):
    @property
    def node(self) -> Sequence[ConvertedNode]: ...

    @property
    def input(self) -> Sequence[NamedTensor]: ...

    @property
    def output(self) -> Sequence[ConvertedValueInfo]: ...

    @property
    def initializer(self) -> Sequence[NamedTensor]: ...


@dataclass(frozen=True, slots=True)
class KeptNodes:
    node_names: tuple[str, ...]
    set_aside_tensor_names: tuple[str, ...]


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


def decide_kept_nodes(graph: ConvertedGraph, kept_inputs: Sequence[str]) -> KeptNodes:
    name_counts = Counter(node.name for subgraph in list_graphs(graph) for node in subgraph.node)
    readers = index_graph_readers(graph.node)
    kept_indices: set[int] = set()
    for kept_input in kept_inputs:
        kept_indices.update(find_kept_computation(graph, kept_input, readers, name_counts))
    kept_nodes = [graph.node[node_index] for node_index in sorted(kept_indices)]
    return KeptNodes(
        node_names=tuple(node.name for node in kept_nodes),
        set_aside_tensor_names=tuple(name for node in kept_nodes for name in node.output if name),
    )


# onnxconverter-common 1.16.0 keeps nodes by name alone, in every graph it converts, so an unnamed
# node would keep every unnamed node, and a shared name a node outside the computation; it leaves a
# kept node's subgraphs unconverted while it converts the outer tensors they read; and it casts to
# float16 a float output of the graph that a kept node writes, whatever its declarations.
def find_kept_computation(
    graph: ConvertedGraph,
    kept_input: str,
    readers: Mapping[str, Sequence[int]],
    name_counts: Counter[str],
) -> tuple[int, ...]:
    where = f"the computation kept in float32 from {kept_input!a}"
    graph_inputs = {value_info.name for value_info in graph.input}
    if kept_input not in graph_inputs:
        raise PrecisionDecisionError(f"{where} starts from no input of the graph")
    computation = list_computation_to_integer_casts(graph.node, readers, (kept_input,))
    if not computation:
        raise PrecisionDecisionError(f"{where} holds no node")
    other_computations = list_other_computations(graph, readers, graph_inputs - {kept_input})
    for node_index in computation:
        refuse_kept_node(
            graph.node[node_index], node_index in other_computations, name_counts, where
        )
    float_outputs = {
        value_info.name
        for value_info in graph.output
        if value_info.type.tensor_type.elem_type == ElementType.FLOAT
    }
    for node_index in computation:
        for name in graph.node[node_index].output:
            if name in float_outputs:
                raise PrecisionDecisionError(
                    f"{where} writes {name!a}, a float output of the graph, which "
                    "onnxconverter-common would cast to float16"
                )
    return computation


# A node that another input's computation reaches would keep that computation in float32 too: the
# sign of a misdeclared input, or of lengths that the graph does not compute apart from the audio.
def list_other_computations(
    graph: ConvertedGraph, readers: Mapping[str, Sequence[int]], other_inputs: Collection[str]
) -> set[int]:
    initializer_names = {initializer.name for initializer in graph.initializer}
    return list_descendants(
        graph.node, readers, [name for name in other_inputs if name not in initializer_names]
    )


def refuse_kept_node(
    node: ConvertedNode, reads_other_inputs: bool, name_counts: Counter[str], where: str
) -> None:
    if not node.name:
        raise PrecisionDecisionError(f"{where} holds a node without a name")
    if name_counts[node.name] > 1:
        raise PrecisionDecisionError(
            f"{where} holds the node {node.name!a}, whose name another node of the model bears"
        )
    if holds_subgraph(node):
        raise PrecisionDecisionError(
            f"{where} holds the node {node.name!a}, which holds a subgraph"
        )
    if reads_other_inputs:
        raise PrecisionDecisionError(
            f"{where} holds the node {node.name!a}, which also reads another input of the graph "
            "or its computation"
        )


def list_graphs(graph: ConvertedGraph) -> list[ConvertedGraph]:
    graphs = []
    pending = [graph]
    while pending:
        current = pending.pop()
        graphs.append(current)
        pending.extend(
            subgraph
            for node in current.node
            for attribute in node.attribute
            for subgraph in list_attribute_graphs(attribute)
        )
    return graphs
