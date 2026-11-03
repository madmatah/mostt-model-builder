from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

from mostt_model_builder.domain.onnx.inspection_report import (
    CONSTANT_PRODUCERS,
    MAX_TRACE_LENGTH,
    ExternalTensor,
    MatrixProduct,
    NodeCount,
    NodeScope,
    WeightSource,
)

# The values of AttributeProto.AttributeType and of TensorProto.DataLocation, and the keys of a
# tensor's external_data, as onnx/onnx.proto defines them: the messages carry them as plain values.
TENSOR_ATTRIBUTE = 4
GRAPH_ATTRIBUTE = 5
TENSORS_ATTRIBUTE = 9
GRAPHS_ATTRIBUTE = 10
SPARSE_TENSOR_ATTRIBUTE = 11
SPARSE_TENSORS_ATTRIBUTE = 12
EXTERNAL_DATA_LOCATION = 1
EXTERNAL_LOCATION_KEY = "location"
EXTERNAL_OFFSET_KEY = "offset"
EXTERNAL_LENGTH_KEY = "length"
# A TrainingInfoProto of onnx/onnx.proto holds two graphs, its initialization and its algorithm.
GRAPHS_PER_TRAINING_INFO = 2
MATRIX_PRODUCTS = frozenset({"MatMul", "Gemm"})
WEIGHT_INPUT_INDEX = 1


class OnnxStringEntry(Protocol):
    @property
    def key(self) -> str: ...

    @property
    def value(self) -> str: ...


class OnnxTensor(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def dims(self) -> Sequence[int]: ...

    @property
    def data_location(self) -> int: ...

    @property
    def external_data(self) -> Sequence[OnnxStringEntry]: ...


class OnnxSparseTensor(Protocol):
    @property
    def values(self) -> OnnxTensor: ...

    @property
    def indices(self) -> OnnxTensor: ...


class OnnxAttribute(Protocol):
    @property
    def type(self) -> int: ...

    @property
    def t(self) -> OnnxTensor: ...

    @property
    def tensors(self) -> Sequence[OnnxTensor]: ...

    @property
    def sparse_tensor(self) -> OnnxSparseTensor: ...

    @property
    def sparse_tensors(self) -> Sequence[OnnxSparseTensor]: ...

    @property
    def g(self) -> "OnnxGraph": ...

    @property
    def graphs(self) -> Sequence["OnnxGraph"]: ...


class OnnxNode(Protocol):
    @property
    def op_type(self) -> str: ...

    @property
    def domain(self) -> str: ...

    @property
    def input(self) -> Sequence[str]: ...

    @property
    def output(self) -> Sequence[str]: ...

    @property
    def attribute(self) -> Sequence[OnnxAttribute]: ...

    @property
    def metadata_props(self) -> Sequence[OnnxStringEntry]: ...


class OnnxGraph(Protocol):
    @property
    def node(self) -> Sequence[OnnxNode]: ...

    @property
    def initializer(self) -> Sequence[OnnxTensor]: ...

    @property
    def sparse_initializer(self) -> Sequence[OnnxSparseTensor]: ...


class OnnxFunction(Protocol):
    @property
    def node(self) -> Sequence[OnnxNode]: ...

    @property
    def attribute_proto(self) -> Sequence[OnnxAttribute]: ...


class OnnxModel(Protocol):
    @property
    def graph(self) -> OnnxGraph: ...

    @property
    def functions(self) -> Sequence[OnnxFunction]: ...

    @property
    def training_info(self) -> Sequence[object]: ...


@dataclass(frozen=True, slots=True)
class ScopedGraph:
    scope: NodeScope
    nodes: Sequence[OnnxNode]
    initializers: Sequence[OnnxTensor]
    sparse_initializers: Sequence[OnnxSparseTensor]
    default_attributes: Sequence[OnnxAttribute]
    visible_initializer_ranks: Mapping[str, int]


def walk_graphs(model: OnnxModel) -> Iterator[ScopedGraph]:
    yield from walk_graph(model.graph, NodeScope.GRAPH, {})
    for function in model.functions:
        yield ScopedGraph(NodeScope.FUNCTION, function.node, (), (), function.attribute_proto, {})
        yield from walk_subgraphs([*function.attribute_proto, *list_attributes(function.node)], {})


def walk_graph(
    graph: OnnxGraph, scope: NodeScope, outer_initializer_ranks: Mapping[str, int]
) -> Iterator[ScopedGraph]:
    visible_initializer_ranks = {
        **outer_initializer_ranks,
        **{initializer.name: len(initializer.dims) for initializer in graph.initializer},
    }
    yield ScopedGraph(
        scope,
        graph.node,
        graph.initializer,
        graph.sparse_initializer,
        (),
        visible_initializer_ranks,
    )
    yield from walk_subgraphs(list_attributes(graph.node), visible_initializer_ranks)


def walk_subgraphs(
    attributes: Sequence[OnnxAttribute], outer_initializer_ranks: Mapping[str, int]
) -> Iterator[ScopedGraph]:
    for attribute in attributes:
        for subgraph in list_attribute_graphs(attribute):
            yield from walk_graph(subgraph, NodeScope.SUBGRAPH, outer_initializer_ranks)


def list_attributes(nodes: Sequence[OnnxNode]) -> list[OnnxAttribute]:
    return [attribute for node in nodes for attribute in node.attribute]


def list_attribute_graphs(attribute: OnnxAttribute) -> Sequence[OnnxGraph]:
    if attribute.type == GRAPH_ATTRIBUTE:
        return (attribute.g,)
    if attribute.type == GRAPHS_ATTRIBUTE:
        return attribute.graphs
    return ()


def list_attribute_tensors(attribute: OnnxAttribute) -> list[OnnxTensor]:
    if attribute.type == TENSOR_ATTRIBUTE:
        return [attribute.t]
    if attribute.type == TENSORS_ATTRIBUTE:
        return list(attribute.tensors)
    if attribute.type == SPARSE_TENSOR_ATTRIBUTE:
        return list_sparse_parts([attribute.sparse_tensor])
    if attribute.type == SPARSE_TENSORS_ATTRIBUTE:
        return list_sparse_parts(attribute.sparse_tensors)
    return []


def list_sparse_parts(sparse_tensors: Sequence[OnnxSparseTensor]) -> list[OnnxTensor]:
    return [part for sparse in sparse_tensors for part in (sparse.values, sparse.indices)]


def list_graph_tensors(graph: ScopedGraph) -> list[OnnxTensor]:
    attributes = [*graph.default_attributes, *list_attributes(graph.nodes)]
    return [
        *graph.initializers,
        *list_sparse_parts(graph.sparse_initializers),
        *(tensor for attribute in attributes for tensor in list_attribute_tensors(attribute)),
    ]


def count_nodes(model: OnnxModel) -> tuple[NodeCount, ...]:
    counts: Counter[tuple[NodeScope, str, str]] = Counter(
        (graph.scope, node.domain, node.op_type)
        for graph in walk_graphs(model)
        for node in graph.nodes
    )
    return tuple(
        NodeCount(scope, domain, operator, count)
        for (scope, domain, operator), count in sorted(counts.items())
    )


def count_nodes_with_metadata(model: OnnxModel) -> int:
    return sum(
        1 for graph in walk_graphs(model) for node in graph.nodes if len(node.metadata_props) > 0
    )


def count_sparse_initializers(model: OnnxModel) -> int:
    return sum(len(graph.sparse_initializers) for graph in walk_graphs(model))


def count_training_graphs(model: OnnxModel) -> int:
    return GRAPHS_PER_TRAINING_INFO * len(model.training_info)


def list_external_tensors(model: OnnxModel) -> tuple[ExternalTensor, ...]:
    tensors = [
        describe_external_tensor(tensor)
        for graph in walk_graphs(model)
        for tensor in list_graph_tensors(graph)
        if tensor.data_location == EXTERNAL_DATA_LOCATION
    ]
    return tuple(sorted(tensors, key=lambda tensor: (tensor.location, tensor.offset_bytes)))


def describe_external_tensor(tensor: OnnxTensor) -> ExternalTensor:
    entries = {entry.key: entry.value for entry in tensor.external_data}
    return ExternalTensor(
        location=entries.get(EXTERNAL_LOCATION_KEY, ""),
        offset_bytes=int(entries.get(EXTERNAL_OFFSET_KEY, "0")),
        length_bytes=int(entries.get(EXTERNAL_LENGTH_KEY, "0")),
    )


def list_products(model: OnnxModel) -> tuple[MatrixProduct, ...]:
    return tuple(product for graph in walk_graphs(model) for product in trace_products(graph))


def trace_products(graph: ScopedGraph) -> list[MatrixProduct]:
    producers = {output: node for node in graph.nodes for output in node.output}
    return [
        trace_weight(node, producers, graph)
        for node in graph.nodes
        if node.op_type in MATRIX_PRODUCTS and len(node.input) > WEIGHT_INPUT_INDEX
    ]


def trace_weight(
    product: OnnxNode, producers: Mapping[str, OnnxNode], graph: ScopedGraph
) -> MatrixProduct:
    ranks = graph.visible_initializer_ranks
    trace: list[str] = []
    name = product.input[WEIGHT_INPUT_INDEX]
    while name not in ranks:
        producer = producers.get(name)
        if producer is None or producer.op_type not in CONSTANT_PRODUCERS:
            return MatrixProduct(
                graph.scope, product.op_type, tuple(trace), WeightSource.COMPUTED, None
            )
        trace.append(producer.op_type)
        if producer.op_type == "Constant":
            return MatrixProduct(
                graph.scope, product.op_type, tuple(trace), WeightSource.CONSTANT, None
            )
        if len(trace) >= MAX_TRACE_LENGTH or not producer.input:
            return MatrixProduct(
                graph.scope, product.op_type, tuple(trace), WeightSource.COMPUTED, None
            )
        name = producer.input[0]
    return MatrixProduct(
        graph.scope, product.op_type, tuple(trace), WeightSource.INITIALIZER, ranks[name]
    )
