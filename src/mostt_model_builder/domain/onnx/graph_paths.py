from collections import defaultdict
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

from mostt_model_builder.domain.onnx.element_types import INTEGER_ELEMENT_TYPES

# The values of AttributeProto.AttributeType that hold graphs, as onnx/onnx.proto defines them: the
# messages carry them as plain values.
GRAPH_ATTRIBUTE = 5
GRAPHS_ATTRIBUTE = 10
ONNX_DOMAINS = frozenset({"", "ai.onnx"})
# Shape and Size write the shape of their data input, never its values, so a computation does not
# reach back through them: the shape of a float16 tensor is no integer computed through float16.
SHAPE_OPERATORS = frozenset({"Shape", "Size"})
CAST_OPERATOR = "Cast"
CAST_TARGET_ATTRIBUTE = "to"


class NamedTensor(Protocol):
    @property
    def name(self) -> str: ...


class GraphAttribute[Graph](Protocol):
    @property
    def type(self) -> int: ...

    @property
    def name(self) -> str: ...

    @property
    def i(self) -> int: ...

    @property
    def g(self) -> Graph: ...

    @property
    def graphs(self) -> Sequence[Graph]: ...


class PathNode(Protocol):
    @property
    def op_type(self) -> str: ...

    @property
    def domain(self) -> str: ...

    @property
    def input(self) -> Sequence[str]: ...

    @property
    def output(self) -> Sequence[str]: ...

    @property
    def attribute(self) -> Sequence[GraphAttribute["PathGraph"]]: ...


class PathGraph(Protocol):
    @property
    def node(self) -> Sequence[PathNode]: ...

    @property
    def input(self) -> Sequence[NamedTensor]: ...

    @property
    def initializer(self) -> Sequence[NamedTensor]: ...


def list_attribute_graphs[Graph](attribute: GraphAttribute[Graph]) -> Sequence[Graph]:
    if attribute.type == GRAPH_ATTRIBUTE:
        return (attribute.g,)
    if attribute.type == GRAPHS_ATTRIBUTE:
        return attribute.graphs
    return ()


def holds_subgraph(node: PathNode) -> bool:
    return any(list_attribute_graphs(attribute) for attribute in node.attribute)


def list_defined_names(graph: PathGraph) -> frozenset[str]:
    return frozenset(
        {
            *(value_info.name for value_info in graph.input),
            *(initializer.name for initializer in graph.initializer),
            *(name for node in graph.node for name in node.output),
        }
    )


# ONNX lets a subgraph read the tensors of the graphs around it without naming them among its
# inputs, so a node that holds one reads every name that the subgraph uses at any depth and that no
# graph between them defines. The empty name marks an omitted optional input, which reads nothing.
def list_node_reads(node: PathNode) -> frozenset[str]:
    reads = {name for name in node.input if name}
    pending: list[tuple[PathGraph, frozenset[str]]] = [
        (subgraph, frozenset())
        for attribute in node.attribute
        for subgraph in list_attribute_graphs(attribute)
    ]
    while pending:
        graph, outer_defined_names = pending.pop()
        defined_names = outer_defined_names | list_defined_names(graph)
        for inner_node in graph.node:
            reads.update(name for name in inner_node.input if name and name not in defined_names)
            pending.extend(
                (subgraph, defined_names)
                for attribute in inner_node.attribute
                for subgraph in list_attribute_graphs(attribute)
            )
    return frozenset(reads)


@dataclass(frozen=True, slots=True)
class GraphPaths:
    value_reads: tuple[frozenset[str], ...]
    written_names: tuple[tuple[str, ...], ...]
    descendants: frozenset[int]
    value_order: tuple[int, ...]


def reads_shape_alone(node: PathNode) -> bool:
    return node.domain in ONNX_DOMAINS and node.op_type in SHAPE_OPERATORS


def casts_to_integer(node: PathNode) -> bool:
    return (
        node.domain in ONNX_DOMAINS
        and node.op_type == CAST_OPERATOR
        and any(
            attribute.name == CAST_TARGET_ATTRIBUTE and attribute.i in INTEGER_ELEMENT_TYPES
            for attribute in node.attribute
        )
    )


def ends_no_walk(_node: PathNode) -> bool:
    return False


def index_readers(reads: Sequence[frozenset[str]]) -> dict[str, list[int]]:
    readers: defaultdict[str, list[int]] = defaultdict(list)
    for node_index, node_reads in enumerate(reads):
        for name in node_reads:
            readers[name].append(node_index)
    return dict(readers)


def index_writers(nodes: Sequence[PathNode]) -> dict[str, list[int]]:
    writers: defaultdict[str, list[int]] = defaultdict(list)
    for node_index, node in enumerate(nodes):
        for name in node.output:
            if name:
                writers[name].append(node_index)
    return dict(writers)


# The walks keep sets of the nodes they have visited rather than recurse: the graph comes from a
# container, and an encoder's audio path chains over a thousand nodes, beyond Python's recursion
# limit.
def list_descendants(
    nodes: Sequence[PathNode],
    readers: Mapping[str, Sequence[int]],
    source_names: Collection[str],
    ends_walk: Callable[[PathNode], bool] = ends_no_walk,
) -> set[int]:
    reached: set[int] = set()
    pending = [name for name in source_names if name]
    while pending:
        for node_index in readers.get(pending.pop(), ()):
            if node_index not in reached:
                reached.add(node_index)
                if not ends_walk(nodes[node_index]):
                    pending.extend(name for name in nodes[node_index].output if name)
    return reached


# A graph from a container may list its nodes out of order, or loop: the nodes that no order places
# come last, and the accumulation over them repeats until it changes nothing.
def order_by_value_reads(
    value_reads: Sequence[frozenset[str]], writers: Mapping[str, Sequence[int]]
) -> tuple[int, ...]:
    dependents: defaultdict[int, list[int]] = defaultdict(list)
    pending_counts = [0] * len(value_reads)
    for node_index, node_reads in enumerate(value_reads):
        for name in node_reads:
            for writer_index in writers.get(name, ()):
                dependents[writer_index].append(node_index)
                pending_counts[node_index] += 1
    ready = [node_index for node_index, count in enumerate(pending_counts) if not count]
    order: list[int] = []
    while ready:
        node_index = ready.pop()
        order.append(node_index)
        for dependent_index in dependents[node_index]:
            pending_counts[dependent_index] -= 1
            if not pending_counts[dependent_index]:
                ready.append(dependent_index)
    placed = set(order)
    return (*order, *(index for index in range(len(value_reads)) if index not in placed))


# The inspection asks for the computation of every cast of a graph, hundreds of them over about
# 2 000 nodes, so the reads, the writers and the descendants of the sources are indexed once, and
# what each tensor's computation holds is gathered in one pass in the order of the values.
def index_graph_paths(nodes: Sequence[PathNode], source_names: Collection[str]) -> GraphPaths:
    reads = [list_node_reads(node) for node in nodes]
    value_reads = tuple(
        frozenset() if reads_shape_alone(node) else node_reads
        for node, node_reads in zip(nodes, reads, strict=True)
    )
    return GraphPaths(
        value_reads=value_reads,
        written_names=tuple(tuple(name for name in node.output if name) for node in nodes),
        descendants=frozenset(list_descendants(nodes, index_readers(reads), source_names)),
        value_order=order_by_value_reads(value_reads, index_writers(nodes)),
    )


def accumulate_path_labels[Label](
    paths: GraphPaths, node_labels: Sequence[frozenset[Label]]
) -> dict[str, frozenset[Label]]:
    accumulated: dict[str, frozenset[Label]] = {}
    is_growing = True
    while is_growing:
        is_growing = False
        for node_index in paths.value_order:
            own_labels = node_labels[node_index] if node_index in paths.descendants else frozenset()
            gathered = own_labels.union(
                *(accumulated.get(name, frozenset()) for name in paths.value_reads[node_index])
            )
            for name in paths.written_names[node_index]:
                known = accumulated.get(name, frozenset())
                if not gathered <= known:
                    accumulated[name] = known | gathered
                    is_growing = True
    return accumulated


def index_graph_readers(nodes: Sequence[PathNode]) -> dict[str, list[int]]:
    return index_readers([list_node_reads(node) for node in nodes])


# Past a cast to an integer type, no float carries the computation, so none of it needs float32.
def list_computation_to_integer_casts(
    nodes: Sequence[PathNode], readers: Mapping[str, Sequence[int]], source_names: Collection[str]
) -> tuple[int, ...]:
    return tuple(sorted(list_descendants(nodes, readers, source_names, casts_to_integer)))
