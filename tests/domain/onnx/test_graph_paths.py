import random
from collections.abc import Sequence
from dataclasses import dataclass, field

import pytest

from mostt_model_builder.domain.onnx.element_types import ElementType
from mostt_model_builder.domain.onnx.graph_paths import (
    GRAPH_ATTRIBUTE,
    GRAPHS_ATTRIBUTE,
    accumulate_path_labels,
    index_graph_paths,
    index_graph_readers,
    list_computation_to_integer_casts,
)

PROPERTY_SEED = 20261107
PROPERTY_CASE_COUNT = 500
MAX_NODE_COUNT = 7
GRAPH_INPUTS = ("input_a", "input_b", "input_c")
OMITTED = ""
SHAPE_READERS = ("Shape", "Size")
# The value of AttributeProto.AttributeType that holds an integer, as onnx/onnx.proto defines it.
INT_ATTRIBUTE = 2
CAST_TARGETS = (ElementType.INT64, ElementType.INT32, ElementType.BOOL, ElementType.FLOAT)
INTEGER_CAST_TARGETS = (ElementType.INT64, ElementType.INT32)


@dataclass(frozen=True, slots=True)
class FakeTensor:
    name: str


@dataclass(frozen=True, slots=True)
class FakeAttribute:
    type: int
    name: str = ""
    i: int = 0
    g: "FakeGraph" = field(default_factory=lambda: FakeGraph())
    graphs: tuple["FakeGraph", ...] = ()


@dataclass(frozen=True, slots=True)
class FakeNode:
    input: tuple[str, ...] = ()
    output: tuple[str, ...] = ()
    attribute: tuple[FakeAttribute, ...] = ()
    op_type: str = "Add"
    domain: str = ""


@dataclass(frozen=True, slots=True)
class FakeGraph:
    node: tuple[FakeNode, ...] = ()
    input: tuple[FakeTensor, ...] = ()
    initializer: tuple[FakeTensor, ...] = ()


@dataclass(frozen=True, slots=True)
class PathCase:
    nodes: tuple[FakeNode, ...]
    source_names: tuple[str, ...]
    target_name: str


@dataclass(frozen=True, slots=True)
class ComputationCase:
    nodes: tuple[FakeNode, ...]
    source_names: tuple[str, ...]


def build_cast(source: str, target: str, element_type: ElementType) -> FakeNode:
    return FakeNode(
        (source,),
        (target,),
        (FakeAttribute(INT_ATTRIBUTE, name="to", i=element_type),),
        op_type="Cast",
    )


def build_if(*branch_nodes: FakeNode, output: str) -> FakeNode:
    return FakeNode(
        input=("condition",),
        output=(output,),
        attribute=(
            FakeAttribute(GRAPH_ATTRIBUTE, g=FakeGraph(node=branch_nodes)),
            FakeAttribute(GRAPH_ATTRIBUTE, g=FakeGraph()),
        ),
    )


def build_loop_over(*body_nodes: FakeNode) -> FakeNode:
    return FakeNode(
        output=("loop_result",),
        attribute=(FakeAttribute(GRAPHS_ATTRIBUTE, graphs=(FakeGraph(node=body_nodes),)),),
    )


PATH_CASES = {
    "a linear chain keeps every node": (
        PathCase(
            (
                FakeNode(("input_a",), ("hidden",)),
                FakeNode(("hidden",), ("target",)),
            ),
            ("input_a",),
            "target",
        ),
        (0, 1),
    ),
    "a branch that leaves the path and never reaches the target stays out": (
        PathCase(
            (
                FakeNode(("input_a",), ("hidden",)),
                FakeNode(("hidden",), ("target",)),
                FakeNode(("hidden",), ("side",)),
            ),
            ("input_a",),
            "target",
        ),
        (0, 1),
    ),
    "a constant subtree that feeds a node of the path stays out": (
        PathCase(
            (
                FakeNode((), ("constant",)),
                FakeNode(("constant",), ("scaled_constant",)),
                FakeNode(("input_a", "scaled_constant"), ("target",)),
            ),
            ("input_a",),
            "target",
        ),
        (2,),
    ),
    "a fan-out that rejoins keeps both branches": (
        PathCase(
            (
                FakeNode(("input_a",), ("hidden",)),
                FakeNode(("hidden",), ("left",)),
                FakeNode(("hidden",), ("right",)),
                FakeNode(("left", "right"), ("target",)),
            ),
            ("input_a",),
            "target",
        ),
        (0, 1, 2, 3),
    ),
    "a node of two outputs is kept for the one that leads to the target": (
        PathCase(
            (
                FakeNode(("input_a",), ("kept", "dropped")),
                FakeNode(("dropped",), ("side",)),
                FakeNode(("kept",), ("target",)),
            ),
            ("input_a",),
            "target",
        ),
        (0, 2),
    ),
    "a target that the input never reaches has no path": (
        PathCase((FakeNode(("input_b",), ("target",)),), ("input_a",), "target"),
        (),
    ),
    "several inputs each start a path": (
        PathCase(
            (
                FakeNode(("input_a",), ("from_a",)),
                FakeNode(("input_b",), ("from_b",)),
                FakeNode(("input_c",), ("from_c",)),
                FakeNode(("from_a", "from_b"), ("target",)),
                FakeNode(("from_c", "target"), ("beyond",)),
            ),
            ("input_a", "input_b"),
            "target",
        ),
        (0, 1, 3),
    ),
    "two nodes that share only an omitted optional tensor make no path": (
        PathCase(
            (
                FakeNode(("input_a",), ("unused", OMITTED)),
                FakeNode((OMITTED,), ("target",)),
            ),
            ("input_a",),
            "target",
        ),
        (),
    ),
    "an If whose nested branch reads a tensor of the path without naming it is on the path": (
        PathCase(
            (
                FakeNode(("input_a",), ("hidden",)),
                build_if(build_loop_over(FakeNode(("hidden",), ("inner",))), output="target"),
            ),
            ("input_a",),
            "target",
        ),
        (0, 1),
    ),
    "a Shape keeps none of its data's computation, since it carries no value of it": (
        PathCase(
            (
                FakeNode(("input_a",), ("hidden",)),
                FakeNode(("hidden",), ("shape",), op_type="Shape"),
                FakeNode(("shape",), ("target",)),
            ),
            ("input_a",),
            "target",
        ),
        (1, 2),
    ),
    "a Size cuts its own read alone, never another reader's of the same data": (
        PathCase(
            (
                FakeNode(("input_a",), ("hidden",)),
                FakeNode(("hidden",), ("size",), op_type="Size"),
                FakeNode(("size", "hidden"), ("target",)),
            ),
            ("input_a",),
            "target",
        ),
        (0, 1, 2),
    ),
    "a Shape of a domain other than ONNX's is an ordinary node": (
        PathCase(
            (
                FakeNode(("input_a",), ("hidden",)),
                FakeNode(("hidden",), ("shape",), op_type="Shape", domain="mostt.local"),
                FakeNode(("shape",), ("target",)),
            ),
            ("input_a",),
            "target",
        ),
        (0, 1, 2),
    ),
    "nodes listed out of the order of their values keep the same path": (
        PathCase(
            (
                FakeNode(("hidden",), ("target",)),
                FakeNode(("input_a",), ("hidden",)),
            ),
            ("input_a",),
            "target",
        ),
        (0, 1),
    ),
    "a loop of nodes, which no order places, keeps every node of the loop on the path": (
        PathCase(
            (
                FakeNode(("input_a", "looped"), ("hidden",)),
                FakeNode(("hidden",), ("looped",)),
                FakeNode(("hidden",), ("target",)),
            ),
            ("input_a",),
            "target",
        ),
        (0, 1, 2),
    ),
    "an If whose branch defines the name it reads stays out": (
        PathCase(
            (
                FakeNode(("input_a",), ("hidden",)),
                build_if(
                    FakeNode((), ("hidden",)),
                    FakeNode(("hidden",), ("inner",)),
                    output="target",
                ),
            ),
            ("input_a",),
            "target",
        ),
        (),
    ),
}

COMPUTATION_CASES = {
    "a chain keeps its cast to an integer type and none of that cast's readers": (
        ComputationCase(
            (
                build_cast("input_a", "as_float", ElementType.FLOAT),
                FakeNode(("as_float",), ("hidden",)),
                build_cast("hidden", "as_integer", ElementType.INT64),
                FakeNode(("as_integer",), ("mask",)),
            ),
            ("input_a",),
        ),
        (0, 1, 2),
    ),
    "a value that two kept nodes read keeps both": (
        ComputationCase(
            (
                FakeNode(("input_a",), ("hidden",)),
                FakeNode(("hidden",), ("next_round",)),
                build_cast("hidden", "as_integer", ElementType.INT64),
            ),
            ("input_a",),
        ),
        (0, 1, 2),
    ),
    "a cast to a boolean is no cast to an integer, and the walk goes past it": (
        ComputationCase(
            (
                build_cast("input_a", "as_boolean", ElementType.BOOL),
                FakeNode(("as_boolean",), ("selected",)),
            ),
            ("input_a",),
        ),
        (0, 1),
    ),
    "a node that reads another input's computation too is kept, for the conversion to refuse": (
        ComputationCase(
            (
                FakeNode(("input_b",), ("from_b",)),
                FakeNode(("input_a", "from_b"), ("mixed",)),
            ),
            ("input_a",),
        ),
        (1,),
    ),
    "the empty name joins no nodes": (
        ComputationCase(
            (
                FakeNode(("input_a",), ("unused", OMITTED)),
                FakeNode((OMITTED,), ("unrelated",)),
            ),
            ("input_a",),
        ),
        (0,),
    ),
    "an If whose branch reads a tensor of the computation is kept": (
        ComputationCase(
            (
                FakeNode(("input_a",), ("hidden",)),
                build_if(FakeNode(("hidden",), ("inner",)), output="chosen"),
            ),
            ("input_a",),
        ),
        (0, 1),
    ),
    "a cast to an integer of a domain other than ONNX's is an ordinary node": (
        ComputationCase(
            (
                FakeNode(
                    ("input_a",),
                    ("as_integer",),
                    (FakeAttribute(INT_ATTRIBUTE, name="to", i=ElementType.INT64),),
                    op_type="Cast",
                    domain="mostt.local",
                ),
                FakeNode(("as_integer",), ("mask",)),
            ),
            ("input_a",),
        ),
        (0, 1),
    ),
}


def generate_case(generator: random.Random) -> PathCase:
    tensors = list(GRAPH_INPUTS)
    nodes = []
    for node_index in range(generator.randint(1, MAX_NODE_COUNT)):
        if generator.random() < 0.2:
            shape_reader = generator.choice(SHAPE_READERS)
            nodes.append(
                FakeNode(
                    (generator.choice(tensors),), (f"tensor_{node_index}",), op_type=shape_reader
                )
            )
            tensors.append(f"tensor_{node_index}")
            continue
        reads = tuple(generator.choice([*tensors, OMITTED]) for _ in range(generator.randint(0, 3)))
        writes = tuple(
            OMITTED if generator.random() < 0.2 else f"tensor_{node_index}_{write_index}"
            for write_index in range(generator.randint(1, 2))
        )
        branch_reads = tuple(generator.choice(tensors) for _ in range(generator.randint(0, 1)))
        attributes = (
            (
                FakeAttribute(
                    GRAPH_ATTRIBUTE,
                    g=FakeGraph(node=(FakeNode(branch_reads, (f"branch_{node_index}",)),)),
                ),
            )
            if branch_reads
            else ()
        )
        nodes.append(FakeNode(reads, writes, attributes))
        tensors.extend(name for name in writes if name)
    generator.shuffle(nodes)
    source_names = tuple(name for name in GRAPH_INPUTS if generator.random() < 0.5)
    return PathCase(tuple(nodes), source_names, generator.choice(tensors))


def list_path_nodes(case: PathCase) -> tuple[int, ...]:
    paths = index_graph_paths(case.nodes, case.source_names)
    node_labels = [frozenset({node_index}) for node_index in range(len(case.nodes))]
    return tuple(sorted(accumulate_path_labels(paths, node_labels).get(case.target_name, ())))


def list_reference_reads(node: FakeNode) -> set[str]:
    branch_reads = {
        name
        for attribute in node.attribute
        for branch_node in attribute.g.node
        for name in branch_node.input
    }
    return {name for name in (*node.input, *branch_reads) if name}


def keep_from_the_last_shape_reader(case: PathCase, trail: Sequence[int]) -> Sequence[int]:
    shape_positions = [
        position
        for position, node_index in enumerate(trail)
        if case.nodes[node_index].op_type in SHAPE_READERS
    ]
    return trail[max(shape_positions, default=0) :]


# Shares no code with the walk, so that one defect cannot hide in both.
def enumerate_path_nodes(case: PathCase) -> tuple[int, ...]:
    found: set[int] = set()

    def follow(name: str, trail: Sequence[int]) -> None:
        if name == case.target_name:
            found.update(keep_from_the_last_shape_reader(case, trail))
        for node_index, node in enumerate(case.nodes):
            if name in list_reference_reads(node):
                for output in node.output:
                    if output:
                        follow(output, (*trail, node_index))

    for source_name in case.source_names:
        follow(source_name, ())
    return tuple(sorted(found))


def generate_computation_case(generator: random.Random) -> ComputationCase:
    tensors = list(GRAPH_INPUTS)
    nodes = []
    for node_index in range(generator.randint(1, MAX_NODE_COUNT)):
        if generator.random() < 0.3:
            nodes.append(
                build_cast(
                    generator.choice(tensors),
                    f"tensor_{node_index}",
                    generator.choice(CAST_TARGETS),
                )
            )
            tensors.append(f"tensor_{node_index}")
            continue
        reads = tuple(generator.choice([*tensors, OMITTED]) for _ in range(generator.randint(0, 3)))
        writes = tuple(
            OMITTED if generator.random() < 0.2 else f"tensor_{node_index}_{write_index}"
            for write_index in range(generator.randint(1, 2))
        )
        nodes.append(FakeNode(reads, writes))
        tensors.extend(name for name in writes if name)
    source_names = tuple(name for name in GRAPH_INPUTS if generator.random() < 0.5)
    return ComputationCase(tuple(nodes), source_names)


def casts_to_an_integer(node: FakeNode) -> bool:
    return node.op_type == "Cast" and node.attribute[0].i in INTEGER_CAST_TARGETS


# Shares no code with the walk, so that one defect cannot hide in both.
def enumerate_computation_nodes(case: ComputationCase) -> tuple[int, ...]:
    found: set[int] = set()

    def follow(name: str, trail: Sequence[int]) -> None:
        found.update(trail)
        if trail and casts_to_an_integer(case.nodes[trail[-1]]):
            return
        for node_index, node in enumerate(case.nodes):
            if name and name in node.input:
                for output in node.output or (OMITTED,):
                    follow(output, (*trail, node_index))

    for source_name in case.source_names:
        follow(source_name, ())
    return tuple(sorted(found))


def list_computation_nodes(case: ComputationCase) -> tuple[int, ...]:
    return list_computation_to_integer_casts(
        case.nodes, index_graph_readers(case.nodes), case.source_names
    )


def find_computation_mismatches(generator: random.Random) -> tuple[list[ComputationCase], int]:
    cases = [generate_computation_case(generator) for _ in range(PROPERTY_CASE_COUNT)]
    return [
        case for case in cases if list_computation_nodes(case) != enumerate_computation_nodes(case)
    ], len(cases)


def find_mismatches(generator: random.Random) -> tuple[list[PathCase], int]:
    cases = [generate_case(generator) for _ in range(PROPERTY_CASE_COUNT)]
    return [case for case in cases if list_path_nodes(case) != enumerate_path_nodes(case)], len(
        cases
    )


class TestTheGraphPaths:
    class TestTheAccumulatePathLabelsFunction:
        @pytest.mark.parametrize(
            ("case", "expected_nodes"), list(PATH_CASES.values()), ids=list(PATH_CASES.keys())
        )
        class TestWhenGivenASmallGraph:
            @pytest.fixture
            def path_nodes(self, case: PathCase) -> tuple[int, ...]:
                return list_path_nodes(case)

            def test_should_gather_the_nodes_that_descend_from_a_source_and_lead_to_the_target(
                self, path_nodes: tuple[int, ...], expected_nodes: tuple[int, ...]
            ) -> None:
                assert path_nodes == expected_nodes

        class TestWhenGivenAnyDirectedAcyclicGraphInAnyOrder:
            def test_should_gather_exactly_the_nodes_of_every_path_of_tensors(self) -> None:
                mismatches, checked_count = find_mismatches(random.Random(PROPERTY_SEED))
                assert (mismatches, checked_count) == ([], PROPERTY_CASE_COUNT), (
                    f"seed {PROPERTY_SEED}"
                )

    class TestTheListComputationToIntegerCastsFunction:
        @pytest.mark.parametrize(
            ("case", "expected_nodes"),
            list(COMPUTATION_CASES.values()),
            ids=list(COMPUTATION_CASES.keys()),
        )
        class TestWhenGivenASmallGraph:
            @pytest.fixture
            def computation_nodes(self, case: ComputationCase) -> tuple[int, ...]:
                return list_computation_nodes(case)

            def test_should_keep_every_node_a_source_reaches_until_its_casts_to_integers(
                self, computation_nodes: tuple[int, ...], expected_nodes: tuple[int, ...]
            ) -> None:
                assert computation_nodes == expected_nodes

        class TestWhenGivenAnyDirectedAcyclicGraph:
            def test_should_keep_exactly_the_nodes_of_every_path_until_a_cast_to_an_integer(
                self,
            ) -> None:
                mismatches, checked_count = find_computation_mismatches(
                    random.Random(PROPERTY_SEED)
                )
                assert (mismatches, checked_count) == ([], PROPERTY_CASE_COUNT), (
                    f"seed {PROPERTY_SEED}"
                )
