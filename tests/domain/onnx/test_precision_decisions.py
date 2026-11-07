import re
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from functools import partial

import pytest

from mostt_model_builder.domain.onnx.conversion_request import ConversionKind
from mostt_model_builder.domain.onnx.element_types import ElementType
from mostt_model_builder.domain.onnx.graph_paths import GRAPH_ATTRIBUTE
from mostt_model_builder.domain.onnx.precision_decisions import (
    CORE_FLOAT32_OPERATORS_IN_BFLOAT16,
    MAX_FINITE_VALUE,
    MIN_POSITIVE_VALUE,
    KeptNodes,
    PrecisionDecisionError,
    decide_kept_nodes,
    list_kept_float32_operators,
    realign_cast_target,
    retype_for_bfloat16,
)

INT64 = ElementType.INT64
# The value of AttributeProto.AttributeType that holds an integer, as onnx/onnx.proto defines it.
INT_ATTRIBUTE = 2
KEPT_OPERATORS = {
    "float16 keeps no operator of the core": (ConversionKind.FLOAT16, (), ()),
    "float16 keeps the operators its recipe adds": (
        ConversionKind.FLOAT16,
        ("Range",),
        ("Range",),
    ),
    "bfloat16 keeps the operators that ONNX Runtime's CUDA provider lacks in bfloat16": (
        ConversionKind.BFLOAT16,
        (),
        ("Pad", "Tile", "Floor"),
    ),
    "bfloat16 keeps the core's operators, then those its recipe adds": (
        ConversionKind.BFLOAT16,
        ("Range",),
        ("Pad", "Tile", "Floor", "Range"),
    ),
    "bfloat16 keeps an operator that the recipe adds again only once": (
        ConversionKind.BFLOAT16,
        ("Floor", "Range"),
        ("Pad", "Tile", "Floor", "Range"),
    ),
}
CAST_TARGETS = {
    "a float32 target whose output is declared float16 targets float16": (
        ElementType.FLOAT,
        ElementType.FLOAT16,
        ElementType.FLOAT16,
    ),
    "a float32 target whose output is declared float32 stays": (
        ElementType.FLOAT,
        ElementType.FLOAT,
        ElementType.FLOAT,
    ),
    "a float32 target whose output has no declared type stays": (
        ElementType.FLOAT,
        None,
        ElementType.FLOAT,
    ),
    "an integer target stays whatever its output declares": (INT64, ElementType.FLOAT16, INT64),
    "a float16 target stays": (ElementType.FLOAT16, ElementType.FLOAT16, ElementType.FLOAT16),
}
RETYPED_ELEMENTS = {
    "float16 becomes bfloat16": (ElementType.FLOAT16, ElementType.BFLOAT16),
    "float32, the type of the inputs and outputs, stays": (ElementType.FLOAT, ElementType.FLOAT),
    "an integer stays": (INT64, INT64),
    "bfloat16 stays": (ElementType.BFLOAT16, ElementType.BFLOAT16),
}


@dataclass(frozen=True, slots=True)
class FakeTensor:
    name: str
    data_type: int = ElementType.FLOAT


@dataclass(frozen=True, slots=True)
class FakeTensorType:
    elem_type: int


@dataclass(frozen=True, slots=True)
class FakeType:
    tensor_type: FakeTensorType


@dataclass(frozen=True, slots=True)
class FakeValueInfo:
    name: str
    type: FakeType


@dataclass(frozen=True, slots=True)
class FakeAttribute:
    type: int
    name: str = ""
    i: int = 0
    g: "FakeGraph" = field(default_factory=lambda: FakeGraph())
    graphs: tuple["FakeGraph", ...] = ()


@dataclass(frozen=True, slots=True)
class FakeNode:
    name: str
    input: tuple[str, ...] = ()
    output: tuple[str, ...] = ()
    attribute: tuple[FakeAttribute, ...] = ()
    op_type: str = "Add"
    domain: str = ""


@dataclass(frozen=True, slots=True)
class FakeGraph:
    node: tuple[FakeNode, ...] = ()
    input: tuple[FakeValueInfo, ...] = ()
    output: tuple[FakeValueInfo, ...] = ()
    initializer: tuple[FakeTensor, ...] = ()
    value_info: tuple[FakeValueInfo, ...] = ()


def declare(name: str, element_type: ElementType = ElementType.FLOAT) -> FakeValueInfo:
    return FakeValueInfo(name, FakeType(FakeTensorType(element_type)))


def build_branch(*nodes: FakeNode) -> FakeAttribute:
    return FakeAttribute(GRAPH_ATTRIBUTE, g=FakeGraph(node=nodes))


def build_cast(name: str, source: str, target: str, element_type: ElementType) -> FakeNode:
    return FakeNode(
        name,
        (source,),
        (target,),
        (FakeAttribute(INT_ATTRIBUTE, name="to", i=element_type),),
        op_type="Cast",
    )


def build_round(first_index: int, source: str, rounded: str) -> tuple[FakeNode, ...]:
    padded, padded_twice, shortened, divided, floored = (
        f"{operator}_{first_index + offset}"
        for offset, operator in enumerate(("add", "add", "sub", "div", "floor"))
    )
    return (
        FakeNode(f"node_{padded}", (source, "padding"), (padded,)),
        FakeNode(f"node_{padded_twice}", (padded, "padding"), (padded_twice,)),
        FakeNode(f"node_{shortened}", (padded_twice, "kernel_size"), (shortened,), op_type="Sub"),
        FakeNode(f"node_{divided}", (shortened, "stride"), (divided,), op_type="Div"),
        FakeNode(f"node_{floored}", (divided,), (floored,), op_type="Floor"),
        FakeNode(f"node_{rounded}", (floored, "one"), (rounded,)),
    )


LENGTH_INPUT = "length"
# The float32 export's encoder, cut down: an audio path into its outputs, and the computation of
# its lengths, three rounds of the subsampling from the cast of length to float32 to the cast of
# encoded_lengths to int64, 20 nodes, whose intermediate lengths three more casts to int64 read, the
# second and the third from values that the next round reads too, add_9 and add_18; a mask reads
# such a cast beside the audio.
FIRST_ROUND = build_round(4, "_to_copy", "add_9")
SECOND_ROUND = build_round(16, "add_9", "add_18")
THIRD_ROUND = build_round(25, "add_18", "add_31")
LENGTH_CAST = build_cast("node__to_copy", LENGTH_INPUT, "_to_copy", ElementType.FLOAT)
INTERMEDIATE_CASTS = (
    build_cast("node__to_copy_3", "floor_8", "_to_copy_3", ElementType.INT64),
    build_cast("node__to_copy_5", "add_9", "_to_copy_5", ElementType.INT64),
    build_cast("node__to_copy_7", "add_18", "_to_copy_7", ElementType.INT64),
)
OUTPUT_LENGTHS_CAST = build_cast("node__to_copy_8", "add_31", "encoded_lengths", ElementType.INT64)
EXPORTED_NODES = (
    FakeNode("node_relu", ("audio_signal",), ("features",), op_type="Relu"),
    FakeNode("node_mul", ("features", "two"), ("scaled",), op_type="Mul"),
    FakeNode("node_matmul", ("scaled", "weight"), ("outputs",), op_type="MatMul"),
    FakeNode(
        "node_audio_branch",
        ("features",),
        ("branched",),
        (build_branch(FakeNode("branch_mul", ("features", "half"), ("branch_scaled",))),),
        op_type="If",
    ),
    FakeNode("node_padding", (), ("padding",), op_type="Constant"),
    LENGTH_CAST,
    *FIRST_ROUND,
    INTERMEDIATE_CASTS[0],
    INTERMEDIATE_CASTS[1],
    *SECOND_ROUND,
    INTERMEDIATE_CASTS[2],
    *THIRD_ROUND,
    OUTPUT_LENGTHS_CAST,
    FakeNode("node_mask", ("_to_copy_5", "features"), ("mask",), op_type="Less"),
)
EXPORTED_GRAPH = FakeGraph(
    node=EXPORTED_NODES,
    input=(declare("audio_signal"), declare(LENGTH_INPUT, ElementType.INT64)),
    output=(declare("outputs"), declare("encoded_lengths", ElementType.INT64)),
    initializer=tuple(
        FakeTensor(name) for name in ("weight", "two", "half", "kernel_size", "stride", "one")
    ),
)
LENGTH_COMPUTATION = (
    LENGTH_CAST,
    *FIRST_ROUND,
    *INTERMEDIATE_CASTS[:2],
    *SECOND_ROUND,
    INTERMEDIATE_CASTS[2],
    *THIRD_ROUND,
    OUTPUT_LENGTHS_CAST,
)


def replace_node(graph: FakeGraph, old_name: str, new_node: FakeNode) -> FakeGraph:
    return replace(
        graph, node=tuple(new_node if node.name == old_name else node for node in graph.node)
    )


KEPT_COMPUTATION_REFUSALS = {
    "an input that is not an input of the graph": (
        EXPORTED_GRAPH,
        "features",
        "the computation kept in float32 from 'features' starts from no input of the graph",
    ),
    "an input that no node reads": (
        replace(
            EXPORTED_GRAPH, input=(*EXPORTED_GRAPH.input, declare("unused", ElementType.INT64))
        ),
        "unused",
        "the computation kept in float32 from 'unused' holds no node",
    ),
    "a node of the computation that reads the computation of another input too": (
        replace_node(
            EXPORTED_GRAPH,
            "node_add_5",
            FakeNode("node_add_5", ("add_4", "features"), ("add_5",)),
        ),
        LENGTH_INPUT,
        "the computation kept in float32 from 'length' holds the node 'node_add_5', which also "
        "reads another input of the graph or its computation",
    ),
    "a node of the computation that reads another input itself": (
        replace_node(
            EXPORTED_GRAPH,
            "node_add_5",
            FakeNode("node_add_5", ("add_4", "audio_signal"), ("add_5",)),
        ),
        LENGTH_INPUT,
        "the computation kept in float32 from 'length' holds the node 'node_add_5', which also "
        "reads another input of the graph or its computation",
    ),
    "a node of the computation that holds a subgraph": (
        replace_node(
            EXPORTED_GRAPH,
            "node_floor_8",
            FakeNode(
                "node_floor_8",
                ("div_7",),
                ("floor_8",),
                (build_branch(FakeNode("inner_floor", ("div_7",), ("inner_floored",))),),
                op_type="If",
            ),
        ),
        LENGTH_INPUT,
        "the computation kept in float32 from 'length' holds the node 'node_floor_8', which holds "
        "a subgraph",
    ),
    "a node of the computation without a name": (
        replace_node(
            EXPORTED_GRAPH,
            "node_div_7",
            FakeNode("", ("sub_6",), ("div_7",), op_type="Div"),
        ),
        LENGTH_INPUT,
        "the computation kept in float32 from 'length' holds a node without a name",
    ),
    "a node of the computation whose name a node of the main graph bears": (
        replace_node(
            EXPORTED_GRAPH,
            "node_relu",
            FakeNode("node_div_7", ("audio_signal",), ("features",), op_type="Relu"),
        ),
        LENGTH_INPUT,
        "the computation kept in float32 from 'length' holds the node 'node_div_7', whose name "
        "another node of the model bears",
    ),
    "a node of the computation whose name a node of a subgraph bears": (
        replace_node(
            EXPORTED_GRAPH,
            "node_audio_branch",
            FakeNode(
                "node_audio_branch",
                ("features",),
                ("branched",),
                (build_branch(FakeNode("node_div_7", ("features",), ("branch_divided",))),),
                op_type="If",
            ),
        ),
        LENGTH_INPUT,
        "the computation kept in float32 from 'length' holds the node 'node_div_7', whose name "
        "another node of the model bears",
    ),
    "a float tensor of the computation that is an output of the graph": (
        replace(EXPORTED_GRAPH, output=(*EXPORTED_GRAPH.output, declare("floor_8"))),
        LENGTH_INPUT,
        "the computation kept in float32 from 'length' writes 'floor_8', a float output of the "
        "graph, which onnxconverter-common would cast to float16",
    ),
}

SECOND_LENGTH_INPUT = "audio_lengths"
SECOND_LENGTH_COMPUTATION = (
    build_cast(
        "node_audio_lengths_float", SECOND_LENGTH_INPUT, "audio_lengths_float", ElementType.FLOAT
    ),
    FakeNode("node_audio_lengths_add", ("audio_lengths_float", "one"), ("audio_lengths_added",)),
    build_cast(
        "node_audio_lengths_int", "audio_lengths_added", "audio_lengths_int", ElementType.INT64
    ),
)
TWO_LENGTHS_GRAPH = replace(
    EXPORTED_GRAPH,
    node=(*EXPORTED_GRAPH.node, *SECOND_LENGTH_COMPUTATION),
    input=(*EXPORTED_GRAPH.input, declare(SECOND_LENGTH_INPUT, ElementType.INT64)),
)
LENGTHS_THAT_SHARE_A_NODE = replace_node(
    TWO_LENGTHS_GRAPH,
    "node_audio_lengths_add",
    FakeNode("node_audio_lengths_add", ("audio_lengths_float", "add_9"), ("audio_lengths_added",)),
)


class TestThePrecisionDecisions:
    class TestTheListKeptFloat32OperatorsFunction:
        @pytest.mark.parametrize(
            ("kind", "added_operators", "expected_operators"),
            list(KEPT_OPERATORS.values()),
            ids=list(KEPT_OPERATORS),
        )
        class TestWhenAConversionKeepsOperatorsInFloat32:
            @pytest.fixture
            def kept_operators(
                self, kind: ConversionKind, added_operators: tuple[str, ...]
            ) -> tuple[str, ...]:
                return list_kept_float32_operators(kind, added_operators)

            def test_should_keep_the_cores_operators_for_its_kind_and_the_recipes(
                self, kept_operators: tuple[str, ...], expected_operators: tuple[str, ...]
            ) -> None:
                assert kept_operators == expected_operators

    class TestTheDecideKeptNodesFunction:
        class TestWhenTheRecipeKeepsTheComputationFromTheLengths:
            @pytest.fixture
            def kept_nodes(self) -> KeptNodes:
                return decide_kept_nodes(EXPORTED_GRAPH, (LENGTH_INPUT,))

            def test_should_keep_its_23_nodes_until_its_casts_to_integers_and_none_of_the_audio(
                self, kept_nodes: KeptNodes
            ) -> None:
                assert kept_nodes.node_names == tuple(node.name for node in LENGTH_COMPUTATION)

            def test_should_set_aside_every_tensor_it_writes_add_9_and_add_18_included(
                self, kept_nodes: KeptNodes
            ) -> None:
                assert kept_nodes.set_aside_tensor_names == tuple(
                    name for node in LENGTH_COMPUTATION for name in node.output
                )

            def test_should_keep_no_constant_since_constants_are_its_parameters(
                self, kept_nodes: KeptNodes
            ) -> None:
                assert "node_padding" not in kept_nodes.node_names

        class TestWhenTheRecipeKeepsTheComputationsOfTwoInputs:
            @pytest.fixture
            def kept_nodes(self) -> KeptNodes:
                return decide_kept_nodes(TWO_LENGTHS_GRAPH, (LENGTH_INPUT, SECOND_LENGTH_INPUT))

            def test_should_keep_each_computation_until_its_own_casts_to_integers(
                self, kept_nodes: KeptNodes
            ) -> None:
                assert kept_nodes.node_names == tuple(
                    node.name for node in (*LENGTH_COMPUTATION, *SECOND_LENGTH_COMPUTATION)
                )

        class TestWhenANodeReadsTheComputationsOfTwoKeptInputs:
            @pytest.fixture
            def decide_act(self) -> Callable[[], KeptNodes]:
                return partial(
                    decide_kept_nodes,
                    LENGTHS_THAT_SHARE_A_NODE,
                    (LENGTH_INPUT, SECOND_LENGTH_INPUT),
                )

            def test_should_refuse_it_since_each_declared_input_is_a_computation_of_its_own(
                self, decide_act: Callable[[], KeptNodes]
            ) -> None:
                with pytest.raises(
                    PrecisionDecisionError,
                    match=re.escape(
                        "the computation kept in float32 from 'length' holds the node "
                        "'node_audio_lengths_add', which also reads another input of the graph or "
                        "its computation"
                    ),
                ):
                    decide_act()

        class TestWhenTheRecipeKeepsNoComputation:
            @pytest.fixture
            def kept_nodes(self) -> KeptNodes:
                return decide_kept_nodes(EXPORTED_GRAPH, ())

            def test_should_keep_no_node_and_set_nothing_aside(self, kept_nodes: KeptNodes) -> None:
                assert kept_nodes == KeptNodes((), ())

        @pytest.mark.parametrize(
            ("graph", "kept_input", "reason"),
            list(KEPT_COMPUTATION_REFUSALS.values()),
            ids=list(KEPT_COMPUTATION_REFUSALS),
        )
        class TestWhenTheComputationCannotBeKept:
            @pytest.fixture
            def decide_act(self, graph: FakeGraph, kept_input: str) -> Callable[[], KeptNodes]:
                return partial(decide_kept_nodes, graph, (kept_input,))

            def test_should_refuse_it_before_converting_and_say_why(
                self, decide_act: Callable[[], KeptNodes], reason: str
            ) -> None:
                with pytest.raises(PrecisionDecisionError, match=re.escape(reason)):
                    decide_act()

    class TestTheRealignCastTargetFunction:
        @pytest.mark.parametrize(
            ("cast_target", "output_element_type", "expected_target"),
            list(CAST_TARGETS.values()),
            ids=list(CAST_TARGETS),
        )
        class TestWhenACastIsRead:
            @pytest.fixture
            def realigned_target(self, cast_target: int, output_element_type: int | None) -> int:
                return realign_cast_target(cast_target, output_element_type)

            def test_should_target_the_type_its_output_declares_when_float16(
                self, realigned_target: int, expected_target: int
            ) -> None:
                assert realigned_target == expected_target

    class TestTheRetypeForBfloat16Function:
        @pytest.mark.parametrize(
            ("element_type", "expected_type"),
            list(RETYPED_ELEMENTS.values()),
            ids=list(RETYPED_ELEMENTS),
        )
        class TestWhenAnElementTypeIsRetyped:
            @pytest.fixture
            def retyped_element(self, element_type: int) -> int:
                return retype_for_bfloat16(element_type)

            def test_should_turn_float16_alone_into_bfloat16(
                self, retyped_element: int, expected_type: int
            ) -> None:
                assert retyped_element == expected_type

    class TestTheCoreFloat32OperatorsInBfloat16Constant:
        def test_should_hold_the_operators_without_a_bfloat16_kernel_on_cuda(self) -> None:
            assert CORE_FLOAT32_OPERATORS_IN_BFLOAT16 == ("Pad", "Tile", "Floor")

    class TestTheElementTypeConstant:
        def test_should_number_the_types_as_onnx_does(self) -> None:
            assert (ElementType.FLOAT, ElementType.FLOAT16, ElementType.BFLOAT16) == (1, 10, 16)

    class TestTheClampBoundConstants:
        def test_should_be_the_defaults_of_onnxconverter_common(self) -> None:
            assert (MIN_POSITIVE_VALUE, MAX_FINITE_VALUE) == (1e-7, 1e4)
