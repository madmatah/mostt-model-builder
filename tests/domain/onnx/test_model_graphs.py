import itertools
from collections.abc import Callable
from dataclasses import dataclass, field

import pytest

from mostt_model_builder.domain.onnx.element_types import ElementType
from mostt_model_builder.domain.onnx.graph_paths import GRAPH_ATTRIBUTE, GRAPHS_ATTRIBUTE
from mostt_model_builder.domain.onnx.inspection_report import (
    ExternalTensor,
    MatrixProduct,
    NodeCount,
    NodeScope,
    TensorComputation,
    WeightSource,
)
from mostt_model_builder.domain.onnx.model_graphs import (
    EXTERNAL_DATA_LOCATION,
    SPARSE_TENSOR_ATTRIBUTE,
    SPARSE_TENSORS_ATTRIBUTE,
    TENSOR_ATTRIBUTE,
    TENSORS_ATTRIBUTE,
    count_nodes,
    count_nodes_with_metadata,
    count_sparse_initializers,
    count_training_graphs,
    index_main_graph,
    list_cast_computations,
    list_external_tensors,
    list_output_computations,
    list_products,
)

DEFAULT_DATA_LOCATION = 0
DATA_FILE_NAME = "encoder.onnx.data"
TENSOR_OFFSET_BYTES = 4096
TENSOR_LENGTH_BYTES = 1024
OTHER_DATA_FILE_NAME = "decoder.onnx.data"
OTHER_DATA_FILE_OFFSET_BYTES = TENSOR_OFFSET_BYTES * 2
LOCAL_DOMAIN = "mostt.local"
LOCAL_FUNCTION_NAME = "RelativePositionAttention"
WEIGHT_NAME = "weight"
SIBLING_WEIGHT_NAME = "then_weight"
# The value of AttributeProto.AttributeType that holds an integer, as onnx/onnx.proto defines it.
INT_ATTRIBUTE = 2


@dataclass(frozen=True, slots=True)
class FakeStringEntry:
    key: str
    value: str


@dataclass(frozen=True, slots=True)
class FakeTensor:
    name: str = WEIGHT_NAME
    dims: tuple[int, ...] = (4, 4)
    data_location: int = DEFAULT_DATA_LOCATION
    external_data: tuple[FakeStringEntry, ...] = ()


@dataclass(frozen=True, slots=True)
class FakeSparseTensor:
    values: FakeTensor = field(default_factory=FakeTensor)
    indices: FakeTensor = field(default_factory=FakeTensor)


@dataclass(frozen=True, slots=True)
class FakeAttribute:
    type: int
    name: str = ""
    i: int = 0
    t: FakeTensor = field(default_factory=FakeTensor)
    tensors: tuple[FakeTensor, ...] = ()
    sparse_tensor: FakeSparseTensor = field(default_factory=FakeSparseTensor)
    sparse_tensors: tuple[FakeSparseTensor, ...] = ()
    g: "FakeGraph" = field(default_factory=lambda: FakeGraph())
    graphs: tuple["FakeGraph", ...] = ()


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
class FakeNode:
    op_type: str
    domain: str = ""
    input: tuple[str, ...] = ()
    output: tuple[str, ...] = ()
    attribute: tuple[FakeAttribute, ...] = ()
    metadata_props: tuple[FakeStringEntry, ...] = ()


@dataclass(frozen=True, slots=True)
class FakeGraph:
    node: tuple[FakeNode, ...] = ()
    initializer: tuple[FakeTensor, ...] = ()
    sparse_initializer: tuple[FakeSparseTensor, ...] = ()
    input: tuple[FakeValueInfo, ...] = ()
    output: tuple[FakeValueInfo, ...] = ()
    value_info: tuple[FakeValueInfo, ...] = ()


@dataclass(frozen=True, slots=True)
class FakeFunction:
    domain: str = LOCAL_DOMAIN
    name: str = LOCAL_FUNCTION_NAME
    node: tuple[FakeNode, ...] = ()
    attribute_proto: tuple[FakeAttribute, ...] = ()


@dataclass(frozen=True, slots=True)
class FakeModel:
    graph: FakeGraph = field(default_factory=FakeGraph)
    functions: tuple[FakeFunction, ...] = ()
    training_info: tuple[object, ...] = ()


@dataclass(frozen=True, slots=True)
class TensorPlacement:
    initializers: tuple[FakeTensor, ...] = ()
    sparse_initializers: tuple[FakeSparseTensor, ...] = ()
    node_attributes: tuple[FakeAttribute, ...] = ()
    default_attributes: tuple[FakeAttribute, ...] = ()


EXTERNAL_TENSOR = FakeTensor(
    data_location=EXTERNAL_DATA_LOCATION,
    external_data=(
        FakeStringEntry("location", DATA_FILE_NAME),
        FakeStringEntry("offset", str(TENSOR_OFFSET_BYTES)),
        FakeStringEntry("length", str(TENSOR_LENGTH_BYTES)),
    ),
)
INTERNAL_TENSOR = FakeTensor()
LISTED_TENSOR = ExternalTensor(DATA_FILE_NAME, TENSOR_OFFSET_BYTES, TENSOR_LENGTH_BYTES)
CALL_TO_THE_LOCAL_FUNCTION = FakeNode(LOCAL_FUNCTION_NAME, LOCAL_DOMAIN)


def place_in_initializer(tensor: FakeTensor) -> TensorPlacement:
    return TensorPlacement(initializers=(tensor,))


def place_in_sparse_values(tensor: FakeTensor) -> TensorPlacement:
    return TensorPlacement(sparse_initializers=(FakeSparseTensor(values=tensor),))


def place_in_sparse_indices(tensor: FakeTensor) -> TensorPlacement:
    return TensorPlacement(sparse_initializers=(FakeSparseTensor(indices=tensor),))


def place_in_tensor_attribute(tensor: FakeTensor) -> TensorPlacement:
    return TensorPlacement(node_attributes=(FakeAttribute(TENSOR_ATTRIBUTE, t=tensor),))


def place_in_tensors_attribute(tensor: FakeTensor) -> TensorPlacement:
    return TensorPlacement(
        node_attributes=(FakeAttribute(TENSORS_ATTRIBUTE, tensors=(INTERNAL_TENSOR, tensor)),)
    )


def place_in_sparse_tensor_attribute(tensor: FakeTensor) -> TensorPlacement:
    return TensorPlacement(
        node_attributes=(
            FakeAttribute(SPARSE_TENSOR_ATTRIBUTE, sparse_tensor=FakeSparseTensor(values=tensor)),
        )
    )


def place_in_sparse_tensors_attribute(tensor: FakeTensor) -> TensorPlacement:
    return TensorPlacement(
        node_attributes=(
            FakeAttribute(
                SPARSE_TENSORS_ATTRIBUTE,
                sparse_tensors=(FakeSparseTensor(), FakeSparseTensor(indices=tensor)),
            ),
        )
    )


def place_in_default_attribute(tensor: FakeTensor) -> TensorPlacement:
    return TensorPlacement(default_attributes=(FakeAttribute(TENSOR_ATTRIBUTE, t=tensor),))


def build_graph(placement: TensorPlacement) -> FakeGraph:
    return FakeGraph(
        node=(FakeNode("Constant", attribute=placement.node_attributes),),
        initializer=placement.initializers,
        sparse_initializer=placement.sparse_initializers,
    )


def build_if(then_branch: FakeGraph, else_branch: FakeGraph) -> FakeNode:
    return FakeNode(
        "If",
        attribute=(
            FakeAttribute(GRAPH_ATTRIBUTE, g=then_branch),
            FakeAttribute(GRAPH_ATTRIBUTE, g=else_branch),
        ),
    )


def build_function(
    nodes: tuple[FakeNode, ...], default_attributes: tuple[FakeAttribute, ...] = ()
) -> FakeFunction:
    return FakeFunction(node=nodes, attribute_proto=default_attributes)


def build_model_with_function(function: FakeFunction) -> FakeModel:
    return FakeModel(graph=FakeGraph(node=(CALL_TO_THE_LOCAL_FUNCTION,)), functions=(function,))


def place_in_main_graph(placement: TensorPlacement) -> FakeModel:
    return FakeModel(graph=build_graph(placement))


def place_in_if_branch(placement: TensorPlacement) -> FakeModel:
    return FakeModel(graph=FakeGraph(node=(build_if(FakeGraph(), build_graph(placement)),)))


def place_in_loop_body(placement: TensorPlacement) -> FakeModel:
    loop = FakeNode("Loop", attribute=(FakeAttribute(GRAPH_ATTRIBUTE, g=build_graph(placement)),))
    return FakeModel(graph=FakeGraph(node=(loop,)))


def place_in_graphs_attribute(placement: TensorPlacement) -> FakeModel:
    node = FakeNode(
        "Scan",
        attribute=(FakeAttribute(GRAPHS_ATTRIBUTE, graphs=(FakeGraph(), build_graph(placement))),),
    )
    return FakeModel(graph=FakeGraph(node=(node,)))


def place_in_function(placement: TensorPlacement) -> FakeModel:
    return build_model_with_function(
        build_function(
            (FakeNode("Constant", attribute=placement.node_attributes),),
            placement.default_attributes,
        )
    )


def place_in_function_subgraph(placement: TensorPlacement) -> FakeModel:
    return build_model_with_function(
        build_function((build_if(FakeGraph(), build_graph(placement)),))
    )


def place_in_function_default_graph(placement: TensorPlacement) -> FakeModel:
    return build_model_with_function(
        build_function((), (FakeAttribute(GRAPH_ATTRIBUTE, g=build_graph(placement)),))
    )


def build_external_tensor(location: str, offset_bytes: int) -> FakeTensor:
    return FakeTensor(
        data_location=EXTERNAL_DATA_LOCATION,
        external_data=(
            FakeStringEntry("location", location),
            FakeStringEntry("offset", str(offset_bytes)),
            FakeStringEntry("length", str(TENSOR_LENGTH_BYTES)),
        ),
    )


GRAPH_TENSOR_PLACES: dict[str, Callable[[FakeTensor], TensorPlacement]] = {
    "an initializer": place_in_initializer,
    "a sparse initializer's values": place_in_sparse_values,
    "a sparse initializer's indices": place_in_sparse_indices,
}
NODE_TENSOR_PLACES: dict[str, Callable[[FakeTensor], TensorPlacement]] = {
    "a tensor attribute": place_in_tensor_attribute,
    "a tensors attribute": place_in_tensors_attribute,
    "a sparse tensor attribute": place_in_sparse_tensor_attribute,
    "a sparse tensors attribute": place_in_sparse_tensors_attribute,
}
FUNCTION_TENSOR_PLACES: dict[str, Callable[[FakeTensor], TensorPlacement]] = {
    "a function's default attribute": place_in_default_attribute,
}
GRAPH_PLACES: dict[str, Callable[[TensorPlacement], FakeModel]] = {
    "the main graph": place_in_main_graph,
    "an If branch": place_in_if_branch,
    "a Loop body": place_in_loop_body,
    "a graphs attribute": place_in_graphs_attribute,
    "a subgraph inside a function": place_in_function_subgraph,
    "a graph among a function's default attributes": place_in_function_default_graph,
}
FUNCTION_PLACES: dict[str, Callable[[TensorPlacement], FakeModel]] = {
    "a function": place_in_function,
}
TENSOR_PLACES_IN_EVERY_GRAPH = itertools.product(
    (GRAPH_TENSOR_PLACES | NODE_TENSOR_PLACES).items(), GRAPH_PLACES.items()
)
TENSOR_PLACES_IN_A_FUNCTION = itertools.product(
    (NODE_TENSOR_PLACES | FUNCTION_TENSOR_PLACES).items(), FUNCTION_PLACES.items()
)
TENSOR_PLACES = {
    f"{tensor_place_name} in {graph_place_name}": (place_tensor, place_graph)
    for (tensor_place_name, place_tensor), (graph_place_name, place_graph) in (
        *TENSOR_PLACES_IN_EVERY_GRAPH,
        *TENSOR_PLACES_IN_A_FUNCTION,
    )
}
PRODUCTS_IN_EVERY_SCOPE = FakeModel(
    graph=FakeGraph(
        node=(
            FakeNode("MatMul", input=("input", WEIGHT_NAME), output=("graph_product",)),
            build_if(
                FakeGraph(node=(FakeNode("MatMul", input=("input", WEIGHT_NAME)),)),
                FakeGraph(),
            ),
            CALL_TO_THE_LOCAL_FUNCTION,
        ),
        initializer=(FakeTensor(WEIGHT_NAME, (4, 4)),),
    ),
    functions=(
        build_function(
            (
                FakeNode("Constant", output=("constant_weight",)),
                FakeNode("Transpose", input=("constant_weight",), output=("transposed",)),
                FakeNode("MatMul", input=("input", "transposed")),
            )
        ),
    ),
)
SIBLING_BRANCHES = FakeModel(
    graph=FakeGraph(
        node=(
            build_if(
                FakeGraph(initializer=(FakeTensor(SIBLING_WEIGHT_NAME, (4, 4)),)),
                FakeGraph(node=(FakeNode("MatMul", input=("input", SIBLING_WEIGHT_NAME)),)),
            ),
        )
    )
)
FUNCTION_WEIGHT_NAMED_LIKE_AN_INITIALIZER = FakeModel(
    graph=FakeGraph(
        node=(CALL_TO_THE_LOCAL_FUNCTION,), initializer=(FakeTensor(WEIGHT_NAME, (4, 4)),)
    ),
    functions=(build_function((FakeNode("MatMul", input=("input", WEIGHT_NAME)),)),),
)
TENSORS_OF_TWO_DATA_FILES = FakeModel(
    graph=FakeGraph(
        initializer=(
            build_external_tensor(DATA_FILE_NAME, TENSOR_OFFSET_BYTES + TENSOR_LENGTH_BYTES),
            build_external_tensor(OTHER_DATA_FILE_NAME, OTHER_DATA_FILE_OFFSET_BYTES),
            build_external_tensor(DATA_FILE_NAME, TENSOR_OFFSET_BYTES),
        )
    )
)
METADATA_IN_A_FUNCTION_SUBGRAPH = build_model_with_function(
    build_function(
        (
            build_if(
                FakeGraph(
                    node=(
                        FakeNode(
                            "Softmax", metadata_props=(FakeStringEntry("source", "/tmp/x.py"),)
                        ),
                    )
                ),
                FakeGraph(),
            ),
        )
    )
)


def declare(name: str, element_type: ElementType) -> FakeValueInfo:
    return FakeValueInfo(name, FakeType(FakeTensorType(element_type)))


def build_cast(source: str, target: str, element_type: ElementType) -> FakeNode:
    return FakeNode(
        "Cast",
        input=(source,),
        output=(target,),
        attribute=(FakeAttribute(INT_ATTRIBUTE, name="to", i=element_type),),
    )


# The length chain of torch 2.15's export of the encoder: each round of the subsampling adds,
# subtracts and divides, with a Floor between casts to float32 and back, beside an audio path that
# ends in the encoder's outputs; the declarations of its tensors give the precision it rounds in.
ENCODER_NODES = (
    FakeNode("Relu", input=("audio_signal",), output=("features",)),
    FakeNode("MatMul", input=("features", WEIGHT_NAME), output=("outputs",)),
    build_cast("length", "length_rounded", ElementType.FLOAT),
    FakeNode("Add", input=("length_rounded", "padding"), output=("padded",)),
    FakeNode("Add", input=("padded", "padding"), output=("padded_twice",)),
    FakeNode("Sub", input=("padded_twice", "kernel_size"), output=("shortened",)),
    FakeNode("Div", input=("shortened", "stride"), output=("divided",)),
    build_cast("divided", "divided_float", ElementType.FLOAT),
    FakeNode("Floor", input=("divided_float",), output=("floored",)),
    build_cast("floored", "floored_rounded", ElementType.FLOAT),
    build_cast("floored_rounded", "encoded_lengths", ElementType.INT64),
)


def declare_length_chain(rounding_type: ElementType) -> tuple[FakeValueInfo, ...]:
    return (
        declare("features", ElementType.FLOAT),
        declare("length_rounded", rounding_type),
        declare("padded", rounding_type),
        declare("padded_twice", rounding_type),
        declare("shortened", rounding_type),
        declare("divided", rounding_type),
        declare("divided_float", ElementType.FLOAT),
        declare("floored", ElementType.FLOAT),
        declare("floored_rounded", rounding_type),
    )


ENCODER_INPUTS = (declare("audio_signal", ElementType.FLOAT), declare("length", ElementType.INT64))
ENCODER_OUTPUTS = (
    declare("outputs", ElementType.FLOAT),
    declare("encoded_lengths", ElementType.INT64),
)
CHAIN_INITIALIZERS = tuple(
    FakeTensor(name, ()) for name in (WEIGHT_NAME, "padding", "kernel_size", "stride")
)


def build_encoder(
    nodes: tuple[FakeNode, ...],
    value_info: tuple[FakeValueInfo, ...],
    functions: tuple[FakeFunction, ...] = (),
    outputs: tuple[FakeValueInfo, ...] = ENCODER_OUTPUTS,
) -> FakeModel:
    return FakeModel(
        graph=FakeGraph(
            node=nodes,
            initializer=CHAIN_INITIALIZERS,
            input=ENCODER_INPUTS,
            output=outputs,
            value_info=value_info,
        ),
        functions=functions,
    )


def replace_chain_node(
    nodes: tuple[FakeNode, ...], operator: str, replacement: FakeNode
) -> tuple[FakeNode, ...]:
    return tuple(replacement if node.op_type == operator else node for node in nodes)


def drop_declaration(value_info: tuple[FakeValueInfo, ...], name: str) -> tuple[FakeValueInfo, ...]:
    return tuple(declaration for declaration in value_info if declaration.name != name)


BFLOAT16_ENCODER = build_encoder(ENCODER_NODES, declare_length_chain(ElementType.BFLOAT16))
FLOAT32_DECLARATIONS = declare_length_chain(ElementType.FLOAT)
BRANCH_THAT_READS_THE_CHAIN = FakeGraph(
    node=(FakeNode("Sub", input=("padded_twice", "kernel_size"), output=("branch_shortened",)),),
    output=(declare("branch_shortened", ElementType.FLOAT),),
)
OUTPUT_COMPUTATIONS = {
    "a cast pair that the conversion removed leaves a float16 declaration that no node writes": (
        build_encoder(
            ENCODER_NODES,
            (*FLOAT32_DECLARATIONS, declare("removed_cast_output", ElementType.FLOAT16)),
        ),
        (ElementType.FLOAT, ElementType.INT64),
    ),
    "a tensor of the computation without a declared type": (
        build_encoder(ENCODER_NODES, drop_declaration(FLOAT32_DECLARATIONS, "shortened")),
        (ElementType.UNDEFINED, ElementType.FLOAT, ElementType.INT64),
    ),
    "an If whose branch reads a tensor of the chain": (
        build_encoder(
            replace_chain_node(
                ENCODER_NODES,
                "Sub",
                FakeNode(
                    "If",
                    input=("condition",),
                    output=("shortened",),
                    attribute=(
                        FakeAttribute(GRAPH_ATTRIBUTE, g=BRANCH_THAT_READS_THE_CHAIN),
                        FakeAttribute(GRAPH_ATTRIBUTE, g=BRANCH_THAT_READS_THE_CHAIN),
                    ),
                ),
            ),
            FLOAT32_DECLARATIONS,
        ),
        (ElementType.UNDEFINED, ElementType.FLOAT, ElementType.INT64),
    ),
    "a call to a function of the model": (
        build_encoder(
            replace_chain_node(
                ENCODER_NODES,
                "Div",
                FakeNode(
                    LOCAL_FUNCTION_NAME,
                    LOCAL_DOMAIN,
                    input=("shortened", "stride"),
                    output=("divided",),
                ),
            ),
            FLOAT32_DECLARATIONS,
            functions=(build_function((FakeNode("Div"),)),),
        ),
        (ElementType.UNDEFINED, ElementType.FLOAT, ElementType.INT64),
    ),
}
INPUT_BORNE_BY_AN_INITIALIZER = FakeModel(
    graph=FakeGraph(
        node=(FakeNode("Identity", input=("scale",), output=("scaled",)),),
        initializer=(FakeTensor("scale", ()),),
        input=(declare("scale", ElementType.FLOAT16),),
        output=(declare("scaled", ElementType.FLOAT16),),
        value_info=(),
    )
)


# onnxconverter-common leaves a float16 cast pair around a length that a cast to int64 off the path
# reads, here add_9 and _to_copy_5, even when it keeps the path from length to encoded_lengths in
# float32.
CONVERTED_WITH_A_CAST_PAIR = FakeModel(
    graph=FakeGraph(
        node=(
            FakeNode("Relu", input=("audio_signal",), output=("features",)),
            FakeNode("MatMul", input=("features", WEIGHT_NAME), output=("outputs",)),
            build_cast("length", "_to_copy", ElementType.FLOAT),
            FakeNode("Add", input=("_to_copy", "padding"), output=("add_9_float32",)),
            build_cast("add_9_float32", "add_9", ElementType.FLOAT16),
            build_cast("add_9", "_to_copy_5", ElementType.INT64),
            build_cast("add_9", "add_16_input", ElementType.FLOAT),
            FakeNode("Add", input=("add_16_input", "padding"), output=("add_16",)),
            build_cast("add_16", "encoded_lengths", ElementType.INT64),
        ),
        initializer=CHAIN_INITIALIZERS,
        input=ENCODER_INPUTS,
        output=ENCODER_OUTPUTS,
        value_info=(
            declare("features", ElementType.FLOAT),
            declare("_to_copy", ElementType.FLOAT),
            declare("add_9_float32", ElementType.FLOAT),
            declare("add_9", ElementType.FLOAT16),
            declare("_to_copy_5", ElementType.INT64),
            declare("add_16_input", ElementType.FLOAT),
            declare("add_16", ElementType.FLOAT),
        ),
    )
)
SHAPE_OF_A_FLOAT16_TENSOR = FakeModel(
    graph=FakeGraph(
        node=(
            build_cast("audio_signal", "audio_half", ElementType.FLOAT16),
            FakeNode("Shape", input=("audio_half",), output=("audio_shape",)),
            build_cast("audio_shape", "frame_count", ElementType.INT64),
        ),
        input=ENCODER_INPUTS,
        output=(declare("audio_half", ElementType.FLOAT16),),
        value_info=(
            declare("audio_half", ElementType.FLOAT16),
            declare("audio_shape", ElementType.INT64),
            declare("frame_count", ElementType.INT64),
        ),
    )
)


class TestTheModelGraphs:
    class TestTheListExternalTensorsFunction:
        @pytest.mark.parametrize(
            ("place_tensor", "place_graph"),
            list(TENSOR_PLACES.values()),
            ids=list(TENSOR_PLACES.keys()),
        )
        class TestWhenAnExternalTensorLiesInAnyPlaceOfAnyGraph:
            @pytest.fixture
            def external_tensors(
                self,
                place_tensor: Callable[[FakeTensor], TensorPlacement],
                place_graph: Callable[[TensorPlacement], FakeModel],
            ) -> tuple[ExternalTensor, ...]:
                return list_external_tensors(place_graph(place_tensor(EXTERNAL_TENSOR)))

            def test_should_list_it_once_with_its_location_offset_and_length(
                self, external_tensors: tuple[ExternalTensor, ...]
            ) -> None:
                assert external_tensors == (LISTED_TENSOR,)

        @pytest.mark.parametrize(
            ("place_tensor", "place_graph"),
            list(TENSOR_PLACES.values()),
            ids=list(TENSOR_PLACES.keys()),
        )
        class TestWhenATensorKeptInsideTheModelLiesInAnyPlaceOfAnyGraph:
            @pytest.fixture
            def external_tensors(
                self,
                place_tensor: Callable[[FakeTensor], TensorPlacement],
                place_graph: Callable[[TensorPlacement], FakeModel],
            ) -> tuple[ExternalTensor, ...]:
                return list_external_tensors(place_graph(place_tensor(INTERNAL_TENSOR)))

            def test_should_list_nothing(
                self, external_tensors: tuple[ExternalTensor, ...]
            ) -> None:
                assert external_tensors == ()

        class TestWhenTwoTensorsShareADataFileBesideAnother:
            @pytest.fixture
            def external_tensors(self) -> tuple[ExternalTensor, ...]:
                return list_external_tensors(TENSORS_OF_TWO_DATA_FILES)

            def test_should_list_them_by_data_file_then_by_offset(
                self, external_tensors: tuple[ExternalTensor, ...]
            ) -> None:
                assert external_tensors == (
                    ExternalTensor(
                        OTHER_DATA_FILE_NAME, OTHER_DATA_FILE_OFFSET_BYTES, TENSOR_LENGTH_BYTES
                    ),
                    ExternalTensor(DATA_FILE_NAME, TENSOR_OFFSET_BYTES, TENSOR_LENGTH_BYTES),
                    ExternalTensor(
                        DATA_FILE_NAME,
                        TENSOR_OFFSET_BYTES + TENSOR_LENGTH_BYTES,
                        TENSOR_LENGTH_BYTES,
                    ),
                )

    class TestTheCountNodesFunction:
        class TestWhenASubgraphLiesInsideAFunction:
            @pytest.fixture
            def node_counts(self) -> tuple[NodeCount, ...]:
                return count_nodes(METADATA_IN_A_FUNCTION_SUBGRAPH)

            def test_should_count_the_nodes_of_every_graph_by_scope(
                self, node_counts: tuple[NodeCount, ...]
            ) -> None:
                assert node_counts == (
                    NodeCount(NodeScope.FUNCTION, "", "If", 1),
                    NodeCount(NodeScope.GRAPH, LOCAL_DOMAIN, LOCAL_FUNCTION_NAME, 1),
                    NodeCount(NodeScope.SUBGRAPH, "", "Softmax", 1),
                )

    class TestTheCountNodesWithMetadataFunction:
        class TestWhenANodeWithMetadataLiesInASubgraphInsideAFunction:
            @pytest.fixture
            def metadata_node_count(self) -> int:
                return count_nodes_with_metadata(METADATA_IN_A_FUNCTION_SUBGRAPH)

            def test_should_count_it(self, metadata_node_count: int) -> None:
                assert metadata_node_count == 1

    class TestTheListProductsFunction:
        class TestWhenTheModelHoldsAProductInEveryScope:
            @pytest.fixture
            def products(self) -> tuple[MatrixProduct, ...]:
                return list_products(PRODUCTS_IN_EVERY_SCOPE)

            def test_should_trace_each_weight_in_its_own_scope(
                self, products: tuple[MatrixProduct, ...]
            ) -> None:
                assert products == (
                    MatrixProduct(NodeScope.GRAPH, "MatMul", (), WeightSource.INITIALIZER, 2),
                    MatrixProduct(NodeScope.SUBGRAPH, "MatMul", (), WeightSource.INITIALIZER, 2),
                    MatrixProduct(
                        NodeScope.FUNCTION,
                        "MatMul",
                        ("Transpose", "Constant"),
                        WeightSource.CONSTANT,
                        None,
                    ),
                )

        class TestWhenABranchReadsAnInitializerOfItsSibling:
            @pytest.fixture
            def products(self) -> tuple[MatrixProduct, ...]:
                return list_products(SIBLING_BRANCHES)

            def test_should_trace_it_as_computed_since_a_branch_sees_only_its_ancestors(
                self, products: tuple[MatrixProduct, ...]
            ) -> None:
                assert products == (
                    MatrixProduct(NodeScope.SUBGRAPH, "MatMul", (), WeightSource.COMPUTED, None),
                )

        class TestWhenAFunctionReadsAWeightNamedLikeAnInitializerOfTheMainGraph:
            @pytest.fixture
            def products(self) -> tuple[MatrixProduct, ...]:
                return list_products(FUNCTION_WEIGHT_NAMED_LIKE_AN_INITIALIZER)

            def test_should_trace_it_as_computed_since_a_function_sees_no_initializer(
                self, products: tuple[MatrixProduct, ...]
            ) -> None:
                assert products == (
                    MatrixProduct(NodeScope.FUNCTION, "MatMul", (), WeightSource.COMPUTED, None),
                )

    class TestTheCountSparseInitializersFunction:
        @pytest.mark.parametrize(
            "place_graph", list(GRAPH_PLACES.values()), ids=list(GRAPH_PLACES.keys())
        )
        class TestWhenASparseInitializerLiesInAnyGraph:
            @pytest.fixture
            def sparse_initializer_count(
                self, place_graph: Callable[[TensorPlacement], FakeModel]
            ) -> int:
                return count_sparse_initializers(
                    place_graph(place_in_sparse_values(INTERNAL_TENSOR))
                )

            def test_should_count_it(self, sparse_initializer_count: int) -> None:
                assert sparse_initializer_count == 1

        class TestWhenTheModelHoldsNone:
            @pytest.fixture
            def sparse_initializer_count(self) -> int:
                return count_sparse_initializers(place_in_main_graph(TensorPlacement()))

            def test_should_count_none(self, sparse_initializer_count: int) -> None:
                assert sparse_initializer_count == 0

    class TestTheCountTrainingGraphsFunction:
        class TestWhenTheModelHoldsOneTrainingInfo:
            @pytest.fixture
            def training_graph_count(self) -> int:
                return count_training_graphs(FakeModel(training_info=(object(),)))

            def test_should_count_its_initialization_and_its_algorithm(
                self, training_graph_count: int
            ) -> None:
                assert training_graph_count == 2

        class TestWhenTheModelHoldsNone:
            @pytest.fixture
            def training_graph_count(self) -> int:
                return count_training_graphs(FakeModel())

            def test_should_count_none(self, training_graph_count: int) -> None:
                assert training_graph_count == 0

    class TestTheListOutputComputationsFunction:
        class TestWhenTheLengthChainRoundsThroughBfloat16:
            @pytest.fixture
            def output_computations(self) -> tuple[TensorComputation, ...]:
                return list_output_computations(index_main_graph(BFLOAT16_ENCODER))

            def test_should_report_the_lengths_computed_through_bfloat16_beside_the_outputs(
                self, output_computations: tuple[TensorComputation, ...]
            ) -> None:
                assert output_computations == (
                    TensorComputation("outputs", ElementType.FLOAT, (ElementType.FLOAT,)),
                    TensorComputation(
                        "encoded_lengths",
                        ElementType.INT64,
                        (ElementType.FLOAT, ElementType.INT64, ElementType.BFLOAT16),
                    ),
                )

        class TestWhenTheLengthChainIsKeptInFloat32:
            @pytest.fixture
            def output_computations(self) -> tuple[TensorComputation, ...]:
                return list_output_computations(
                    index_main_graph(build_encoder(ENCODER_NODES, FLOAT32_DECLARATIONS))
                )

            def test_should_report_the_lengths_computed_through_float32_and_int64_alone(
                self, output_computations: tuple[TensorComputation, ...]
            ) -> None:
                assert output_computations[1] == TensorComputation(
                    "encoded_lengths", ElementType.INT64, (ElementType.FLOAT, ElementType.INT64)
                )

        @pytest.mark.parametrize(
            ("model", "expected_types"),
            list(OUTPUT_COMPUTATIONS.values()),
            ids=list(OUTPUT_COMPUTATIONS.keys()),
        )
        class TestWhenTheLengthChainHolds:
            @pytest.fixture
            def output_computations(self, model: FakeModel) -> tuple[TensorComputation, ...]:
                return list_output_computations(index_main_graph(model))

            def test_should_report_the_types_that_the_chain_writes_or_one_it_cannot_tell(
                self,
                output_computations: tuple[TensorComputation, ...],
                expected_types: tuple[ElementType, ...],
            ) -> None:
                assert output_computations[1].computation_types == expected_types

        class TestWhenAnInitializerBearsAGraphInput:
            @pytest.fixture
            def output_computations(self) -> tuple[TensorComputation, ...]:
                return list_output_computations(index_main_graph(INPUT_BORNE_BY_AN_INITIALIZER))

            def test_should_start_no_computation_from_it_since_it_is_a_parameter(
                self, output_computations: tuple[TensorComputation, ...]
            ) -> None:
                assert output_computations == (
                    TensorComputation("scaled", ElementType.FLOAT16, ()),
                )

        class TestWhenTheGraphListsTheLengthsBeforeTheOutputs:
            @pytest.fixture
            def output_computations(self) -> tuple[TensorComputation, ...]:
                return list_output_computations(
                    index_main_graph(
                        build_encoder(
                            ENCODER_NODES, FLOAT32_DECLARATIONS, outputs=ENCODER_OUTPUTS[::-1]
                        )
                    )
                )

            def test_should_report_the_outputs_in_the_graphs_order(
                self, output_computations: tuple[TensorComputation, ...]
            ) -> None:
                assert [output.name for output in output_computations] == [
                    "encoded_lengths",
                    "outputs",
                ]

    class TestTheListCastComputationsFunction:
        class TestWhenALengthGoesThroughTheCastPairThatTheConversionLeft:
            @pytest.fixture
            def cast_computations(self) -> tuple[TensorComputation, ...]:
                return list_cast_computations(index_main_graph(CONVERTED_WITH_A_CAST_PAIR))

            def test_should_report_the_cast_to_int64_computed_through_float16(
                self, cast_computations: tuple[TensorComputation, ...]
            ) -> None:
                assert [cast for cast in cast_computations if cast.name == "_to_copy_5"] == [
                    TensorComputation(
                        "_to_copy_5",
                        ElementType.INT64,
                        (ElementType.FLOAT, ElementType.INT64, ElementType.FLOAT16),
                    )
                ]

            def test_should_report_the_output_lengths_computed_through_float16_too(
                self, cast_computations: tuple[TensorComputation, ...]
            ) -> None:
                assert [cast for cast in cast_computations if cast.name == "encoded_lengths"] == [
                    TensorComputation(
                        "encoded_lengths",
                        ElementType.INT64,
                        (ElementType.FLOAT, ElementType.INT64, ElementType.FLOAT16),
                    )
                ]

        class TestWhenTheGraphCastsToSeveralTypes:
            @pytest.fixture
            def cast_computations(self) -> tuple[TensorComputation, ...]:
                return list_cast_computations(index_main_graph(CONVERTED_WITH_A_CAST_PAIR))

            def test_should_report_every_cast_in_the_graphs_order_with_its_target(
                self, cast_computations: tuple[TensorComputation, ...]
            ) -> None:
                assert [(cast.name, cast.element_type) for cast in cast_computations] == [
                    ("_to_copy", ElementType.FLOAT),
                    ("add_9", ElementType.FLOAT16),
                    ("_to_copy_5", ElementType.INT64),
                    ("add_16_input", ElementType.FLOAT),
                    ("encoded_lengths", ElementType.INT64),
                ]

        class TestWhenAnIntegerCastReadsTheShapeOfAFloat16Tensor:
            @pytest.fixture
            def cast_computations(self) -> tuple[TensorComputation, ...]:
                return list_cast_computations(index_main_graph(SHAPE_OF_A_FLOAT16_TENSOR))

            def test_should_leave_the_float16_values_out_since_a_shape_carries_none_of_them(
                self, cast_computations: tuple[TensorComputation, ...]
            ) -> None:
                assert cast_computations[1] == TensorComputation(
                    "frame_count", ElementType.INT64, (ElementType.INT64,)
                )
