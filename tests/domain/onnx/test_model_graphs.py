import itertools
from collections.abc import Callable
from dataclasses import dataclass, field

import pytest

from mostt_model_builder.domain.onnx.inspection_report import (
    ExternalTensor,
    MatrixProduct,
    NodeCount,
    NodeScope,
    WeightSource,
)
from mostt_model_builder.domain.onnx.model_graphs import (
    EXTERNAL_DATA_LOCATION,
    GRAPH_ATTRIBUTE,
    GRAPHS_ATTRIBUTE,
    SPARSE_TENSOR_ATTRIBUTE,
    SPARSE_TENSORS_ATTRIBUTE,
    TENSOR_ATTRIBUTE,
    TENSORS_ATTRIBUTE,
    count_nodes,
    count_nodes_with_metadata,
    count_sparse_initializers,
    count_training_graphs,
    list_external_tensors,
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
    t: FakeTensor = field(default_factory=FakeTensor)
    tensors: tuple[FakeTensor, ...] = ()
    sparse_tensor: FakeSparseTensor = field(default_factory=FakeSparseTensor)
    sparse_tensors: tuple[FakeSparseTensor, ...] = ()
    g: "FakeGraph" = field(default_factory=lambda: FakeGraph())
    graphs: tuple["FakeGraph", ...] = ()


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


@dataclass(frozen=True, slots=True)
class FakeFunction:
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
