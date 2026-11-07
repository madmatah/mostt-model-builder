import itertools
import re
from collections.abc import Callable
from dataclasses import replace
from functools import partial

import pytest

from mostt_model_builder.domain.onnx.element_types import INTEGER_ELEMENT_TYPES, ElementType
from mostt_model_builder.domain.onnx.graph_rules import (
    NARROW_FLOAT_ELEMENT_TYPES,
    EightBitBlocksOutput,
    GraphRuleError,
    GraphSource,
    ProductCounts,
    count_eight_bit_block_products,
    count_products,
    is_constant_weight,
    is_plain,
    judge_graph,
    name_element_type,
)
from mostt_model_builder.domain.onnx.inspection_report import (
    ExternalTensor,
    FileInspection,
    LocalFunction,
    MatrixProduct,
    NodeCount,
    NodeScope,
    Opset,
    TensorComputation,
    WeightSource,
)

EXTERNAL_DATA_FILE = "encoder.onnx.data"
LOCAL_DOMAIN = "mostt.local"
LOCAL_FUNCTION = LocalFunction(LOCAL_DOMAIN, "RelativePositionAttention")
CALL_TO_THE_LOCAL_FUNCTION = NodeCount(
    NodeScope.GRAPH, LOCAL_DOMAIN, "RelativePositionAttention", 17
)
PLAIN_PRODUCT = MatrixProduct(NodeScope.GRAPH, "MatMul", (), WeightSource.INITIALIZER, 2)
PLAIN_GEMM = MatrixProduct(NodeScope.GRAPH, "Gemm", (), WeightSource.INITIALIZER, 2)
COMPUTED_PRODUCT = MatrixProduct(NodeScope.FUNCTION, "MatMul", (), WeightSource.COMPUTED, None)
TRANSPOSED_PRODUCT = MatrixProduct(
    NodeScope.GRAPH, "MatMul", ("Transpose",), WeightSource.INITIALIZER, 2
)
SQUEEZED_PRODUCT = MatrixProduct(
    NodeScope.GRAPH, "MatMul", ("Squeeze",), WeightSource.INITIALIZER, 3
)
CONSTANT_NODE_PRODUCT = MatrixProduct(
    NodeScope.SUBGRAPH, "MatMul", ("Constant",), WeightSource.CONSTANT, None
)
THREE_DIMENSIONAL_PRODUCT = MatrixProduct(
    NodeScope.GRAPH, "MatMul", (), WeightSource.INITIALIZER, 3
)
TRANSPOSED_GEMM = MatrixProduct(
    NodeScope.GRAPH, "Gemm", ("Transpose",), WeightSource.INITIALIZER, 2
)
CAST_THEN_TRANSPOSED_PRODUCT = MatrixProduct(
    NodeScope.GRAPH, "MatMul", ("Transpose", "Cast"), WeightSource.INITIALIZER, 2
)
ACCEPTED_INSPECTION = FileInspection(
    name="encoder.onnx",
    opsets=(Opset("", 17), Opset("com.microsoft", 1), Opset(LOCAL_DOMAIN, 1)),
    functions=(LOCAL_FUNCTION,),
    nodes=(
        NodeCount(NodeScope.GRAPH, "", "MatMul", 120),
        NodeCount(NodeScope.GRAPH, "com.microsoft", "MatMulNBits", 4),
        CALL_TO_THE_LOCAL_FUNCTION,
        NodeCount(NodeScope.SUBGRAPH, "", "Add", 4),
        NodeCount(NodeScope.FUNCTION, "", "Softmax", 2),
    ),
    metadata_props_node_count=0,
    products=(PLAIN_PRODUCT, PLAIN_GEMM, COMPUTED_PRODUCT),
    external_tensors=(
        ExternalTensor(EXTERNAL_DATA_FILE, 0, 4096),
        ExternalTensor(EXTERNAL_DATA_FILE, 4096, 1024),
    ),
    training_graph_count=0,
    sparse_initializer_count=0,
    outputs=(),
    casts=(),
)
WHERE = "the graph of encoder.onnx"
DECLARED_BLOCK_PRODUCT_COUNT = 4
EIGHT_BIT_OUTPUT = replace(ACCEPTED_INSPECTION, products=(COMPUTED_PRODUCT,), external_tensors=())
NOT_PLAIN = "whose constant weight is not a plain 2-D initializer"


# A bfloat16 encoder converted without keeping its lengths: torch's export computes them in float32,
# and the conversion rounds every node of that chain but the Floor to bfloat16.
LENGTHS_THROUGH_BFLOAT16 = TensorComputation(
    "encoded_lengths",
    ElementType.INT64,
    (ElementType.FLOAT, ElementType.INT64, ElementType.BFLOAT16),
)
THROUGH_FLOAT32 = (ElementType.FLOAT, ElementType.INT64)
LENGTHS_THROUGH_FLOAT32 = TensorComputation("encoded_lengths", ElementType.INT64, THROUGH_FLOAT32)
OUTPUTS_THROUGH_BFLOAT16 = TensorComputation(
    "outputs", ElementType.FLOAT, (ElementType.FLOAT, ElementType.BFLOAT16)
)
THROUGH_FLOAT16 = (ElementType.FLOAT, ElementType.INT64, ElementType.FLOAT16)
# onnxconverter-common leaves a float16 cast pair around a length that a cast to int64 off the path
# from length to encoded_lengths reads, even when it keeps that path in float32.
ROUNDED_CAST = TensorComputation("_to_copy_5", ElementType.INT64, THROUGH_FLOAT16)
FLOAT_CONVERSION_WHERE = (
    f"{WHERE}, a float conversion's output, computes integers through a float narrower than "
    "float32, which cannot hold every integer they may carry, or through a type the inspection "
    "cannot tell:"
)
NARROW_INTEGER_REFUSALS = {
    "the lengths computed through bfloat16, as the converted encoder computes them": (
        replace(ACCEPTED_INSPECTION, outputs=(OUTPUTS_THROUGH_BFLOAT16, LENGTHS_THROUGH_BFLOAT16)),
        f"{FLOAT_CONVERSION_WHERE} the int64 output encoded_lengths through bfloat16",
    ),
    "the lengths computed through float16": (
        replace(
            ACCEPTED_INSPECTION,
            outputs=(TensorComputation("encoded_lengths", ElementType.INT64, THROUGH_FLOAT16),),
        ),
        f"{FLOAT_CONVERSION_WHERE} the int64 output encoded_lengths through float16",
    ),
    "a cast to int64 computed through float16": (
        replace(ACCEPTED_INSPECTION, outputs=(LENGTHS_THROUGH_FLOAT32,), casts=(ROUNDED_CAST,)),
        f"{FLOAT_CONVERSION_WHERE} the int64 cast _to_copy_5 through float16",
    ),
    "the lengths computed through a type the inspection cannot tell": (
        replace(
            ACCEPTED_INSPECTION,
            outputs=(
                TensorComputation(
                    "encoded_lengths",
                    ElementType.INT64,
                    (ElementType.UNDEFINED, ElementType.FLOAT, ElementType.INT64),
                ),
            ),
        ),
        f"{FLOAT_CONVERSION_WHERE} the int64 output encoded_lengths through a type the "
        "inspection cannot tell",
    ),
    "a cast through float16 and a type the inspection cannot tell": (
        replace(
            ACCEPTED_INSPECTION,
            casts=(
                TensorComputation(
                    "_to_copy_5",
                    ElementType.INT64,
                    (ElementType.UNDEFINED, *THROUGH_FLOAT16),
                ),
            ),
        ),
        f"{FLOAT_CONVERSION_WHERE} the int64 cast _to_copy_5 through float16 and a type the "
        "inspection cannot tell",
    ),
    "a cast without a target, which may write an integer, computed through float16": (
        replace(
            ACCEPTED_INSPECTION,
            casts=(TensorComputation("_to_copy_5", ElementType.UNDEFINED, THROUGH_FLOAT16),),
        ),
        f"{FLOAT_CONVERSION_WHERE} the cast _to_copy_5, of a type the inspection cannot tell, "
        "through float16",
    ),
    "an output that is no tensor, which may hold integers, computed through bfloat16": (
        replace(
            ACCEPTED_INSPECTION,
            outputs=(
                TensorComputation(
                    "encoded_lengths",
                    ElementType.UNDEFINED,
                    (ElementType.FLOAT, ElementType.BFLOAT16),
                ),
            ),
        ),
        f"{FLOAT_CONVERSION_WHERE} the output encoded_lengths, of a type the inspection cannot "
        "tell, through bfloat16",
    ),
    "the published files' lengths, the output and three casts through float16, all at once": (
        replace(
            ACCEPTED_INSPECTION,
            outputs=(
                OUTPUTS_THROUGH_BFLOAT16,
                TensorComputation("encoded_lengths", ElementType.INT64, THROUGH_FLOAT16),
            ),
            casts=(
                TensorComputation("_to_copy_3", ElementType.INT64, THROUGH_FLOAT16),
                TensorComputation("features_half", ElementType.FLOAT16, THROUGH_FLOAT16),
                ROUNDED_CAST,
                TensorComputation("_to_copy_7", ElementType.INT64, THROUGH_FLOAT16),
            ),
        ),
        f"{FLOAT_CONVERSION_WHERE} the int64 output encoded_lengths through float16; the int64 "
        "cast _to_copy_3 through float16; the int64 cast _to_copy_5 through float16; the int64 "
        "cast _to_copy_7 through float16",
    ),
}
LISTINGS: dict[str, Callable[[TensorComputation], FileInspection]] = {
    "output": lambda computation: replace(ACCEPTED_INSPECTION, outputs=(computation,)),
    "cast": lambda computation: replace(ACCEPTED_INSPECTION, casts=(computation,)),
}
INTEGERS_THROUGH_NARROW_FLOATS = {
    f"{integer_type.name} through {narrow_type.name} as an {kind}": (
        LISTINGS[kind](TensorComputation("counts", integer_type, (narrow_type, integer_type))),
        f"{FLOAT_CONVERSION_WHERE} the {integer_type.name.lower()} {kind} counts through "
        f"{narrow_type.name.lower()}",
    )
    for integer_type, narrow_type, kind in itertools.product(
        sorted(INTEGER_ELEMENT_TYPES), sorted(NARROW_FLOAT_ELEMENT_TYPES), LISTINGS
    )
}
NARROW = "narrower than float32"
NOT_NARROW = "not narrower"
NARROW_KINDS = {
    ElementType.UNDEFINED: NOT_NARROW,
    ElementType.FLOAT: NOT_NARROW,
    ElementType.UINT8: NOT_NARROW,
    ElementType.INT8: NOT_NARROW,
    ElementType.UINT16: NOT_NARROW,
    ElementType.INT16: NOT_NARROW,
    ElementType.INT32: NOT_NARROW,
    ElementType.INT64: NOT_NARROW,
    ElementType.STRING: NOT_NARROW,
    ElementType.BOOL: NOT_NARROW,
    ElementType.FLOAT16: NARROW,
    ElementType.DOUBLE: NOT_NARROW,
    ElementType.UINT32: NOT_NARROW,
    ElementType.UINT64: NOT_NARROW,
    ElementType.COMPLEX64: NOT_NARROW,
    ElementType.COMPLEX128: NOT_NARROW,
    ElementType.BFLOAT16: NARROW,
    ElementType.FLOAT8E4M3FN: NARROW,
    ElementType.FLOAT8E4M3FNUZ: NARROW,
    ElementType.FLOAT8E5M2: NARROW,
    ElementType.FLOAT8E5M2FNUZ: NARROW,
    ElementType.UINT4: NOT_NARROW,
    ElementType.INT4: NOT_NARROW,
    ElementType.FLOAT4E2M1: NARROW,
    ElementType.FLOAT8E8M0: NARROW,
    ElementType.UINT2: NOT_NARROW,
    ElementType.INT2: NOT_NARROW,
    ElementType.FLOAT6E2M3: NARROW,
    ElementType.FLOAT6E3M2: NARROW,
}
ELEMENT_TYPE_NAMES = {
    "float32, which onnx names FLOAT": (ElementType.FLOAT, "float32"),
    "float64, which onnx names DOUBLE": (ElementType.DOUBLE, "float64"),
    "bfloat16 under its own name": (ElementType.BFLOAT16, "bfloat16"),
    "int64 under its own name": (ElementType.INT64, "int64"),
    "the undefined type under its own name": (ElementType.UNDEFINED, "undefined"),
}
ACCEPTED_COMPUTATIONS = {
    "an integer output and an integer cast computed through float32 alone": replace(
        ACCEPTED_INSPECTION,
        outputs=(LENGTHS_THROUGH_FLOAT32,),
        casts=(TensorComputation("_to_copy_5", ElementType.INT64, THROUGH_FLOAT32),),
    ),
    "a float output computed through bfloat16": replace(
        ACCEPTED_INSPECTION, outputs=(OUTPUTS_THROUGH_BFLOAT16,)
    ),
    "a cast to float16 computed through bfloat16": replace(
        ACCEPTED_INSPECTION,
        casts=(
            TensorComputation(
                "features_half",
                ElementType.FLOAT16,
                (ElementType.FLOAT, ElementType.FLOAT16, ElementType.BFLOAT16),
            ),
        ),
    ),
}
OTHER_SOURCES = {
    "the conversions' input": (
        ACCEPTED_INSPECTION,
        GraphSource.CONVERSION_INPUT,
        EXTERNAL_DATA_FILE,
    ),
    "the 8-bit conversion's output": (
        EIGHT_BIT_OUTPUT,
        EightBitBlocksOutput(DECLARED_BLOCK_PRODUCT_COUNT),
        None,
    ),
    "any other graph": (ACCEPTED_INSPECTION, GraphSource.OTHER, EXTERNAL_DATA_FILE),
}


def name_narrow_kind(element_type: ElementType) -> str:
    return NARROW if element_type in NARROW_FLOAT_ELEMENT_TYPES else NOT_NARROW


def add_node(node: NodeCount) -> FileInspection:
    return replace(ACCEPTED_INSPECTION, nodes=(*ACCEPTED_INSPECTION.nodes, node))


def replace_products(*products: MatrixProduct) -> FileInspection:
    return replace(ACCEPTED_INSPECTION, products=products)


ACCEPTED_GRAPHS = {
    "plain and computed products of the conversions' input": (
        ACCEPTED_INSPECTION,
        GraphSource.CONVERSION_INPUT,
        EXTERNAL_DATA_FILE,
    ),
    "a graph that keeps every tensor inside it": (
        replace(ACCEPTED_INSPECTION, external_tensors=()),
        GraphSource.CONVERSION_INPUT,
        None,
    ),
    "a graph that declares external data it does not read": (
        replace(ACCEPTED_INSPECTION, external_tensors=()),
        GraphSource.OTHER,
        EXTERNAL_DATA_FILE,
    ),
    "the 8-bit conversion's output, every constant weight in its declared count of blocks": (
        replace(ACCEPTED_INSPECTION, products=(COMPUTED_PRODUCT,), external_tensors=()),
        EightBitBlocksOutput(DECLARED_BLOCK_PRODUCT_COUNT),
        None,
    ),
}
NON_PLAIN_PRODUCTS = {
    "a weight through a Transpose": TRANSPOSED_PRODUCT,
    "a weight through a Squeeze": SQUEEZED_PRODUCT,
    "a weight held by a Constant node": CONSTANT_NODE_PRODUCT,
    "a direct 3-D initializer": THREE_DIMENSIONAL_PRODUCT,
    "a Gemm whose weight is transposed": TRANSPOSED_GEMM,
}
CONSTANT_WEIGHT_PRODUCTS = {
    "a plain MatMul": PLAIN_PRODUCT,
    "a plain Gemm": PLAIN_GEMM,
    **NON_PLAIN_PRODUCTS,
}
NON_PLAIN_REFUSALS = {
    "a weight through a Transpose": (
        TRANSPOSED_PRODUCT,
        f"has a MatMul {NOT_PLAIN}: it reads an initializer of rank 2 through Transpose",
    ),
    "a weight through a Squeeze": (
        SQUEEZED_PRODUCT,
        f"has a MatMul {NOT_PLAIN}: it reads an initializer of rank 3 through Squeeze",
    ),
    "a weight held by a Constant node": (
        CONSTANT_NODE_PRODUCT,
        f"has a MatMul {NOT_PLAIN}: it reads a Constant node, directly",
    ),
    "a direct 3-D initializer": (
        THREE_DIMENSIONAL_PRODUCT,
        f"has a MatMul {NOT_PLAIN}: it reads an initializer of rank 3, directly",
    ),
    "a Gemm whose weight is transposed": (
        TRANSPOSED_GEMM,
        f"has a Gemm {NOT_PLAIN}: it reads an initializer of rank 2 through Transpose",
    ),
}
REFUSED_GRAPHS = {
    "an unknown domain in the graph": (
        add_node(NodeCount(NodeScope.GRAPH, "com.example", "Fused", 1)),
        f"{WHERE} has a node Fused of the domain 'com.example' in its graph, which is neither "
        "ONNX's, ONNX Runtime's, nor a function the model defines",
    ),
    "an unknown domain in a subgraph": (
        add_node(NodeCount(NodeScope.SUBGRAPH, "com.example", "Fused", 1)),
        f"{WHERE} has a node Fused of the domain 'com.example' in its subgraph",
    ),
    "an unknown domain in a local function": (
        add_node(NodeCount(NodeScope.FUNCTION, "com.example", "Fused", 1)),
        f"{WHERE} has a node Fused of the domain 'com.example' in its function",
    ),
    "a call to a function the model does not define": (
        replace(ACCEPTED_INSPECTION, functions=()),
        f"{WHERE} has a node RelativePositionAttention of the domain 'mostt.local' in its graph",
    ),
    "a call to a function of that name in another domain": (
        replace(
            ACCEPTED_INSPECTION, functions=(LocalFunction("mostt.other", LOCAL_FUNCTION.name),)
        ),
        f"{WHERE} has a node RelativePositionAttention of the domain 'mostt.local'",
    ),
    "nodes that hold metadata_props": (
        replace(ACCEPTED_INSPECTION, metadata_props_node_count=3),
        f"{WHERE} has 3 nodes that hold metadata_props, which may name paths of the container "
        "that exported it",
    ),
    "an external tensor in a file other than the declared one": (
        replace(
            ACCEPTED_INSPECTION,
            external_tensors=(
                ExternalTensor(EXTERNAL_DATA_FILE, 0, 4096),
                ExternalTensor("decoder.onnx.data", 0, 1024),
            ),
        ),
        f"{WHERE} reads a tensor from decoder.onnx.data, whose external data is encoder.onnx.data",
    ),
}


class TestTheGraphRules:
    class TestTheJudgeGraphFunction:
        @pytest.mark.parametrize(
            ("inspection", "source", "external_data_file"),
            list(ACCEPTED_GRAPHS.values()),
            ids=list(ACCEPTED_GRAPHS.keys()),
        )
        class TestWhenTheGraphHoldsEveryRule:
            @pytest.fixture
            def judge_act(
                self,
                inspection: FileInspection,
                source: GraphSource,
                external_data_file: str | None,
            ) -> Callable[[], object]:
                return partial(judge_graph, inspection, source, external_data_file)

            def test_should_accept_it(self, judge_act: Callable[[], object]) -> None:
                assert judge_act() is None

        @pytest.mark.parametrize(
            ("inspection", "reason"), list(REFUSED_GRAPHS.values()), ids=list(REFUSED_GRAPHS.keys())
        )
        class TestWhenTheGraphBreaksARuleOfEveryGraph:
            @pytest.fixture
            def judge_act(self, inspection: FileInspection) -> Callable[[], object]:
                return partial(judge_graph, inspection, GraphSource.OTHER, EXTERNAL_DATA_FILE)

            def test_should_refuse_it_and_say_why(
                self, judge_act: Callable[[], object], reason: str
            ) -> None:
                with pytest.raises(GraphRuleError, match=re.escape(reason)):
                    judge_act()

        class TestWhenTheGraphReadsAnExternalTensorWithoutDeclaringExternalData:
            @pytest.fixture
            def judge_act(self) -> Callable[[], object]:
                return partial(judge_graph, ACCEPTED_INSPECTION, GraphSource.OTHER, None)

            def test_should_refuse_it_since_its_tensors_belong_inside_it(
                self, judge_act: Callable[[], object]
            ) -> None:
                with pytest.raises(
                    GraphRuleError,
                    match=re.escape(
                        f"{WHERE} reads a tensor from encoder.onnx.data, which keeps every tensor "
                        "inside it"
                    ),
                ):
                    judge_act()

        class TestWhenTheModelHoldsATrainingGraph:
            @pytest.fixture
            def judge_act(self) -> Callable[[], object]:
                return partial(
                    judge_graph,
                    replace(ACCEPTED_INSPECTION, training_graph_count=2),
                    GraphSource.OTHER,
                    EXTERNAL_DATA_FILE,
                )

            def test_should_refuse_it_since_no_runtime_of_the_candidate_runs_one(
                self, judge_act: Callable[[], object]
            ) -> None:
                with pytest.raises(
                    GraphRuleError,
                    match=re.escape(
                        f"{WHERE} holds 2 training graphs, which no runtime of the candidate runs"
                    ),
                ):
                    judge_act()

        class TestWhenTheModelHoldsASparseInitializer:
            @pytest.fixture
            def judge_act(self) -> Callable[[], object]:
                return partial(
                    judge_graph,
                    replace(ACCEPTED_INSPECTION, sparse_initializer_count=1),
                    GraphSource.OTHER,
                    EXTERNAL_DATA_FILE,
                )

            def test_should_refuse_it_since_no_product_rule_judges_its_weights(
                self, judge_act: Callable[[], object]
            ) -> None:
                with pytest.raises(
                    GraphRuleError,
                    match=re.escape(
                        f"{WHERE} holds 1 sparse initializers, whose weights no rule of the "
                        "products judges"
                    ),
                ):
                    judge_act()

        @pytest.mark.parametrize(
            ("product", "reason"),
            list(NON_PLAIN_REFUSALS.values()),
            ids=list(NON_PLAIN_REFUSALS.keys()),
        )
        class TestWhenTheConversionsInputReadsAConstantWeightThatIsNotPlain:
            @pytest.fixture
            def judge_act(self, product: MatrixProduct) -> Callable[[], object]:
                return partial(
                    judge_graph,
                    replace_products(PLAIN_PRODUCT, product),
                    GraphSource.CONVERSION_INPUT,
                    EXTERNAL_DATA_FILE,
                )

            def test_should_refuse_it_since_the_quantizer_would_leave_it_unquantized(
                self, judge_act: Callable[[], object], reason: str
            ) -> None:
                with pytest.raises(
                    GraphRuleError, match=re.escape(f"{WHERE}, the conversions' input, {reason}")
                ):
                    judge_act()

        class TestWhenTheConversionsInputReadsAWeightCastThenTransposed:
            @pytest.fixture
            def judge_act(self) -> Callable[[], object]:
                return partial(
                    judge_graph,
                    replace_products(CAST_THEN_TRANSPOSED_PRODUCT),
                    GraphSource.CONVERSION_INPUT,
                    EXTERNAL_DATA_FILE,
                )

            def test_should_name_the_path_from_the_weight_to_the_product(
                self, judge_act: Callable[[], object]
            ) -> None:
                with pytest.raises(
                    GraphRuleError,
                    match=re.escape("of rank 2 through Cast -> Transpose"),
                ):
                    judge_act()

        @pytest.mark.parametrize(
            "product", list(NON_PLAIN_PRODUCTS.values()), ids=list(NON_PLAIN_PRODUCTS.keys())
        )
        class TestWhenAnotherGraphReadsAConstantWeightThatIsNotPlain:
            @pytest.fixture
            def judge_act(self, product: MatrixProduct) -> Callable[[], object]:
                return partial(
                    judge_graph,
                    replace_products(PLAIN_PRODUCT, product),
                    GraphSource.OTHER,
                    EXTERNAL_DATA_FILE,
                )

            def test_should_accept_it_since_no_conversion_quantizes_that_graph(
                self, judge_act: Callable[[], object]
            ) -> None:
                assert judge_act() is None

        @pytest.mark.parametrize(
            "product",
            list(CONSTANT_WEIGHT_PRODUCTS.values()),
            ids=list(CONSTANT_WEIGHT_PRODUCTS.keys()),
        )
        class TestWhenTheEightBitConversionsOutputKeepsAConstantWeight:
            @pytest.fixture
            def judge_act(self, product: MatrixProduct) -> Callable[[], object]:
                return partial(
                    judge_graph,
                    replace(EIGHT_BIT_OUTPUT, products=(COMPUTED_PRODUCT, product)),
                    EightBitBlocksOutput(DECLARED_BLOCK_PRODUCT_COUNT),
                    None,
                )

            def test_should_refuse_it_since_the_quantization_left_it_in_float(
                self, judge_act: Callable[[], object], product: MatrixProduct
            ) -> None:
                with pytest.raises(
                    GraphRuleError,
                    match=re.escape(
                        f"{WHERE}, the 8-bit conversion's output, keeps a {product.operator} of a "
                        "constant weight unquantized"
                    ),
                ):
                    judge_act()

        class TestWhenTheEightBitConversionsOutputHoldsAnotherCountOfBlocks:
            @pytest.fixture
            def judge_act(self) -> Callable[[], object]:
                return partial(judge_graph, EIGHT_BIT_OUTPUT, EightBitBlocksOutput(265), None)

            def test_should_refuse_it_since_a_partial_quantization_would_pass_otherwise(
                self, judge_act: Callable[[], object]
            ) -> None:
                with pytest.raises(
                    GraphRuleError,
                    match=re.escape(
                        f"{WHERE}, the 8-bit conversion's output, holds 4 MatMulNBits nodes, and "
                        "its recipe declares 265"
                    ),
                ):
                    judge_act()

        @pytest.mark.parametrize(
            ("inspection", "reason"),
            list(NARROW_INTEGER_REFUSALS.values()),
            ids=list(NARROW_INTEGER_REFUSALS.keys()),
        )
        class TestWhenAFloatConversionsOutputComputesAnIntegerThroughANarrowFloat:
            @pytest.fixture
            def judge_act(self, inspection: FileInspection) -> Callable[[], object]:
                return partial(
                    judge_graph,
                    inspection,
                    GraphSource.FLOAT_CONVERSION_OUTPUT,
                    EXTERNAL_DATA_FILE,
                )

            def test_should_refuse_it_since_the_bench_cannot_see_a_rounded_integer(
                self, judge_act: Callable[[], object], reason: str
            ) -> None:
                with pytest.raises(GraphRuleError, match=re.escape(reason)):
                    judge_act()

        @pytest.mark.parametrize(
            ("inspection", "reason"),
            list(INTEGERS_THROUGH_NARROW_FLOATS.values()),
            ids=list(INTEGERS_THROUGH_NARROW_FLOATS.keys()),
        )
        class TestWhenAFloatConversionsOutputComputesAnyIntegerThroughAnyNarrowFloat:
            @pytest.fixture
            def judge_act(self, inspection: FileInspection) -> Callable[[], object]:
                return partial(
                    judge_graph,
                    inspection,
                    GraphSource.FLOAT_CONVERSION_OUTPUT,
                    EXTERNAL_DATA_FILE,
                )

            def test_should_refuse_it_and_name_the_integer_and_its_float(
                self, judge_act: Callable[[], object], reason: str
            ) -> None:
                with pytest.raises(GraphRuleError, match=f"^{re.escape(reason)}$"):
                    judge_act()

        @pytest.mark.parametrize(
            "inspection",
            list(ACCEPTED_COMPUTATIONS.values()),
            ids=list(ACCEPTED_COMPUTATIONS.keys()),
        )
        class TestWhenAFloatConversionsOutputComputesNoIntegerThroughANarrowFloat:
            @pytest.fixture
            def verdict(self, inspection: FileInspection) -> None:
                return judge_graph(
                    inspection, GraphSource.FLOAT_CONVERSION_OUTPUT, EXTERNAL_DATA_FILE
                )

            def test_should_accept_it(self, verdict: None) -> None:
                assert verdict is None

        @pytest.mark.parametrize(
            ("inspection", "source", "external_data_file"),
            list(OTHER_SOURCES.values()),
            ids=list(OTHER_SOURCES.keys()),
        )
        class TestWhenAGraphOfAnotherSourceComputesAnIntegerThroughBfloat16:
            @pytest.fixture
            def verdict(
                self,
                inspection: FileInspection,
                source: GraphSource | EightBitBlocksOutput,
                external_data_file: str | None,
            ) -> None:
                return judge_graph(
                    replace(inspection, outputs=(LENGTHS_THROUGH_BFLOAT16,), casts=(ROUNDED_CAST,)),
                    source,
                    external_data_file,
                )

            def test_should_accept_it_since_no_step_of_the_build_changed_its_types(
                self, verdict: None
            ) -> None:
                assert verdict is None

    class TestTheNarrowFloatElementTypesConstant:
        class TestWhenEachElementTypeOfOnnxIsSorted:
            @pytest.mark.parametrize(
                ("element_type", "expected_kind"),
                list(NARROW_KINDS.items()),
                ids=[element_type.name for element_type in NARROW_KINDS],
            )
            def test_should_hold_every_float_narrower_than_float32_alone(
                self, element_type: ElementType, expected_kind: str
            ) -> None:
                assert name_narrow_kind(element_type) == expected_kind

        def test_should_sort_every_element_type_of_onnx(self) -> None:
            assert set(NARROW_KINDS) == set(ElementType)

    class TestTheNameElementTypeFunction:
        class TestWhenAnElementTypeIsNamed:
            @pytest.mark.parametrize(
                ("element_type", "expected_name"),
                list(ELEMENT_TYPE_NAMES.values()),
                ids=list(ELEMENT_TYPE_NAMES.keys()),
            )
            def test_should_name_it_as_a_reader_knows_it(
                self, element_type: ElementType, expected_name: str
            ) -> None:
                assert name_element_type(element_type) == expected_name

    class TestTheCountEightBitBlockProductsFunction:
        class TestWhenTheGraphHoldsMatMulNBitsInSeveralScopesAndDomains:
            @pytest.fixture
            def product_count(self) -> int:
                return count_eight_bit_block_products(
                    replace(
                        ACCEPTED_INSPECTION,
                        nodes=(
                            *ACCEPTED_INSPECTION.nodes,
                            NodeCount(NodeScope.SUBGRAPH, "com.microsoft", "MatMulNBits", 2),
                            NodeCount(NodeScope.GRAPH, LOCAL_DOMAIN, "MatMulNBits", 1),
                        ),
                    )
                )

            def test_should_count_onnx_runtimes_nodes_in_every_scope(
                self, product_count: int
            ) -> None:
                assert product_count == 6

    class TestTheCountProductsFunction:
        class TestWhenTheGraphHoldsPlainNonPlainAndComputedProducts:
            @pytest.fixture
            def product_counts(self) -> ProductCounts:
                return count_products(
                    replace_products(
                        PLAIN_PRODUCT,
                        PLAIN_GEMM,
                        TRANSPOSED_PRODUCT,
                        CONSTANT_NODE_PRODUCT,
                        COMPUTED_PRODUCT,
                        THREE_DIMENSIONAL_PRODUCT,
                    )
                )

            def test_should_count_the_constant_weights_and_the_plain_ones_among_them(
                self, product_counts: ProductCounts
            ) -> None:
                assert product_counts == ProductCounts(constant_weight_count=5, plain_count=2)

        class TestWhenTheGraphHoldsNoProduct:
            @pytest.fixture
            def product_counts(self) -> ProductCounts:
                return count_products(replace_products())

            def test_should_count_none(self, product_counts: ProductCounts) -> None:
                assert product_counts == ProductCounts(constant_weight_count=0, plain_count=0)

    class TestTheIsPlainFunction:
        @pytest.mark.parametrize("product", [PLAIN_PRODUCT, PLAIN_GEMM], ids=["a MatMul", "a Gemm"])
        class TestWhenTheProductReadsA2DInitializerDirectly:
            @pytest.fixture
            def is_plain_product(self, product: MatrixProduct) -> bool:
                return is_plain(product)

            def test_should_call_it_plain(self, is_plain_product: bool) -> None:
                assert is_plain_product is True

        @pytest.mark.parametrize(
            "product",
            [*NON_PLAIN_PRODUCTS.values(), COMPUTED_PRODUCT],
            ids=[*NON_PLAIN_PRODUCTS.keys(), "a computed weight"],
        )
        class TestWhenTheProductReadsAnythingElse:
            @pytest.fixture
            def is_plain_product(self, product: MatrixProduct) -> bool:
                return is_plain(product)

            def test_should_not_call_it_plain(self, is_plain_product: bool) -> None:
                assert is_plain_product is False

    class TestTheIsConstantWeightFunction:
        @pytest.mark.parametrize(
            "product",
            [PLAIN_PRODUCT, TRANSPOSED_PRODUCT, CONSTANT_NODE_PRODUCT],
            ids=["a direct initializer", "a transposed initializer", "a Constant node"],
        )
        class TestWhenTheWeightIsAnInitializerOrAConstant:
            @pytest.fixture
            def is_constant(self, product: MatrixProduct) -> bool:
                return is_constant_weight(product)

            def test_should_call_it_a_constant_weight(self, is_constant: bool) -> None:
                assert is_constant is True

        class TestWhenTheWeightIsComputed:
            @pytest.fixture
            def is_constant(self) -> bool:
                return is_constant_weight(COMPUTED_PRODUCT)

            def test_should_not_call_it_a_constant_weight_since_it_is_an_activation(
                self, is_constant: bool
            ) -> None:
                assert is_constant is False
