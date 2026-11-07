from dataclasses import dataclass
from enum import Enum, auto

from mostt_model_builder.domain.onnx.element_types import INTEGER_ELEMENT_TYPES, ElementType
from mostt_model_builder.domain.onnx.inspection_report import (
    FileInspection,
    MatrixProduct,
    TensorComputation,
    WeightSource,
)
from mostt_model_builder.domain.onnx.onnx_runtime_names import (
    EIGHT_BIT_BLOCKS_OPERATOR,
    ONNX_RUNTIME_DOMAIN,
)

# ONNX's own operators, and ONNX Runtime's contrib operators, such as MatMulNBits.
ALLOWED_DOMAINS = frozenset({"", ONNX_RUNTIME_DOMAIN})
PLAIN_WEIGHT_RANK = 2
# None of these holds every integer of a length: float16 holds each one only up to 2 048, bfloat16
# up to 256, and the 8-, 6- and 4-bit floats fewer still, where float32 goes up to 2**24.
NARROW_FLOAT_ELEMENT_TYPES = frozenset(
    {
        ElementType.FLOAT16,
        ElementType.BFLOAT16,
        ElementType.FLOAT8E4M3FN,
        ElementType.FLOAT8E4M3FNUZ,
        ElementType.FLOAT8E5M2,
        ElementType.FLOAT8E5M2FNUZ,
        ElementType.FLOAT8E8M0,
        ElementType.FLOAT6E2M3,
        ElementType.FLOAT6E3M2,
        ElementType.FLOAT4E2M1,
    }
)
# A cast without a target, or an output that is no tensor, may carry an integer all the same.
POSSIBLE_INTEGER_ELEMENT_TYPES = INTEGER_ELEMENT_TYPES | {ElementType.UNDEFINED}
# onnx names float32 FLOAT and float64 DOUBLE, which a reader of a refusal would not recognize.
FLOAT_WIDTH_NAMES = {ElementType.FLOAT: "float32", ElementType.DOUBLE: "float64"}


class GraphRuleError(ValueError):
    pass


class GraphSource(Enum):
    CONVERSION_INPUT = auto()
    FLOAT_CONVERSION_OUTPUT = auto()
    OTHER = auto()


@dataclass(frozen=True, slots=True)
class EightBitBlocksOutput:
    declared_product_count: int


@dataclass(frozen=True, slots=True)
class ProductCounts:
    constant_weight_count: int
    plain_count: int


def is_constant_weight(product: MatrixProduct) -> bool:
    return product.weight_source is not WeightSource.COMPUTED


def is_plain(product: MatrixProduct) -> bool:
    return (
        not product.trace
        and product.weight_source is WeightSource.INITIALIZER
        and product.weight_rank == PLAIN_WEIGHT_RANK
    )


def count_products(inspection: FileInspection) -> ProductCounts:
    constant_weight = [product for product in inspection.products if is_constant_weight(product)]
    return ProductCounts(
        constant_weight_count=len(constant_weight),
        plain_count=sum(1 for product in constant_weight if is_plain(product)),
    )


def count_eight_bit_block_products(inspection: FileInspection) -> int:
    return sum(
        node.count
        for node in inspection.nodes
        if node.domain == ONNX_RUNTIME_DOMAIN and node.operator == EIGHT_BIT_BLOCKS_OPERATOR
    )


def judge_graph(
    inspection: FileInspection,
    source: GraphSource | EightBitBlocksOutput,
    external_data_file: str | None,
) -> None:
    where = f"the graph of {inspection.name}"
    refuse_foreign_domains(inspection, where)
    if inspection.metadata_props_node_count:
        raise GraphRuleError(
            f"{where} has {inspection.metadata_props_node_count} nodes that hold metadata_props, "
            "which may name paths of the container that exported it"
        )
    if inspection.training_graph_count:
        raise GraphRuleError(
            f"{where} holds {inspection.training_graph_count} training graphs, which no runtime "
            "of the candidate runs"
        )
    if inspection.sparse_initializer_count:
        raise GraphRuleError(
            f"{where} holds {inspection.sparse_initializer_count} sparse initializers, whose "
            "weights no rule of the products judges"
        )
    refuse_foreign_external_data(inspection, external_data_file, where)
    if source is GraphSource.CONVERSION_INPUT:
        refuse_products_that_are_not_plain(inspection, where)
    elif source is GraphSource.FLOAT_CONVERSION_OUTPUT:
        refuse_integers_through_narrow_floats(inspection, where)
    elif isinstance(source, EightBitBlocksOutput):
        refuse_unquantized_products(inspection, source.declared_product_count, where)


def refuse_foreign_domains(inspection: FileInspection, where: str) -> None:
    functions = {(function.domain, function.name) for function in inspection.functions}
    for node in inspection.nodes:
        if node.domain in ALLOWED_DOMAINS or (node.domain, node.operator) in functions:
            continue
        raise GraphRuleError(
            f"{where} has a node {node.operator} of the domain {node.domain!a} in its "
            f"{node.scope.value}, which is neither ONNX's, ONNX Runtime's, nor a function the "
            "model defines"
        )


def refuse_foreign_external_data(
    inspection: FileInspection, external_data_file: str | None, where: str
) -> None:
    for tensor in inspection.external_tensors:
        if tensor.location != external_data_file:
            expected = (
                "which keeps every tensor inside it"
                if external_data_file is None
                else f"whose external data is {external_data_file}"
            )
            raise GraphRuleError(f"{where} reads a tensor from {tensor.location}, {expected}")


def refuse_products_that_are_not_plain(inspection: FileInspection, where: str) -> None:
    for product in inspection.products:
        if is_constant_weight(product) and not is_plain(product):
            raise GraphRuleError(
                f"{where}, the conversions' input, has a {product.operator} whose constant weight "
                f"is not a plain 2-D initializer: it reads {describe_weight(product)}"
            )


def refuse_unquantized_products(
    inspection: FileInspection, declared_product_count: int, where: str
) -> None:
    for product in inspection.products:
        if is_constant_weight(product):
            raise GraphRuleError(
                f"{where}, the 8-bit conversion's output, keeps a {product.operator} of a "
                f"constant weight unquantized: it reads {describe_weight(product)}"
            )
    product_count = count_eight_bit_block_products(inspection)
    if product_count != declared_product_count:
        raise GraphRuleError(
            f"{where}, the 8-bit conversion's output, holds {product_count} "
            f"{EIGHT_BIT_BLOCKS_OPERATOR} nodes, and its recipe declares {declared_product_count}"
        )


# The bench cannot see an integer, such as an encoder's lengths, that a narrow float rounded: a
# truncated length passes unseen, and one beyond the encoder's output only crashes its reader. One
# refusal names every such integer, so that the whole defect shows at once.
def refuse_integers_through_narrow_floats(inspection: FileInspection, where: str) -> None:
    roundings = [
        rounding
        for kind, computations in (("output", inspection.outputs), ("cast", inspection.casts))
        for computation in computations
        if (rounding := describe_rounding(kind, computation)) is not None
    ]
    if roundings:
        raise GraphRuleError(
            f"{where}, a float conversion's output, computes integers through a float narrower "
            "than float32, which cannot hold every integer they may carry, or through a type the "
            f"inspection cannot tell: {'; '.join(roundings)}"
        )


def describe_rounding(kind: str, computation: TensorComputation) -> str | None:
    if computation.element_type not in POSSIBLE_INTEGER_ELEMENT_TYPES:
        return None
    causes = [
        name_element_type(element_type)
        for element_type in computation.computation_types
        if element_type in NARROW_FLOAT_ELEMENT_TYPES
    ]
    if ElementType.UNDEFINED in computation.computation_types:
        causes.append("a type the inspection cannot tell")
    if not causes:
        return None
    if computation.element_type is ElementType.UNDEFINED:
        subject = f"the {kind} {computation.name}, of a type the inspection cannot tell,"
    else:
        subject = f"the {name_element_type(computation.element_type)} {kind} {computation.name}"
    return f"{subject} through {' and '.join(causes)}"


def name_element_type(element_type: ElementType) -> str:
    return FLOAT_WIDTH_NAMES.get(element_type, element_type.name.lower())


def describe_weight(product: MatrixProduct) -> str:
    trace = product.trace
    if product.weight_source is WeightSource.INITIALIZER:
        source = f"an initializer of rank {product.weight_rank}"
    else:
        source = "a Constant node"
        trace = trace[:-1]
    if not trace:
        return f"{source}, directly"
    return f"{source} through {' -> '.join(reversed(trace))}"
