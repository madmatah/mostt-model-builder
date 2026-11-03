from dataclasses import dataclass
from enum import Enum, auto

from mostt_model_builder.domain.onnx.inspection_report import (
    FileInspection,
    MatrixProduct,
    WeightSource,
)

# ONNX's own operators, and ONNX Runtime's contrib operators, such as MatMulNBits.
ALLOWED_DOMAINS = frozenset({"", "com.microsoft"})
PLAIN_WEIGHT_RANK = 2


class GraphRuleError(ValueError):
    pass


class GraphSource(Enum):
    CONVERSION_INPUT = auto()
    OTHER = auto()


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


def judge_graph(
    inspection: FileInspection, source: GraphSource, external_data_file: str | None
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
