import json
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum

from mostt_model_builder.domain.formats.file_names import is_file_name
from mostt_model_builder.domain.json_documents.json_depth import MAX_JSON_DEPTH, refuse_deep_json

INSPECTION_REPORT_FILE_NAME = "inspection.json"
INSPECTION_REQUEST_FILE_NAME = "request.json"
# The report of an encoder of 1 643 nodes and 613 external tensors weighs about 100 KB; the bounds
# only refuse a report that no graph of the build could give.
INSPECTION_REPORT_MAX_SIZE_BYTES = 16 * 2**20
MAX_FILE_COUNT = 64
INSPECTION_REQUEST_MAX_SIZE_BYTES = 64 * 1024
MAX_LIST_LENGTH = 100_000
MAX_COUNT = 2**31
MAX_OFFSET_BYTES = 2**62
MAX_RANK = 16
MAX_OPSET_VERSION = 1000
DOMAIN_GRAMMAR = re.compile(r"[A-Za-z0-9._-]{0,128}")
OPERATOR_GRAMMAR = re.compile(r"[A-Za-z0-9_]{1,128}")
# The producers the inspection traces a product's weight through: each holds only constants when
# its own first input does, so that a weight reached through them is still a constant.
CONSTANT_PRODUCERS = frozenset(
    {"Constant", "Identity", "Transpose", "Squeeze", "Unsqueeze", "Reshape", "Cast"}
)
MAX_TRACE_LENGTH = 16
REPORT_KEYS = frozenset({"files"})
FILE_KEYS = frozenset(
    {
        "name",
        "opsets",
        "functions",
        "nodes",
        "metadata_props_node_count",
        "products",
        "external_tensors",
        "training_graph_count",
        "sparse_initializer_count",
    }
)
OPSET_KEYS = frozenset({"domain", "version"})
FUNCTION_KEYS = frozenset({"domain", "name"})
NODE_KEYS = frozenset({"scope", "domain", "operator", "count"})
PRODUCT_KEYS = frozenset({"scope", "operator", "trace", "weight_source", "weight_rank"})
EXTERNAL_TENSOR_KEYS = frozenset({"location", "offset_bytes", "length_bytes"})
REQUEST_KEYS = frozenset({"files"})


class InspectionReportError(ValueError):
    pass


class NodeScope(StrEnum):
    GRAPH = "graph"
    SUBGRAPH = "subgraph"
    FUNCTION = "function"


class WeightSource(StrEnum):
    INITIALIZER = "initializer"
    CONSTANT = "constant"
    COMPUTED = "computed"


@dataclass(frozen=True, slots=True)
class Opset:
    domain: str
    version: int


@dataclass(frozen=True, slots=True)
class LocalFunction:
    domain: str
    name: str


@dataclass(frozen=True, slots=True)
class NodeCount:
    scope: NodeScope
    domain: str
    operator: str
    count: int


@dataclass(frozen=True, slots=True)
class MatrixProduct:
    scope: NodeScope
    operator: str
    trace: tuple[str, ...]
    weight_source: WeightSource
    weight_rank: int | None


@dataclass(frozen=True, slots=True)
class ExternalTensor:
    location: str
    offset_bytes: int
    length_bytes: int


@dataclass(frozen=True, slots=True)
class FileInspection:
    name: str
    opsets: tuple[Opset, ...]
    functions: tuple[LocalFunction, ...]
    nodes: tuple[NodeCount, ...]
    metadata_props_node_count: int
    products: tuple[MatrixProduct, ...]
    external_tensors: tuple[ExternalTensor, ...]
    training_graph_count: int
    sparse_initializer_count: int


@dataclass(frozen=True, slots=True)
class InspectionReport:
    files: tuple[FileInspection, ...]


def render_inspection_request(file_names: Sequence[str]) -> bytes:
    return json.dumps({"files": list(file_names)}, indent=1).encode("ascii")


def parse_inspection_request(content: bytes) -> tuple[str, ...]:
    if len(content) > INSPECTION_REQUEST_MAX_SIZE_BYTES:
        raise InspectionReportError(
            f"the request holds {len(content)} bytes, beyond {INSPECTION_REQUEST_MAX_SIZE_BYTES}"
        )
    document = load_document(content, "the request")
    require_keys(document, REQUEST_KEYS, "the request")
    names = require_list(document["files"], "the request's files", MAX_FILE_COUNT)
    return tuple(require_file_name(name, "a file of the request") for name in names)


def render_inspection_report(report: InspectionReport) -> bytes:
    document = {
        "files": [
            {
                "name": inspection.name,
                "opsets": [
                    {"domain": opset.domain, "version": opset.version}
                    for opset in inspection.opsets
                ],
                "functions": [
                    {"domain": function.domain, "name": function.name}
                    for function in inspection.functions
                ],
                "nodes": [
                    {
                        "scope": node.scope.value,
                        "domain": node.domain,
                        "operator": node.operator,
                        "count": node.count,
                    }
                    for node in inspection.nodes
                ],
                "metadata_props_node_count": inspection.metadata_props_node_count,
                "products": [
                    {
                        "scope": product.scope.value,
                        "operator": product.operator,
                        "trace": list(product.trace),
                        "weight_source": product.weight_source.value,
                        "weight_rank": product.weight_rank,
                    }
                    for product in inspection.products
                ],
                "external_tensors": [
                    {
                        "location": tensor.location,
                        "offset_bytes": tensor.offset_bytes,
                        "length_bytes": tensor.length_bytes,
                    }
                    for tensor in inspection.external_tensors
                ],
                "training_graph_count": inspection.training_graph_count,
                "sparse_initializer_count": inspection.sparse_initializer_count,
            }
            for inspection in report.files
        ]
    }
    return json.dumps(document, indent=1).encode("ascii")


def parse_inspection_report(content: bytes, expected_names: Sequence[str]) -> InspectionReport:
    if len(content) > INSPECTION_REPORT_MAX_SIZE_BYTES:
        raise InspectionReportError(
            f"the inspection's report holds {len(content)} bytes, beyond "
            f"{INSPECTION_REPORT_MAX_SIZE_BYTES}"
        )
    document = load_document(content, "the inspection's report")
    require_keys(document, REPORT_KEYS, "the inspection's report")
    entries = require_list(document["files"], "the report's files", MAX_FILE_COUNT)
    files = tuple(parse_file_inspection(entry) for entry in entries)
    names = [inspection.name for inspection in files]
    if names != list(expected_names):
        raise InspectionReportError(
            f"the inspection's report describes {names!a}, not {list(expected_names)}"
        )
    return InspectionReport(files)


def load_document(content: bytes, what: str) -> Mapping[str, object]:
    try:
        document = json.loads(
            refuse_deep_json(content.decode("ascii"), MAX_JSON_DEPTH),
            parse_constant=refuse_constant,
            object_pairs_hook=refuse_duplicate_keys,
        )
    except (UnicodeDecodeError, ValueError) as error:
        raise InspectionReportError(f"{what} is not ASCII JSON: {error}") from error
    if not isinstance(document, dict):
        raise InspectionReportError(f"{what} is not a JSON object")
    return document


def refuse_constant(name: str) -> object:
    raise InspectionReportError(f"the report holds {name}, which is not a JSON number")


def parse_file_inspection(entry: object) -> FileInspection:
    fields = require_object(entry, FILE_KEYS, "a file of the report")
    name = require_file_name(fields["name"], "a file of the report")
    where = f"the inspection of {name}"
    return FileInspection(
        name=name,
        opsets=parse_entries(fields["opsets"], f"{where}: its opsets", parse_opset),
        functions=parse_entries(fields["functions"], f"{where}: its functions", parse_function),
        nodes=parse_entries(fields["nodes"], f"{where}: its nodes", parse_node_count),
        metadata_props_node_count=require_count(
            fields["metadata_props_node_count"], f"{where}: its count of nodes with metadata"
        ),
        products=parse_entries(fields["products"], f"{where}: its products", parse_product),
        external_tensors=parse_entries(
            fields["external_tensors"], f"{where}: its external tensors", parse_external_tensor
        ),
        training_graph_count=require_count(
            fields["training_graph_count"], f"{where}: its count of training graphs"
        ),
        sparse_initializer_count=require_count(
            fields["sparse_initializer_count"], f"{where}: its count of sparse initializers"
        ),
    )


def parse_entries[Entry](
    field: object, what: str, parse_entry: Callable[[object, str], Entry]
) -> tuple[Entry, ...]:
    return tuple(parse_entry(entry, what) for entry in require_list(field, what, MAX_LIST_LENGTH))


def parse_opset(entry: object, what: str) -> Opset:
    fields = require_object(entry, OPSET_KEYS, what)
    version = require_count(fields["version"], f"{what}: a version")
    if version > MAX_OPSET_VERSION:
        raise InspectionReportError(f"{what}: the version {version} is beyond {MAX_OPSET_VERSION}")
    return Opset(require_grammar(fields["domain"], DOMAIN_GRAMMAR, f"{what}: a domain"), version)


def parse_function(entry: object, what: str) -> LocalFunction:
    fields = require_object(entry, FUNCTION_KEYS, what)
    return LocalFunction(
        require_grammar(fields["domain"], DOMAIN_GRAMMAR, f"{what}: a domain"),
        require_grammar(fields["name"], OPERATOR_GRAMMAR, f"{what}: a name"),
    )


def parse_node_count(entry: object, what: str) -> NodeCount:
    fields = require_object(entry, NODE_KEYS, what)
    count = require_count(fields["count"], f"{what}: a count")
    if count == 0:
        raise InspectionReportError(f"{what}: a count of 0 nodes")
    return NodeCount(
        scope=require_choice(fields["scope"], NodeScope, f"{what}: a scope"),
        domain=require_grammar(fields["domain"], DOMAIN_GRAMMAR, f"{what}: a domain"),
        operator=require_grammar(fields["operator"], OPERATOR_GRAMMAR, f"{what}: an operator"),
        count=count,
    )


def parse_product(entry: object, what: str) -> MatrixProduct:
    fields = require_object(entry, PRODUCT_KEYS, what)
    trace = require_list(fields["trace"], f"{what}: a trace", MAX_TRACE_LENGTH)
    for producer in trace:
        if not isinstance(producer, str) or producer not in CONSTANT_PRODUCERS:
            raise InspectionReportError(
                f"{what}: the trace passes through {producer!a}, which holds no constant"
            )
    weight_source = require_choice(fields["weight_source"], WeightSource, f"{what}: a source")
    weight_rank = fields["weight_rank"]
    if weight_source is WeightSource.INITIALIZER:
        rank = require_count(weight_rank, f"{what}: a rank")
        if rank > MAX_RANK:
            raise InspectionReportError(f"{what}: the rank {rank} is beyond {MAX_RANK}")
        weight_rank = rank
    elif weight_rank is not None:
        raise InspectionReportError(f"{what}: a rank for a weight that no initializer holds")
    operator = require_grammar(fields["operator"], OPERATOR_GRAMMAR, f"{what}: an operator")
    if operator not in {"MatMul", "Gemm"}:
        raise InspectionReportError(f"{what}: {operator} is not a matrix product")
    return MatrixProduct(
        scope=require_choice(fields["scope"], NodeScope, f"{what}: a scope"),
        operator=operator,
        trace=tuple(str(producer) for producer in trace),
        weight_source=weight_source,
        weight_rank=weight_rank if isinstance(weight_rank, int) else None,
    )


def parse_external_tensor(entry: object, what: str) -> ExternalTensor:
    fields = require_object(entry, EXTERNAL_TENSOR_KEYS, what)
    offset_bytes = require_count(fields["offset_bytes"], f"{what}: an offset", MAX_OFFSET_BYTES)
    length_bytes = require_count(fields["length_bytes"], f"{what}: a length", MAX_OFFSET_BYTES)
    if length_bytes == 0:
        raise InspectionReportError(f"{what}: a tensor of 0 bytes")
    return ExternalTensor(
        require_file_name(fields["location"], f"{what}: a location"), offset_bytes, length_bytes
    )


def require_object(entry: object, keys: frozenset[str], what: str) -> Mapping[str, object]:
    if not isinstance(entry, dict):
        raise InspectionReportError(f"{what} is not an object")
    require_keys(entry, keys, what)
    return entry


def require_keys(document: Mapping[str, object], keys: frozenset[str], what: str) -> None:
    if set(document) != keys:
        raise InspectionReportError(
            f"{what} holds the keys {sorted(document)!a}, not {sorted(keys)}"
        )


def require_list(field: object, what: str, max_length: int) -> list[object]:
    if not isinstance(field, list):
        raise InspectionReportError(f"{what} is not a list")
    if len(field) > max_length:
        raise InspectionReportError(f"{what} holds {len(field)} entries, more than {max_length}")
    return field


def require_count(field: object, what: str, maximum: int = MAX_COUNT) -> int:
    if isinstance(field, bool) or not isinstance(field, int) or not 0 <= field <= maximum:
        raise InspectionReportError(f"{what}, {field!a}, is not a count up to {maximum}")
    return field


def require_grammar(field: object, grammar: re.Pattern[str], what: str) -> str:
    if not isinstance(field, str) or not grammar.fullmatch(field):
        raise InspectionReportError(f"{what}, {field!a}, does not match {grammar.pattern}")
    return field


def require_file_name(field: object, what: str) -> str:
    if not isinstance(field, str) or not is_file_name(field):
        raise InspectionReportError(f"{what}, {field!a}, is not a file name")
    return field


def require_choice[Choice: StrEnum](field: object, choices: type[Choice], what: str) -> Choice:
    if not isinstance(field, str) or field not in {choice.value for choice in choices}:
        choice_names = ", ".join(choice.value for choice in choices)
        raise InspectionReportError(f"{what}, {field!a}, is not one of {choice_names}")
    return choices(field)


def refuse_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    keys = [key for key, _ in pairs]
    if len(set(keys)) != len(keys):
        raise ValueError("a key appears twice in an object")
    return dict(pairs)
