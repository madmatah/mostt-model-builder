import json
import re
from collections.abc import Callable
from functools import partial

import pytest

from mostt_model_builder.domain.onnx.inspection_report import (
    INSPECTION_REPORT_MAX_SIZE_BYTES,
    MAX_COUNT,
    MAX_FILE_COUNT,
    MAX_LIST_LENGTH,
    MAX_OFFSET_BYTES,
    MAX_RANK,
    MAX_TRACE_LENGTH,
    ExternalTensor,
    FileInspection,
    InspectionReport,
    InspectionReportError,
    LocalFunction,
    MatrixProduct,
    NodeCount,
    NodeScope,
    Opset,
    WeightSource,
    parse_inspection_report,
    parse_inspection_request,
    render_inspection_report,
    render_inspection_request,
)

ENCODER_NAME = "encoder.onnx"
DECODER_NAME = "decoder.onnx"
FILE_NAMES = (ENCODER_NAME, DECODER_NAME)
LOCAL_DOMAIN = "mostt.local"
ENCODER = FileInspection(
    name=ENCODER_NAME,
    opsets=(Opset("", 17), Opset("com.microsoft", 1), Opset(LOCAL_DOMAIN, 1)),
    functions=(LocalFunction(LOCAL_DOMAIN, "RelativePositionAttention"),),
    nodes=(
        NodeCount(NodeScope.GRAPH, "", "MatMul", 120),
        NodeCount(NodeScope.GRAPH, LOCAL_DOMAIN, "RelativePositionAttention", 17),
        NodeCount(NodeScope.SUBGRAPH, "", "Add", 4),
        NodeCount(NodeScope.FUNCTION, "", "Softmax", 2),
    ),
    metadata_props_node_count=0,
    products=(
        MatrixProduct(NodeScope.GRAPH, "MatMul", (), WeightSource.INITIALIZER, 2),
        MatrixProduct(NodeScope.GRAPH, "Gemm", ("Transpose", "Cast"), WeightSource.INITIALIZER, 2),
        MatrixProduct(NodeScope.FUNCTION, "MatMul", ("Constant",), WeightSource.CONSTANT, None),
        MatrixProduct(NodeScope.SUBGRAPH, "MatMul", (), WeightSource.COMPUTED, None),
    ),
    external_tensors=(
        ExternalTensor("encoder.onnx.data", 0, 4096),
        ExternalTensor("encoder.onnx.data", 4096, 1024),
    ),
    training_graph_count=0,
    sparse_initializer_count=0,
)
DECODER = FileInspection(
    name=DECODER_NAME,
    opsets=(Opset("", 17),),
    functions=(),
    nodes=(NodeCount(NodeScope.GRAPH, "", "LSTM", 1),),
    metadata_props_node_count=3,
    products=(),
    external_tensors=(),
    training_graph_count=2,
    sparse_initializer_count=1,
)
REPORT = InspectionReport((ENCODER, DECODER))
WHERE = "the inspection of encoder.onnx"


def build_opset(**changes: object) -> dict[str, object]:
    return {"domain": "", "version": 17} | changes


def build_function(**changes: object) -> dict[str, object]:
    return {"domain": LOCAL_DOMAIN, "name": "RelativePositionAttention"} | changes


def build_node(**changes: object) -> dict[str, object]:
    return {"scope": "graph", "domain": "", "operator": "MatMul", "count": 120} | changes


def build_product(**changes: object) -> dict[str, object]:
    return {
        "scope": "graph",
        "operator": "MatMul",
        "trace": [],
        "weight_source": "initializer",
        "weight_rank": 2,
    } | changes


def build_external_tensor(**changes: object) -> dict[str, object]:
    return {"location": "encoder.onnx.data", "offset_bytes": 0, "length_bytes": 4096} | changes


def build_file(name: object = ENCODER_NAME, **changes: object) -> dict[str, object]:
    return {
        "name": name,
        "opsets": [build_opset()],
        "functions": [build_function()],
        "nodes": [build_node()],
        "metadata_props_node_count": 0,
        "products": [build_product()],
        "external_tensors": [build_external_tensor()],
        "training_graph_count": 0,
        "sparse_initializer_count": 0,
    } | changes


def remove_key(entry: dict[str, object], key: str) -> dict[str, object]:
    return {name: field for name, field in entry.items() if name != key}


def encode_report(*files: object, **extra_keys: object) -> bytes:
    return json.dumps({"files": list(files)} | extra_keys).encode("ascii")


def build_report(**encoder_changes: object) -> bytes:
    return encode_report(build_file(**encoder_changes), build_file(DECODER_NAME))


REFUSED_REPORTS = {
    "more than 16 MiB": (
        b" " * (INSPECTION_REPORT_MAX_SIZE_BYTES + 1),
        f"the inspection's report holds {INSPECTION_REPORT_MAX_SIZE_BYTES + 1} bytes, beyond "
        f"{INSPECTION_REPORT_MAX_SIZE_BYTES}",
    ),
    "not json": (b"{", "the inspection's report is not ASCII JSON"),
    "a key twice": (
        b'{"files": [], "files": []}',
        "the inspection's report is not ASCII JSON: a key appears twice in an object",
    ),
    "bytes beyond ascii": (
        build_report().replace(b"MatMul", b"MatMul\xc3\xa9", 1),
        "the inspection's report is not ASCII JSON",
    ),
    "a count that is not a number": (
        build_report(metadata_props_node_count=float("nan")),
        "the report holds NaN, which is not a JSON number",
    ),
    "a list": (b"[]", "the inspection's report is not a JSON object"),
    "an unknown key in the report": (
        encode_report(build_file(), build_file(DECODER_NAME), command="rm"),
        "the inspection's report holds the keys ['command', 'files'], not ['files']",
    ),
    "files that are not a list": (b'{"files": {}}', "the report's files is not a list"),
    "more files than the bound": (
        encode_report(*[0] * (MAX_FILE_COUNT + 1)),
        f"the report's files holds {MAX_FILE_COUNT + 1} entries, more than {MAX_FILE_COUNT}",
    ),
    "a file that is not an object": (
        encode_report(1, build_file(DECODER_NAME)),
        "a file of the report is not an object",
    ),
    "a file without its products": (
        encode_report(remove_key(build_file(), "products"), build_file(DECODER_NAME)),
        "a file of the report holds the keys ['external_tensors', 'functions', "
        "'metadata_props_node_count', 'name', 'nodes', 'opsets', 'sparse_initializer_count', "
        "'training_graph_count'], not",
    ),
    "a file with an unknown key": (
        build_report(command="rm"),
        "a file of the report holds the keys ['command',",
    ),
    "a file named by a path": (
        encode_report(build_file("../encoder.onnx"), build_file(DECODER_NAME)),
        "a file of the report, '../encoder.onnx', is not a file name",
    ),
    "a file whose name is not a string": (
        encode_report(build_file(7), build_file(DECODER_NAME)),
        "a file of the report, 7, is not a file name",
    ),
    "a file other than the expected ones": (
        encode_report(build_file(), build_file("joiner.onnx")),
        "the inspection's report describes ['encoder.onnx', 'joiner.onnx'], not "
        "['encoder.onnx', 'decoder.onnx']",
    ),
    "the files in another order": (
        encode_report(build_file(DECODER_NAME), build_file()),
        "the inspection's report describes ['decoder.onnx', 'encoder.onnx'], not",
    ),
    "a file missing": (
        encode_report(build_file()),
        "the inspection's report describes ['encoder.onnx'], not",
    ),
    "opsets that are not a list": (build_report(opsets={}), f"{WHERE}: its opsets is not a list"),
    "an opset with an unknown key": (
        build_report(opsets=[build_opset(command="rm")]),
        f"{WHERE}: its opsets holds the keys ['command', 'domain', 'version'], not",
    ),
    "an opset domain outside its grammar": (
        build_report(opsets=[build_opset(domain="mostt local")]),
        f"{WHERE}: its opsets: a domain, 'mostt local', does not match [A-Za-z0-9._-]{{0,128}}",
    ),
    "an opset version beyond the bound": (
        build_report(opsets=[build_opset(version=1001)]),
        f"{WHERE}: its opsets: the version 1001 is beyond 1000",
    ),
    "a boolean opset version": (
        build_report(opsets=[build_opset(version=True)]),
        f"{WHERE}: its opsets: a version, True, is not a count up to {MAX_COUNT}",
    ),
    "a function without its domain": (
        build_report(functions=[remove_key(build_function(), "domain")]),
        f"{WHERE}: its functions holds the keys ['name'], not ['domain', 'name']",
    ),
    "a function name outside its grammar": (
        build_report(functions=[build_function(name="Relative Attention")]),
        f"{WHERE}: its functions: a name, 'Relative Attention', does not match",
    ),
    "a node without its count": (
        build_report(nodes=[remove_key(build_node(), "count")]),
        f"{WHERE}: its nodes holds the keys ['domain', 'operator', 'scope'], not",
    ),
    "a count of no node": (
        build_report(nodes=[build_node(count=0)]),
        f"{WHERE}: its nodes: a count of 0 nodes",
    ),
    "a boolean count of nodes": (
        build_report(nodes=[build_node(count=True)]),
        f"{WHERE}: its nodes: a count, True, is not a count up to {MAX_COUNT}",
    ),
    "a count of nodes beyond the bound": (
        build_report(nodes=[build_node(count=MAX_COUNT + 1)]),
        f"{WHERE}: its nodes: a count, {MAX_COUNT + 1}, is not a count up to {MAX_COUNT}",
    ),
    "a fractional count of nodes": (
        build_report(nodes=[build_node(count=1.5)]),
        f"{WHERE}: its nodes: a count, 1.5, is not a count",
    ),
    "a node scope outside its enum": (
        build_report(nodes=[build_node(scope="model")]),
        f"{WHERE}: its nodes: a scope, 'model', is not one of",
    ),
    "a node domain outside its grammar": (
        build_report(nodes=[build_node(domain="ai/onnx")]),
        f"{WHERE}: its nodes: a domain, 'ai/onnx', does not match",
    ),
    "a node operator outside its grammar": (
        build_report(nodes=[build_node(operator="Mat-Mul")]),
        f"{WHERE}: its nodes: an operator, 'Mat-Mul', does not match [A-Za-z0-9_]{{1,128}}",
    ),
    "an empty node operator": (
        build_report(nodes=[build_node(operator="")]),
        f"{WHERE}: its nodes: an operator, '', does not match",
    ),
    "more nodes than the bound": (
        build_report(nodes=[0] * (MAX_LIST_LENGTH + 1)),
        f"{WHERE}: its nodes holds {MAX_LIST_LENGTH + 1} entries, more than {MAX_LIST_LENGTH}",
    ),
    "a negative count of nodes with metadata": (
        build_report(metadata_props_node_count=-1),
        f"{WHERE}: its count of nodes with metadata, -1, is not a count up to {MAX_COUNT}",
    ),
    "a boolean count of nodes with metadata": (
        build_report(metadata_props_node_count=False),
        f"{WHERE}: its count of nodes with metadata, False, is not a count",
    ),
    "a negative count of training graphs": (
        build_report(training_graph_count=-1),
        f"{WHERE}: its count of training graphs, -1, is not a count up to {MAX_COUNT}",
    ),
    "a count of training graphs beyond the bound": (
        build_report(training_graph_count=MAX_COUNT + 1),
        f"{WHERE}: its count of training graphs, {MAX_COUNT + 1}, is not a count up to",
    ),
    "a negative count of sparse initializers": (
        build_report(sparse_initializer_count=-1),
        f"{WHERE}: its count of sparse initializers, -1, is not a count up to {MAX_COUNT}",
    ),
    "a count of sparse initializers beyond the bound": (
        build_report(sparse_initializer_count=MAX_COUNT + 1),
        f"{WHERE}: its count of sparse initializers, {MAX_COUNT + 1}, is not a count up to",
    ),
    "a document nested 3 000 deep": (b"[" * 3000, "the document nests deeper than 32 levels"),
    "a trace through a producer that holds no constant": (
        build_report(products=[build_product(trace=["Transpose", "Add"])]),
        f"{WHERE}: its products: the trace passes through 'Add', which holds no constant",
    ),
    "a trace longer than the bound": (
        build_report(products=[build_product(trace=["Identity"] * (MAX_TRACE_LENGTH + 1))]),
        f"{WHERE}: its products: a trace holds {MAX_TRACE_LENGTH + 1} entries, more than "
        f"{MAX_TRACE_LENGTH}",
    ),
    "a trace that holds a list in place of a producer": (
        build_report(products=[build_product(trace=[[]])]),
        f"{WHERE}: its products: the trace passes through [], which holds no constant",
    ),
    "a trace that is not a list": (
        build_report(products=[build_product(trace="Transpose")]),
        f"{WHERE}: its products: a trace is not a list",
    ),
    "a weight source outside its enum": (
        build_report(products=[build_product(weight_source="parameter")]),
        f"{WHERE}: its products: a source, 'parameter', is not one of",
    ),
    "a rank on a constant node's weight": (
        build_report(products=[build_product(weight_source="constant", weight_rank=2)]),
        f"{WHERE}: its products: a rank for a weight that no initializer holds",
    ),
    "a rank on a computed weight": (
        build_report(products=[build_product(weight_source="computed", weight_rank=2)]),
        f"{WHERE}: its products: a rank for a weight that no initializer holds",
    ),
    "no rank on an initializer": (
        build_report(products=[build_product(weight_rank=None)]),
        f"{WHERE}: its products: a rank, None, is not a count up to {MAX_COUNT}",
    ),
    "a rank beyond the bound": (
        build_report(products=[build_product(weight_rank=MAX_RANK + 1)]),
        f"{WHERE}: its products: the rank 17 is beyond {MAX_RANK}",
    ),
    "a product that is a convolution": (
        build_report(products=[build_product(operator="Conv")]),
        f"{WHERE}: its products: Conv is not a matrix product",
    ),
    "a product scope outside its enum": (
        build_report(products=[build_product(scope="model")]),
        f"{WHERE}: its products: a scope, 'model', is not one of",
    ),
    "an external location that is a path": (
        build_report(external_tensors=[build_external_tensor(location="../x")]),
        f"{WHERE}: its external tensors: a location, '../x', is not a file name",
    ),
    "an external location in a directory": (
        build_report(external_tensors=[build_external_tensor(location="a/b")]),
        f"{WHERE}: its external tensors: a location, 'a/b', is not a file name",
    ),
    "an external tensor of no byte": (
        build_report(external_tensors=[build_external_tensor(length_bytes=0)]),
        f"{WHERE}: its external tensors: a tensor of 0 bytes",
    ),
    "an external offset beyond the bound": (
        build_report(external_tensors=[build_external_tensor(offset_bytes=MAX_OFFSET_BYTES + 1)]),
        f"{WHERE}: its external tensors: an offset, {MAX_OFFSET_BYTES + 1}, is not a count up to "
        f"{MAX_OFFSET_BYTES}",
    ),
    "a negative external offset": (
        build_report(external_tensors=[build_external_tensor(offset_bytes=-1)]),
        f"{WHERE}: its external tensors: an offset, -1, is not a count",
    ),
    "an external length beyond the bound": (
        build_report(external_tensors=[build_external_tensor(length_bytes=MAX_OFFSET_BYTES + 1)]),
        f"{WHERE}: its external tensors: a length, {MAX_OFFSET_BYTES + 1}, is not a count",
    ),
    "more external tensors than the bound": (
        build_report(external_tensors=[0] * (MAX_LIST_LENGTH + 1)),
        f"{WHERE}: its external tensors holds {MAX_LIST_LENGTH + 1} entries, more than",
    ),
}
REFUSED_REQUESTS = {
    "more than 64 KiB": (
        b" " * (64 * 1024 + 1),
        f"the request holds {64 * 1024 + 1} bytes, beyond {64 * 1024}",
    ),
    "a name that is a path": (
        b'{"files": ["../encoder.onnx"]}',
        "a file of the request, '../encoder.onnx', is not a file name",
    ),
    "a name that is not a string": (
        b'{"files": [1]}',
        "a file of the request, 1, is not a file name",
    ),
    "an unknown key": (
        b'{"files": [], "command": "rm"}',
        "the request holds the keys ['command', 'files'], not ['files']",
    ),
    "more files than the bound": (
        json.dumps({"files": ["encoder.onnx"] * (MAX_FILE_COUNT + 1)}).encode("ascii"),
        f"the request's files holds {MAX_FILE_COUNT + 1} entries, more than {MAX_FILE_COUNT}",
    ),
    "not json": (b"{", "the request is not ASCII JSON"),
}


class TestTheInspectionReport:
    class TestTheParseInspectionReportFunction:
        class TestWhenTheInspectionReportsTwoFiles:
            @pytest.fixture
            def inspection_report(self) -> InspectionReport:
                return parse_inspection_report(render_inspection_report(REPORT), FILE_NAMES)

            def test_should_read_the_report_it_rendered(
                self, inspection_report: InspectionReport
            ) -> None:
                assert inspection_report == REPORT

        class TestWhenNoFileIsExpected:
            @pytest.fixture
            def inspection_report(self) -> InspectionReport:
                return parse_inspection_report(render_inspection_report(InspectionReport(())), ())

            def test_should_read_an_empty_report(self, inspection_report: InspectionReport) -> None:
                assert inspection_report == InspectionReport(())

        @pytest.mark.parametrize(
            ("content", "reason"), list(REFUSED_REPORTS.values()), ids=list(REFUSED_REPORTS.keys())
        )
        class TestWhenTheReportIsMalformed:
            @pytest.fixture
            def parse_act(self, content: bytes) -> Callable[[], InspectionReport]:
                return partial(parse_inspection_report, content, FILE_NAMES)

            def test_should_refuse_it_and_say_why(
                self, parse_act: Callable[[], InspectionReport], reason: str
            ) -> None:
                with pytest.raises(InspectionReportError, match=re.escape(reason)):
                    parse_act()

    class TestTheParseInspectionRequestFunction:
        class TestWhenTheCoreRequestsTwoFiles:
            @pytest.fixture
            def requested_names(self) -> tuple[str, ...]:
                return parse_inspection_request(render_inspection_request(FILE_NAMES))

            def test_should_read_the_names_in_order(self, requested_names: tuple[str, ...]) -> None:
                assert requested_names == FILE_NAMES

        @pytest.mark.parametrize(
            ("content", "reason"),
            list(REFUSED_REQUESTS.values()),
            ids=list(REFUSED_REQUESTS.keys()),
        )
        class TestWhenTheRequestIsMalformed:
            @pytest.fixture
            def parse_act(self, content: bytes) -> Callable[[], tuple[str, ...]]:
                return partial(parse_inspection_request, content)

            def test_should_refuse_it_and_say_why(
                self, parse_act: Callable[[], tuple[str, ...]], reason: str
            ) -> None:
                with pytest.raises(InspectionReportError, match=re.escape(reason)):
                    parse_act()
