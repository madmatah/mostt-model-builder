import json
import re
from collections.abc import Callable
from functools import partial

import pytest

from mostt_model_builder.domain.onnx.node_placement import (
    MAX_COUNT,
    MAX_ELEMENT_TYPE_COUNT,
    MAX_NODE_COUNT,
    PLACEMENT_MAX_SIZE_BYTES,
    NodePlacementError,
    PlacedNodes,
    PlacementSummary,
    judge_placement,
    parse_placement_summary,
    render_placement_summary,
)

CUDA_PROVIDER = "CUDAExecutionProvider"
CPU_PROVIDER = "CPUExecutionProvider"
CUDA_MATRIX_PRODUCTS = PlacedNodes(CUDA_PROVIDER, "MatMul", ("float16",), 640)
CUDA_NORMALIZATIONS = PlacedNodes(CUDA_PROVIDER, "LayerNormalization", ("float", "float16"), 24)
CPU_SHAPES = PlacedNodes(CPU_PROVIDER, "Shape", ("int64",), 12)
CPU_MASKS = PlacedNodes(CPU_PROVIDER, "Equal", ("bool", "int64"), 3)
SUMMARY = PlacementSummary((CUDA_MATRIX_PRODUCTS, CUDA_NORMALIZATIONS, CPU_SHAPES, CPU_MASKS))
WHERE_THE_GPU_RUN_PLACES = (
    "where a run on the GPU places there only nodes of int64 and bool tensors"
)
NODE_FIELDS: dict[str, object] = {
    "provider": CUDA_PROVIDER,
    "operator": "MatMul",
    "element_types": ["float16"],
    "count": 640,
}
NODE_KEYS = "['count', 'element_types', 'operator', 'provider']"


def build_summary(**changes: object) -> bytes:
    return json.dumps({"nodes": [NODE_FIELDS | changes]}).encode("ascii")


def build_summary_without(key: str) -> bytes:
    remaining_fields = {name: field for name, field in NODE_FIELDS.items() if name != key}
    return json.dumps({"nodes": [remaining_fields]}).encode("ascii")


REFUSED_SUMMARIES = {
    "a document nested 3 000 deep": (b"[" * 3000, "the document nests deeper than 32 levels"),
    "more than 4 MiB": (
        b" " * (PLACEMENT_MAX_SIZE_BYTES + 1),
        f"the placement summary holds {PLACEMENT_MAX_SIZE_BYTES + 1} bytes, beyond "
        f"{PLACEMENT_MAX_SIZE_BYTES}",
    ),
    "not json": (b"{", "the placement summary is not ASCII JSON"),
    "a key twice": (
        b'{"nodes": [], "nodes": []}',
        "the placement summary is not ASCII JSON: a key appears twice in an object",
    ),
    "bytes beyond ascii": (
        build_summary().replace(b"MatMul", b"MatMul\xc3\xa9"),
        "the placement summary is not ASCII JSON",
    ),
    "a count that is not a number": (
        build_summary(count=float("nan")),
        "the placement summary holds NaN, which is not a JSON number",
    ),
    "a list": (b"[]", "the placement summary is not an object of ['nodes']"),
    "an unknown key in the summary": (
        b'{"nodes": [], "command": "rm"}',
        "the placement summary is not an object of ['nodes']",
    ),
    "no node": (
        b'{"nodes": []}',
        f"the placement summary's nodes are not a list of 1 to {MAX_NODE_COUNT} entries",
    ),
    "nodes that are not a list": (
        b'{"nodes": {}}',
        "the placement summary's nodes are not a list of 1 to",
    ),
    "more nodes than the bound": (
        json.dumps({"nodes": [0] * (MAX_NODE_COUNT + 1)}).encode("ascii"),
        "the placement summary's nodes are not a list of 1 to",
    ),
    "a node that is not an object": (
        b'{"nodes": [1]}',
        f"a node of the placement summary is not an object of {NODE_KEYS}",
    ),
    "a node without its count": (
        build_summary_without("count"),
        f"a node of the placement summary is not an object of {NODE_KEYS}",
    ),
    "a node with an unknown key": (
        build_summary(command="rm"),
        f"a node of the placement summary is not an object of {NODE_KEYS}",
    ),
    "a provider outside its grammar": (
        build_summary(provider="CUDA"),
        "a provider, 'CUDA', does not match [A-Za-z0-9]{1,64}ExecutionProvider",
    ),
    "a provider with a space": (
        build_summary(provider="CUDA ExecutionProvider"),
        "a provider, 'CUDA ExecutionProvider', does not match",
    ),
    "an operator outside its grammar": (
        build_summary(operator="Mat-Mul"),
        "an operator, 'Mat-Mul', does not match [A-Za-z0-9_]{1,128}",
    ),
    "an element type outside its grammar": (
        build_summary(element_types=["Float"]),
        "the element types ['Float'] of a node are not a list of names",
    ),
    "an element type that is not a string": (
        build_summary(element_types=[1]),
        "the element types [1] of a node are not a list of names",
    ),
    "element types written as one string": (
        build_summary(element_types="float"),
        "the element types 'float' of a node are not a list of names",
    ),
    "more element types than the bound": (
        build_summary(element_types=["float"] * (MAX_ELEMENT_TYPE_COUNT + 1)),
        "of a node are not a list of names",
    ),
    "a count of no node": (build_summary(count=0), "the count 0 of a node is not a positive count"),
    "a negative count": (build_summary(count=-1), "the count -1 of a node is not a positive"),
    "a boolean count": (build_summary(count=True), "the count True of a node is not a positive"),
    "a fractional count": (build_summary(count=1.5), "the count 1.5 of a node is not a positive"),
    "a count beyond the bound": (
        build_summary(count=MAX_COUNT + 1),
        f"the count {MAX_COUNT + 1} of a node is not a positive count",
    ),
}
ACCEPTED_PLACEMENTS = {
    "every node on CUDA, with floating-point tensors": PlacementSummary(
        (CUDA_MATRIX_PRODUCTS, CUDA_NORMALIZATIONS)
    ),
    "nodes on the CPU with only int64 and bool tensors": PlacementSummary((CPU_SHAPES, CPU_MASKS)),
    "a CPU node without a tensor": PlacementSummary((PlacedNodes(CPU_PROVIDER, "Shape", (), 1),)),
    "both": SUMMARY,
}
REFUSED_PLACEMENTS = {
    "a CPU node of float tensors": (
        PlacementSummary(
            (*SUMMARY.nodes, PlacedNodes(CPU_PROVIDER, "Gather", ("float", "int64"), 3))
        ),
        f"nodes Gather run on the CPU with float tensors, 3 of them, {WHERE_THE_GPU_RUN_PLACES}",
    ),
    "a CPU node of two floating-point types, named in order": (
        PlacementSummary((PlacedNodes(CPU_PROVIDER, "MatMul", ("float16", "float"), 2),)),
        f"nodes MatMul run on the CPU with float tensors, 2 of them, {WHERE_THE_GPU_RUN_PLACES}",
    ),
    "a CPU node of int32 tensors": (
        PlacementSummary((PlacedNodes(CPU_PROVIDER, "Slice", ("int32",), 1),)),
        f"nodes Slice run on the CPU with int32 tensors, 1 of them, {WHERE_THE_GPU_RUN_PLACES}",
    ),
}


class TestTheNodePlacement:
    class TestTheParsePlacementSummaryFunction:
        class TestWhenTheStepWroteItsSummary:
            @pytest.fixture
            def placement_summary(self) -> PlacementSummary:
                return parse_placement_summary(render_placement_summary(SUMMARY))

            def test_should_read_the_summary_it_rendered(
                self, placement_summary: PlacementSummary
            ) -> None:
                assert placement_summary == SUMMARY

        @pytest.mark.parametrize(
            ("content", "reason"),
            list(REFUSED_SUMMARIES.values()),
            ids=list(REFUSED_SUMMARIES.keys()),
        )
        class TestWhenTheSummaryIsMalformed:
            @pytest.fixture
            def parse_act(self, content: bytes) -> Callable[[], PlacementSummary]:
                return partial(parse_placement_summary, content)

            def test_should_refuse_it_and_say_why(
                self, parse_act: Callable[[], PlacementSummary], reason: str
            ) -> None:
                with pytest.raises(NodePlacementError, match=re.escape(reason)):
                    parse_act()

    class TestTheCountNodesOnMethod:
        class TestWhenTheSummaryPlacesNodesOnCudaAndOnTheCpu:
            @pytest.fixture
            def node_counts(self) -> dict[str, int]:
                return {
                    provider: SUMMARY.count_nodes_on(provider)
                    for provider in (CUDA_PROVIDER, CPU_PROVIDER, "TensorrtExecutionProvider")
                }

            def test_should_add_the_counts_of_each_provider_and_none_for_another(
                self, node_counts: dict[str, int]
            ) -> None:
                assert node_counts == {
                    CUDA_PROVIDER: 664,
                    CPU_PROVIDER: 15,
                    "TensorrtExecutionProvider": 0,
                }

    class TestTheJudgePlacementFunction:
        @pytest.mark.parametrize(
            "summary", list(ACCEPTED_PLACEMENTS.values()), ids=list(ACCEPTED_PLACEMENTS.keys())
        )
        class TestWhenOnlyIntegerAndBooleanNodesRunOnTheCpu:
            @pytest.fixture
            def judge_act(self, summary: PlacementSummary) -> Callable[[], object]:
                return partial(judge_placement, summary)

            def test_should_accept_it(self, judge_act: Callable[[], object]) -> None:
                assert judge_act() is None

        @pytest.mark.parametrize(
            ("summary", "reason"),
            list(REFUSED_PLACEMENTS.values()),
            ids=list(REFUSED_PLACEMENTS.keys()),
        )
        class TestWhenAPartOfTheModelFellBackToTheCpu:
            @pytest.fixture
            def judge_act(self, summary: PlacementSummary) -> Callable[[], object]:
                return partial(judge_placement, summary)

            def test_should_refuse_it_since_only_shapes_and_masks_belong_there_and_name_the_nodes(
                self, judge_act: Callable[[], object], reason: str
            ) -> None:
                with pytest.raises(NodePlacementError, match=re.escape(reason)):
                    judge_act()
