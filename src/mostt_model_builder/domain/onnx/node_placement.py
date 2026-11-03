import json
import re
from collections.abc import Mapping
from dataclasses import dataclass

from mostt_model_builder.domain.json_documents.json_depth import MAX_JSON_DEPTH, refuse_deep_json

PLACEMENT_FILE_NAME = "placement.json"
PLACEMENT_MAX_SIZE_BYTES = 4 * 2**20
MAX_NODE_COUNT = 100_000
CPU_PROVIDER = "CPUExecutionProvider"
CPU_ELEMENT_TYPES = frozenset({"int64", "bool"})
PROVIDER_GRAMMAR = re.compile(r"[A-Za-z0-9]{1,64}ExecutionProvider")
OPERATOR_GRAMMAR = re.compile(r"[A-Za-z0-9_]{1,128}")
ELEMENT_TYPE_GRAMMAR = re.compile(r"[a-z0-9_]{1,32}")
SUMMARY_KEYS = frozenset({"nodes"})
NODE_KEYS = frozenset({"provider", "operator", "element_types", "count"})
MAX_ELEMENT_TYPE_COUNT = 16
MAX_COUNT = 2**31


class NodePlacementError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class PlacedNodes:
    provider: str
    operator: str
    element_types: tuple[str, ...]
    count: int


@dataclass(frozen=True, slots=True)
class PlacementSummary:
    nodes: tuple[PlacedNodes, ...]

    def count_nodes_on(self, provider: str) -> int:
        return sum(placed.count for placed in self.nodes if placed.provider == provider)


def render_placement_summary(summary: PlacementSummary) -> bytes:
    document = {
        "nodes": [
            {
                "provider": placed.provider,
                "operator": placed.operator,
                "element_types": list(placed.element_types),
                "count": placed.count,
            }
            for placed in summary.nodes
        ]
    }
    return json.dumps(document, indent=1).encode("ascii")


def parse_placement_summary(content: bytes) -> PlacementSummary:
    if len(content) > PLACEMENT_MAX_SIZE_BYTES:
        raise NodePlacementError(
            f"the placement summary holds {len(content)} bytes, beyond {PLACEMENT_MAX_SIZE_BYTES}"
        )
    try:
        document = json.loads(
            refuse_deep_json(content.decode("ascii"), MAX_JSON_DEPTH),
            parse_constant=refuse_constant,
            object_pairs_hook=refuse_duplicate_keys,
        )
    except (UnicodeDecodeError, ValueError) as error:
        raise NodePlacementError(f"the placement summary is not ASCII JSON: {error}") from error
    if not isinstance(document, dict) or set(document) != SUMMARY_KEYS:
        raise NodePlacementError(
            f"the placement summary is not an object of {sorted(SUMMARY_KEYS)}"
        )
    entries = document["nodes"]
    if not isinstance(entries, list) or not entries or len(entries) > MAX_NODE_COUNT:
        raise NodePlacementError(
            f"the placement summary's nodes are not a list of 1 to {MAX_NODE_COUNT} entries"
        )
    return PlacementSummary(tuple(parse_placed_nodes(entry) for entry in entries))


def refuse_constant(name: str) -> object:
    raise NodePlacementError(f"the placement summary holds {name}, which is not a JSON number")


def parse_placed_nodes(entry: object) -> PlacedNodes:
    if not isinstance(entry, dict) or set(entry) != NODE_KEYS:
        raise NodePlacementError(
            f"a node of the placement summary is not an object of {sorted(NODE_KEYS)}"
        )
    fields: Mapping[str, object] = entry
    element_types = fields["element_types"]
    if (
        not isinstance(element_types, list)
        or len(element_types) > MAX_ELEMENT_TYPE_COUNT
        or not all(
            isinstance(element_type, str) and ELEMENT_TYPE_GRAMMAR.fullmatch(element_type)
            for element_type in element_types
        )
    ):
        raise NodePlacementError(
            f"the element types {element_types!a} of a node are not a list of names"
        )
    count = fields["count"]
    if isinstance(count, bool) or not isinstance(count, int) or not 0 < count <= MAX_COUNT:
        raise NodePlacementError(f"the count {count!a} of a node is not a positive count")
    return PlacedNodes(
        provider=require_grammar(fields["provider"], PROVIDER_GRAMMAR, "a provider"),
        operator=require_grammar(fields["operator"], OPERATOR_GRAMMAR, "an operator"),
        element_types=tuple(str(element_type) for element_type in element_types),
        count=count,
    )


def require_grammar(field: object, grammar: re.Pattern[str], what: str) -> str:
    if not isinstance(field, str) or not grammar.fullmatch(field):
        raise NodePlacementError(f"{what}, {field!a}, does not match {grammar.pattern}")
    return field


def judge_placement(summary: PlacementSummary) -> None:
    for placed in summary.nodes:
        if placed.provider != CPU_PROVIDER:
            continue
        foreign_types = sorted(set(placed.element_types) - CPU_ELEMENT_TYPES)
        if foreign_types:
            raise NodePlacementError(
                f"nodes {placed.operator} run on the CPU with {foreign_types[0]} tensors, "
                f"{placed.count} of them, where a run on the GPU places there only nodes of "
                "int64 and bool tensors"
            )


def refuse_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    keys = [key for key, _ in pairs]
    if len(set(keys)) != len(keys):
        raise ValueError("a key appears twice in an object")
    return dict(pairs)
