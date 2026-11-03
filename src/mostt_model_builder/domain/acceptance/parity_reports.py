import json
import math
import re
from collections.abc import Sequence
from dataclasses import dataclass

from mostt_model_builder.domain.json_documents.json_depth import MAX_JSON_DEPTH, refuse_deep_json

PARITY_NAME_GRAMMAR = re.compile(r"[a-z0-9][a-z0-9-]{0,63}")
PARITY_REPORT_MAX_SIZE_BYTES = 4096
PARITY_REPORT_KEYS = frozenset({"name", "maximum_absolute_difference", "compared_count"})
# A parity compares a few tens of utterances or decoding steps; the bound only refuses a count
# that no parity of a build could reach.
MAX_COMPARED_COUNT = 1_000_000


class ParityReportError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class ParityReport:
    name: str
    maximum_absolute_difference: float
    compared_count: int


def summarize_differences(name: str, differences: Sequence[float]) -> ParityReport:
    if not differences:
        raise ParityReportError(f"the parity {name} compared nothing")
    non_finite = [difference for difference in differences if not math.isfinite(difference)]
    if non_finite:
        raise ParityReportError(
            f"the parity {name} holds a difference that is not finite, {non_finite[0]!a}"
        )
    return ParityReport(name, max(differences), len(differences))


def render_parity_report(report: ParityReport) -> bytes:
    document = {
        "name": report.name,
        "maximum_absolute_difference": report.maximum_absolute_difference,
        "compared_count": report.compared_count,
    }
    return json.dumps(document, allow_nan=False).encode("ascii")


def parse_parity_report(content: bytes, expected_name: str) -> ParityReport:
    if len(content) > PARITY_REPORT_MAX_SIZE_BYTES:
        raise ParityReportError(
            f"the parity report holds {len(content)} bytes, more than "
            f"{PARITY_REPORT_MAX_SIZE_BYTES}"
        )
    document = load_strict_json(content)
    if not isinstance(document, dict) or set(document) != PARITY_REPORT_KEYS:
        raise ParityReportError(
            f"the parity report is not an object of exactly {sorted(PARITY_REPORT_KEYS)}"
        )
    name = document["name"]
    if name != expected_name:
        raise ParityReportError(f"the parity report names the parity {name!a}, not {expected_name}")
    return ParityReport(
        name=expected_name,
        maximum_absolute_difference=parse_difference(document["maximum_absolute_difference"]),
        compared_count=parse_compared_count(document["compared_count"]),
    )


def parse_difference(field: object) -> float:
    if not isinstance(field, float) or not math.isfinite(field) or field < 0:
        raise ParityReportError(
            f"the parity report's maximum_absolute_difference, {field!a}, is not a finite "
            "decimal number of at least 0"
        )
    return field


def parse_compared_count(field: object) -> int:
    if (
        not isinstance(field, int)
        or isinstance(field, bool)
        or not 1 <= field <= MAX_COMPARED_COUNT
    ):
        raise ParityReportError(
            f"the parity report's compared_count, {field!a}, is not a whole number from 1 "
            f"to {MAX_COMPARED_COUNT}"
        )
    return field


def load_strict_json(content: bytes) -> object:
    try:
        return json.loads(
            refuse_deep_json(content.decode("ascii"), MAX_JSON_DEPTH),
            object_pairs_hook=refuse_duplicate_keys,
            parse_constant=refuse_constant,
        )
    except (UnicodeDecodeError, ValueError) as error:
        raise ParityReportError(f"the parity report is not strict JSON: {error}") from error


def refuse_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    keys = [key for key, _ in pairs]
    if len(set(keys)) != len(keys):
        raise ValueError("a key appears twice in an object")
    return dict(pairs)


def refuse_constant(constant: str) -> object:
    raise ValueError(f"{constant} is not a JSON number")
