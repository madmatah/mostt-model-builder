import json
from dataclasses import dataclass

from mostt_model_builder.domain.containers.container_steps import (
    MAX_MEMORY_MIB,
    MAX_PROCESS_COUNT,
    PEAKS_MAX_SIZE_BYTES,
)
from mostt_model_builder.domain.json_documents.json_depth import MAX_JSON_DEPTH, refuse_deep_json

PEAKS_KEYS = ("memory_peak_bytes", "process_count_peak")
BYTES_PER_MIB = 2**20
# No container exceeds its own limit, and no limit exceeds the ceilings of the step table.
MAX_MEMORY_PEAK_BYTES = MAX_MEMORY_MIB * BYTES_PER_MIB
MAX_PROCESS_COUNT_PEAK = MAX_PROCESS_COUNT


class StepPeaksError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class StepPeaks:
    memory_peak_bytes: int
    process_count_peak: int

    @property
    def memory_peak_mib(self) -> float:
        return self.memory_peak_bytes / BYTES_PER_MIB


def render_step_peaks(peaks: StepPeaks) -> bytes:
    document = {
        "memory_peak_bytes": peaks.memory_peak_bytes,
        "process_count_peak": peaks.process_count_peak,
    }
    return json.dumps(document).encode("ascii")


def parse_step_peaks(content: bytes) -> StepPeaks:
    if len(content) > PEAKS_MAX_SIZE_BYTES:
        raise StepPeaksError(
            f"the peaks hold {len(content)} bytes, more than {PEAKS_MAX_SIZE_BYTES}"
        )
    try:
        document = json.loads(
            refuse_deep_json(content.decode("ascii"), MAX_JSON_DEPTH),
            object_pairs_hook=refuse_duplicate_keys,
            parse_constant=refuse_constant,
        )
    except (UnicodeDecodeError, ValueError) as error:
        raise StepPeaksError(f"the peaks are not strict JSON: {error}") from error
    if not isinstance(document, dict) or sorted(document) != sorted(PEAKS_KEYS):
        raise StepPeaksError(f"the peaks are not an object of exactly {list(PEAKS_KEYS)}")
    return StepPeaks(
        memory_peak_bytes=parse_count(
            document["memory_peak_bytes"], MAX_MEMORY_PEAK_BYTES, "memory_peak_bytes"
        ),
        process_count_peak=parse_count(
            document["process_count_peak"], MAX_PROCESS_COUNT_PEAK, "process_count_peak"
        ),
    )


def parse_count(field: object, maximum: int, key: str) -> int:
    if not isinstance(field, int) or isinstance(field, bool) or not 0 <= field <= maximum:
        raise StepPeaksError(
            f"the peaks' {key}, {field!a}, is not a whole number from 0 to {maximum}"
        )
    return field


def refuse_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    keys = [key for key, _ in pairs]
    if len(set(keys)) != len(keys):
        raise ValueError("a key appears twice in an object")
    return dict(pairs)


def refuse_constant(constant: str) -> object:
    raise ValueError(f"{constant} is not a JSON number")
