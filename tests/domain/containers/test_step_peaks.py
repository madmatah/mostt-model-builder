import re
from collections.abc import Callable
from functools import partial

import pytest

from mostt_model_builder.domain.containers.step_peaks import (
    StepPeaks,
    StepPeaksError,
    parse_step_peaks,
    render_step_peaks,
)

PEAKS = StepPeaks(memory_peak_bytes=22_020_096, process_count_peak=6)
REFUSED_PEAKS = {
    "a list": (b"[1, 2]", "not an object of exactly"),
    "a document nested 3 000 deep": (b"[" * 3000, "the document nests deeper than 32 levels"),
    "a missing key": (b'{"memory_peak_bytes": 1}', "not an object of exactly"),
    "an unknown key": (
        b'{"memory_peak_bytes": 1, "process_count_peak": 1, "command": "rm"}',
        "not an object of exactly",
    ),
    "a key twice": (
        b'{"memory_peak_bytes": 1, "memory_peak_bytes": 2, "process_count_peak": 1}',
        "a key appears twice",
    ),
    "a negative count": (
        b'{"memory_peak_bytes": -1, "process_count_peak": 1}',
        "memory_peak_bytes, -1, is not a whole number",
    ),
    "a fraction": (
        b'{"memory_peak_bytes": 1.5, "process_count_peak": 1}',
        "memory_peak_bytes, 1.5, is not a whole number",
    ),
    "a boolean": (
        b'{"memory_peak_bytes": 1, "process_count_peak": true}',
        "process_count_peak, True, is not a whole number",
    ),
    "a memory beyond the largest limit": (
        b'{"memory_peak_bytes": 60129542145, "process_count_peak": 1}',
        "memory_peak_bytes, 60129542145, is not a whole number from 0 to 60129542144",
    ),
    "processes beyond the largest limit": (
        b'{"memory_peak_bytes": 1, "process_count_peak": 8193}',
        "process_count_peak, 8193, is not a whole number from 0 to 8192",
    ),
    "not a number": (b'{"memory_peak_bytes": NaN, "process_count_peak": 1}', "NaN is not"),
    "bytes beyond ascii": (b'{"memory_peak_bytes\xc3\xa9": 1}', "not strict JSON"),
    "more than 4 KiB": (b" " * 4097, "the peaks hold 4097 bytes, more than 4096"),
}


class TestTheStepPeaks:
    class TestTheParseStepPeaksFunction:
        class TestWhenAStepWroteItsPeaks:
            @pytest.fixture
            def peaks(self) -> StepPeaks:
                return parse_step_peaks(render_step_peaks(PEAKS))

            def test_should_read_its_memory_and_process_peaks(self, peaks: StepPeaks) -> None:
                assert peaks == PEAKS

            def test_should_give_the_memory_in_mib(self, peaks: StepPeaks) -> None:
                assert peaks.memory_peak_mib == pytest.approx(21.0)

        @pytest.mark.parametrize(
            ("content", "reason"), list(REFUSED_PEAKS.values()), ids=list(REFUSED_PEAKS.keys())
        )
        class TestWhenThePeaksAreMalformed:
            @pytest.fixture
            def parse_act(self, content: bytes) -> Callable[[], StepPeaks]:
                return partial(parse_step_peaks, content)

            def test_should_refuse_them_and_say_why(
                self, parse_act: Callable[[], StepPeaks], reason: str
            ) -> None:
                with pytest.raises(StepPeaksError, match=re.escape(reason)):
                    parse_act()
