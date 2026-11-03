import re
from collections.abc import Callable
from functools import partial

import pytest

from mostt_model_builder.domain.acceptance.parity_reports import (
    ParityReport,
    ParityReportError,
    parse_parity_report,
    render_parity_report,
    summarize_differences,
)

PARITY_NAME = "encoder-onnx-against-nemo"
REPORT = ParityReport(PARITY_NAME, 4.18e-06, 20)
ZERO_DIFFERENCE_REPORT = ParityReport(PARITY_NAME, 0.0, 20)


def build_report(difference: str = "4.18e-06", count: str = "20", name: str = PARITY_NAME) -> bytes:
    return (
        f'{{"name": "{name}", "maximum_absolute_difference": {difference}, '
        f'"compared_count": {count}}}'
    ).encode("ascii")


REFUSED_REPORTS = {
    "a list": (b"[1, 2]", "not an object of exactly"),
    "a missing key": (
        b'{"name": "encoder-onnx-against-nemo", "compared_count": 20}',
        "not an object of exactly",
    ),
    "an unknown key": (
        build_report()[:-1] + b', "command": "rm"}',
        "not an object of exactly",
    ),
    "a key twice": (
        build_report()[:-1] + b', "compared_count": 21}',
        "a key appears twice",
    ),
    "another parity's name": (
        build_report(name="decoder-onnx-against-nemo"),
        "names the parity 'decoder-onnx-against-nemo', not encoder-onnx-against-nemo",
    ),
    "a negative difference": (
        build_report(difference="-1e-06"),
        "maximum_absolute_difference, -1e-06, is not a finite decimal number of at least 0",
    ),
    "an infinite difference written as an overflow": (
        build_report(difference="1e999"),
        "maximum_absolute_difference, inf, is not a finite",
    ),
    "a difference that is not a number": (build_report(difference="NaN"), "NaN is not"),
    "an infinite difference": (build_report(difference="Infinity"), "Infinity is not"),
    "a difference written as a whole number": (
        build_report(difference="0"),
        "maximum_absolute_difference, 0, is not a finite decimal number",
    ),
    "a difference written as a string": (
        build_report(difference='"4e-06"'),
        "maximum_absolute_difference, '4e-06', is not a finite decimal number",
    ),
    "no utterance compared": (
        build_report(count="0"),
        "compared_count, 0, is not a whole number from 1 to 1000000",
    ),
    "a count beyond its bound": (
        build_report(count="1000001"),
        "compared_count, 1000001, is not a whole number from 1 to 1000000",
    ),
    "a fractional count": (build_report(count="20.5"), "compared_count, 20.5, is not a whole"),
    "a boolean count": (build_report(count="true"), "compared_count, True, is not a whole"),
    "bytes beyond ascii": (
        build_report().replace(b"encoder", b"encoder\xc3\xa9"),
        "the parity report is not strict JSON",
    ),
    "more than 4 KiB": (b" " * 4097, "the parity report holds 4097 bytes, more than 4096"),
    "a document nested 3 000 deep": (b"[" * 3000, "the document nests deeper than 32 levels"),
}
DIFFERENCES = (1.2e-06, 4.18e-06, 0.0)
NOT_A_NUMBER = float("nan")
INFINITY = float("inf")
REFUSED_DIFFERENCES = {
    "a difference that is not a number first": (
        (NOT_A_NUMBER, *DIFFERENCES),
        "the parity encoder-onnx-against-nemo holds a difference that is not finite, nan",
    ),
    "a difference that is not a number last": (
        (*DIFFERENCES, NOT_A_NUMBER),
        "the parity encoder-onnx-against-nemo holds a difference that is not finite, nan",
    ),
    "an infinite difference": (
        (*DIFFERENCES, INFINITY),
        "the parity encoder-onnx-against-nemo holds a difference that is not finite, inf",
    ),
    "no difference": ((), "the parity encoder-onnx-against-nemo compared nothing"),
}


class TestTheParityReports:
    class TestTheParseParityReportFunction:
        class TestWhenAStepWroteTheReportOfItsParity:
            @pytest.fixture
            def parity_report(self) -> ParityReport:
                return parse_parity_report(render_parity_report(REPORT), PARITY_NAME)

            def test_should_read_its_name_its_difference_and_its_count(
                self, parity_report: ParityReport
            ) -> None:
                assert parity_report == REPORT

        class TestWhenTheOutputsMatchExactly:
            @pytest.fixture
            def parity_report(self) -> ParityReport:
                return parse_parity_report(
                    render_parity_report(ZERO_DIFFERENCE_REPORT), PARITY_NAME
                )

            def test_should_read_a_difference_of_zero(self, parity_report: ParityReport) -> None:
                assert parity_report.maximum_absolute_difference == 0.0

        @pytest.mark.parametrize(
            ("content", "reason"), list(REFUSED_REPORTS.values()), ids=list(REFUSED_REPORTS.keys())
        )
        class TestWhenTheReportIsMalformed:
            @pytest.fixture
            def parse_act(self, content: bytes) -> Callable[[], ParityReport]:
                return partial(parse_parity_report, content, PARITY_NAME)

            def test_should_refuse_it_and_say_why(
                self, parse_act: Callable[[], ParityReport], reason: str
            ) -> None:
                with pytest.raises(ParityReportError, match=re.escape(reason)):
                    parse_act()

    class TestTheSummarizeDifferencesFunction:
        class TestWhenEveryDifferenceIsFinite:
            @pytest.fixture
            def parity_report(self) -> ParityReport:
                return summarize_differences(PARITY_NAME, DIFFERENCES)

            def test_should_report_their_maximum_and_their_count(
                self, parity_report: ParityReport
            ) -> None:
                assert parity_report == ParityReport(PARITY_NAME, 4.18e-06, 3)

        @pytest.mark.parametrize(
            ("differences", "reason"),
            list(REFUSED_DIFFERENCES.values()),
            ids=list(REFUSED_DIFFERENCES.keys()),
        )
        class TestWhenADifferenceCannotBeTrusted:
            @pytest.fixture
            def summarize_act(self, differences: tuple[float, ...]) -> Callable[[], ParityReport]:
                return partial(summarize_differences, PARITY_NAME, differences)

            def test_should_refuse_them_since_max_would_drop_a_difference_that_is_not_a_number(
                self, summarize_act: Callable[[], ParityReport], reason: str
            ) -> None:
                with pytest.raises(ParityReportError, match=f"^{re.escape(reason)}$"):
                    summarize_act()
