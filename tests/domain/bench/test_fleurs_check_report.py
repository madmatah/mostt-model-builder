import re
from collections.abc import Callable
from functools import partial

import pytest

from mostt_model_builder.domain.bench.fleurs_check_report import (
    FleursCheckReport,
    FleursCheckReportError,
    LanguageCheck,
    RecordingDuration,
    parse_fleurs_check_report,
    render_fleurs_check_report,
)

LANGUAGES = ("fr_fr", "en_us")
REPORT = FleursCheckReport(
    (
        LanguageCheck("fr_fr", (RecordingDuration("7105.wav", 150_720),)),
        LanguageCheck("en_us", (RecordingDuration("1274.wav", 202_240),)),
    )
)
FRENCH_ENTRY = (
    '{"language": "fr_fr", "utterances": '
    '[{"recording_name": "7105.wav", "duration_samples": 16000}]}'
)
ENGLISH_ENTRY = FRENCH_ENTRY.replace("fr_fr", "en_us")


def build_report(*language_entries: str) -> bytes:
    return f'{{"languages": [{", ".join(language_entries)}]}}'.encode()


REFUSED_REPORTS = {
    "not json": (b"{", "not strict JSON"),
    "a document nested 3 000 deep": (b"[" * 3000, "the document nests deeper than 32 levels"),
    "bytes beyond utf-8": (b'{"languages": ["\xff"]}', "not strict JSON"),
    "a key twice": (b'{"languages": [], "languages": []}', "a key appears twice"),
    "an unknown key": (b'{"languages": [], "command": "x"}', "holds the keys"),
    "languages that are not a list": (b'{"languages": {}}', "are not a list"),
    "a language missing": (build_report(FRENCH_ENTRY), "checks the languages ['fr_fr']"),
    "the languages in another order": (
        build_report(ENGLISH_ENTRY, FRENCH_ENTRY),
        "checks the languages ['en_us', 'fr_fr']",
    ),
    "no utterance": (
        build_report('{"language": "fr_fr", "utterances": []}', ENGLISH_ENTRY),
        "its utterances are not a list of 1 to 10000 entries",
    ),
    "a recording that is a path": (
        build_report(FRENCH_ENTRY.replace("7105.wav", "../7105.wav"), ENGLISH_ENTRY),
        "the recording '../7105.wav' does not match",
    ),
    "a recording twice": (
        build_report(
            FRENCH_ENTRY.replace(
                '"utterances": [',
                '"utterances": [{"recording_name": "7105.wav", "duration_samples": 1}, ',
            ),
            ENGLISH_ENTRY,
        ),
        "a recording is reported twice",
    ),
    "a duration of no sample": (
        build_report(FRENCH_ENTRY.replace("16000", "0"), ENGLISH_ENTRY),
        "the duration 0 is not a count of samples from 1 to 9600000",
    ),
    "a duration beyond ten minutes": (
        build_report(FRENCH_ENTRY.replace("16000", "9600001"), ENGLISH_ENTRY),
        "the duration 9600001 is not a count of samples",
    ),
    "a fractional duration": (
        build_report(FRENCH_ENTRY.replace("16000", "1.5"), ENGLISH_ENTRY),
        "the duration 1.5 is not a count of samples",
    ),
    "more than 1 MiB": (b" " * (2**20 + 1), "holds 1048577 bytes, more than 1048576"),
}


class TestTheFleursCheckReport:
    class TestTheParseFleursCheckReportFunction:
        class TestWhenTheStepReportsEveryLanguage:
            @pytest.fixture
            def report(self) -> FleursCheckReport:
                return parse_fleurs_check_report(render_fleurs_check_report(REPORT), LANGUAGES)

            def test_should_read_the_durations_it_rendered(self, report: FleursCheckReport) -> None:
                assert report == REPORT

            def test_should_give_each_languages_duration_in_seconds(
                self, report: FleursCheckReport
            ) -> None:
                assert [language.duration_s for language in report.languages] == pytest.approx(
                    [9.42, 12.64]
                )

        @pytest.mark.parametrize(
            ("content", "reason"), list(REFUSED_REPORTS.values()), ids=list(REFUSED_REPORTS.keys())
        )
        class TestWhenTheReportIsMalformed:
            @pytest.fixture
            def parse_act(self, content: bytes) -> Callable[[], FleursCheckReport]:
                return partial(parse_fleurs_check_report, content, LANGUAGES)

            def test_should_refuse_it_and_say_why(
                self, parse_act: Callable[[], FleursCheckReport], reason: str
            ) -> None:
                with pytest.raises(FleursCheckReportError, match=re.escape(reason)):
                    parse_act()
