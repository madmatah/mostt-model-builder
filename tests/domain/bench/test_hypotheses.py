import re
from collections.abc import Callable
from dataclasses import replace
from functools import partial
from typing import cast

import pytest

from mostt_model_builder.domain.bench.hypotheses import (
    MAX_DECODING_TIME_MS,
    MAX_LINE_COUNT,
    MAX_LINE_SIZE_BYTES,
    MAX_TEXT_LENGTH,
    MAX_TOKEN_COUNT,
    MAX_TOKEN_ID,
    RUN_LINES_MAX_SIZE_BYTES,
    Hypothesis,
    HypothesisError,
    RunLine,
    judge_hypotheses,
    judge_hypothesis,
    parse_run_lines,
)

LANGUAGES = ("fr_fr", "en_us")
FRENCH_HYPOTHESIS = Hypothesis(
    "fr_fr", "7105.wav", (12, 7, 30, 4), "le chat dort ici", "le chat dort ici", 0, 812.5
)
ENGLISH_HYPOTHESIS = Hypothesis(
    "en_us", "1274.wav", (0, 9, 0, 3, 5), "<unk>the<unk> cat sleeps", "the cat sleeps", 2, 640.0
)
FRENCH_UTTERANCE_IN_ENGLISH = replace(ENGLISH_HYPOTHESIS, utterance_id="7105.wav")
TWO_LINES = b'{"utterance_id": "7105.wav", "tokens": [12, 7]}\n{"utterance_id": "1274.wav"}\n'
LINE_PREFIX = '{"text": "'
LINE_SUFFIX = '"}'
FILLED_LINE_AT_ITS_BOUND = (
    LINE_PREFIX + "a" * (MAX_LINE_SIZE_BYTES - len(LINE_PREFIX) - len(LINE_SUFFIX)) + LINE_SUFFIX
).encode("ascii")

ACCEPTED_LINES: dict[str, tuple[bytes, tuple[RunLine, ...]]] = {
    "a text in utf-8 beyond ascii": (
        '{"text": "l\'été déjà"}\n'.encode(),
        ({"text": "l'été déjà"},),
    ),
    "a line of exactly its size bound": (
        FILLED_LINE_AT_ITS_BOUND + b"\n",
        ({"text": "a" * (MAX_LINE_SIZE_BYTES - len(LINE_PREFIX) - len(LINE_SUFFIX))},),
    ),
    "exactly the bound of lines": (b"{}\n" * MAX_LINE_COUNT, ({},) * MAX_LINE_COUNT),
}

REFUSED_LINES = {
    "a line nested 3 000 deep": (
        b"[" * 3000 + b"\n",
        "line 1 is not a JSON object: the document nests deeper than 32 levels",
    ),
    "more than 16 MiB": (
        b" " * (RUN_LINES_MAX_SIZE_BYTES + 1),
        f"the lines hold {RUN_LINES_MAX_SIZE_BYTES + 1} bytes, beyond {RUN_LINES_MAX_SIZE_BYTES}",
    ),
    "no line feed after the last line": (
        b'{"utterance_id": "7105.wav"}',
        "the lines do not end with a line feed",
    ),
    "no line at all": (b"", "the lines do not end with a line feed"),
    "a key twice in a line": (
        b'{"language": "fr_fr", "language": "en_us"}\n',
        "line 1 is not a JSON object: a key appears twice in an object",
    ),
    "more lines than any run writes": (
        b"{}\n" * (MAX_LINE_COUNT + 1),
        f"the run wrote {MAX_LINE_COUNT + 1} lines, more than {MAX_LINE_COUNT}",
    ),
    "a line beyond its size bound": (
        b"{}\n" + b" " + FILLED_LINE_AT_ITS_BOUND + b"\n",
        f"line 2 holds more than {MAX_LINE_SIZE_BYTES} bytes",
    ),
    "a line that is a list": (b"{}\n[1, 2]\n", "line 2 is not a JSON object"),
    "a line that is a string": (b'"7105.wav"\n', "line 1 is not a JSON object"),
    "an empty line": (b"{}\n\n{}\n", "line 2 is not a JSON object"),
    "a line that is not json": (b"{}\n{}\n{\n", "line 3 is not a JSON object"),
    "a time that is not a number": (
        b'{"decoding_time_ms": NaN}\n',
        "a line holds NaN, which is not a JSON number",
    ),
    "an infinite time": (
        b'{"decoding_time_ms": Infinity}\n',
        "a line holds Infinity, which is not a JSON number",
    ),
    "a negative infinite time": (
        b'{"decoding_time_ms": -Infinity}\n',
        "a line holds -Infinity, which is not a JSON number",
    ),
    "bytes beyond utf-8": (b'{"text": "\xff"}\n', "line 1 is not a JSON object"),
}

ACCEPTED_HYPOTHESES = {
    "an empty text, the hypothesis of a silence": replace(
        FRENCH_HYPOTHESIS, token_ids=(), text="", text_without_unknown_tokens=""
    ),
    "a text of exactly its length bound": replace(
        FRENCH_HYPOTHESIS,
        text="a" * MAX_TEXT_LENGTH,
        text_without_unknown_tokens="a" * MAX_TEXT_LENGTH,
    ),
    "a text with accents, punctuation and a no-break space": replace(
        FRENCH_HYPOTHESIS,
        text="l\u2019été\u00a0? déjà !",
        text_without_unknown_tokens="l\u2019été\u00a0? déjà !",
    ),
    "unknown tokens that the text without them leaves out": ENGLISH_HYPOTHESIS,
    "exactly the bound of tokens": replace(FRENCH_HYPOTHESIS, token_ids=(7,) * MAX_TOKEN_COUNT),
    "the highest token id": replace(FRENCH_HYPOTHESIS, token_ids=(MAX_TOKEN_ID,)),
    "a decoding time of zero": replace(FRENCH_HYPOTHESIS, decoding_time_ms=0.0),
    "a decoding time at its bound": replace(
        FRENCH_HYPOTHESIS, decoding_time_ms=MAX_DECODING_TIME_MS
    ),
    "a decoding time written as a whole number": replace(FRENCH_HYPOTHESIS, decoding_time_ms=812),
}

WHERE = "the hypothesis of '7105.wav'"
REFUSED_HYPOTHESES = {
    "a language outside its run": (
        replace(FRENCH_HYPOTHESIS, language="de_de"),
        f"{WHERE} is in 'de_de', not a language of its run",
    ),
    "an utterance id that is a path": (
        replace(FRENCH_HYPOTHESIS, utterance_id="../7105.wav"),
        r"the hypothesis of '../7105.wav': its utterance id does not match [0-9]{1,24}\.wav",
    ),
    "an utterance id without its suffix": (
        replace(FRENCH_HYPOTHESIS, utterance_id="7105"),
        r"its utterance id does not match [0-9]{1,24}\.wav",
    ),
    "an utterance id beyond the digits of a recording": (
        replace(FRENCH_HYPOTHESIS, utterance_id="1" * 25 + ".wav"),
        r"its utterance id does not match [0-9]{1,24}\.wav",
    ),
    "a text beyond its length bound": (
        replace(FRENCH_HYPOTHESIS, text="a" * (MAX_TEXT_LENGTH + 1)),
        f"{WHERE} holds more than {MAX_TEXT_LENGTH} characters",
    ),
    "a zero-width space": (
        replace(FRENCH_HYPOTHESIS, text="le\u200bchat"),
        f"{WHERE} holds a character a reader cannot see",
    ),
    "a right-to-left override": (
        replace(FRENCH_HYPOTHESIS, text="le \u202echat"),
        f"{WHERE} holds a character a reader cannot see",
    ),
    "a bell, a control character": (
        replace(FRENCH_HYPOTHESIS, text="le chat\x07"),
        f"{WHERE} holds a character a reader cannot see",
    ),
    "a line feed that would split a report's line": (
        replace(FRENCH_HYPOTHESIS, text="le chat\ndort"),
        f"{WHERE} holds a character a reader cannot see",
    ),
    "a line separator": (
        replace(FRENCH_HYPOTHESIS, text="le chat\u2028dort"),
        f"{WHERE} holds a character a reader cannot see",
    ),
    "a paragraph separator": (
        replace(FRENCH_HYPOTHESIS, text="le chat\u2029dort"),
        f"{WHERE} holds a character a reader cannot see",
    ),
    "a private-use character": (
        replace(FRENCH_HYPOTHESIS, text="le chat\ue000"),
        f"{WHERE} holds a character a reader cannot see",
    ),
    "an unassigned code point": (
        replace(FRENCH_HYPOTHESIS, text="le chat\u0378"),
        f"{WHERE} holds a character a reader cannot see",
    ),
    "a lone surrogate": (
        replace(FRENCH_HYPOTHESIS, text="le chat\ud800"),
        f"{WHERE} holds a character a reader cannot see",
    ),
    "token ids in a list rather than a tuple": (
        replace(FRENCH_HYPOTHESIS, token_ids=cast(tuple[int, ...], [12, 7])),
        f"{WHERE}: its token ids are not a tuple",
    ),
    "more tokens than its bound": (
        replace(FRENCH_HYPOTHESIS, token_ids=(7,) * (MAX_TOKEN_COUNT + 1)),
        f"{WHERE} holds more than {MAX_TOKEN_COUNT} tokens",
    ),
    "a negative token id": (
        replace(FRENCH_HYPOTHESIS, token_ids=(12, -1)),
        f"{WHERE} holds the token id -1, not a whole number from 0 to {MAX_TOKEN_ID}",
    ),
    "a token id beyond its bound": (
        replace(FRENCH_HYPOTHESIS, token_ids=(MAX_TOKEN_ID + 1,)),
        f"{WHERE} holds the token id {MAX_TOKEN_ID + 1}, not a whole number",
    ),
    "a boolean token id": (
        replace(FRENCH_HYPOTHESIS, token_ids=(True,)),
        f"{WHERE} holds the token id True, not a whole number",
    ),
    "a token id written as a string": (
        replace(FRENCH_HYPOTHESIS, token_ids=cast(tuple[int, ...], ("12",))),
        f"{WHERE} holds the token id '12', not a whole number",
    ),
    "a text without its unknown tokens beyond the length bound": (
        replace(ENGLISH_HYPOTHESIS, text_without_unknown_tokens="a" * (MAX_TEXT_LENGTH + 1)),
        "the hypothesis of '1274.wav', without its unknown tokens, holds more than "
        f"{MAX_TEXT_LENGTH} characters",
    ),
    "a text without its unknown tokens that holds a zero-width space": (
        replace(ENGLISH_HYPOTHESIS, text_without_unknown_tokens="the\u200bcat"),
        "without its unknown tokens, holds a character a reader cannot see",
    ),
    "more unknown tokens than tokens": (
        replace(ENGLISH_HYPOTHESIS, unknown_token_count=6),
        "the hypothesis of '1274.wav' counts 6 unknown tokens among 5 tokens",
    ),
    "two texts that differ without an unknown token": (
        replace(FRENCH_HYPOTHESIS, text_without_unknown_tokens="le chat dort"),
        f"{WHERE} counts no unknown token, and its two texts differ",
    ),
    "a negative count of unknown tokens": (
        replace(FRENCH_HYPOTHESIS, unknown_token_count=-1),
        f"{WHERE} counts -1 unknown tokens, not a count",
    ),
    "a boolean count of unknown tokens": (
        replace(FRENCH_HYPOTHESIS, unknown_token_count=True),
        f"{WHERE} counts True unknown tokens, not a count",
    ),
    "a fractional count of unknown tokens": (
        replace(FRENCH_HYPOTHESIS, unknown_token_count=cast(int, 1.5)),
        f"{WHERE} counts 1.5 unknown tokens, not a count",
    ),
    "a negative decoding time": (
        replace(FRENCH_HYPOTHESIS, decoding_time_ms=-1.0),
        f"{WHERE} took -1.0 ms, outside 0 to 3600000",
    ),
    "a decoding time beyond its bound": (
        replace(FRENCH_HYPOTHESIS, decoding_time_ms=MAX_DECODING_TIME_MS + 1),
        f"{WHERE} took 3600001.0 ms, outside 0 to 3600000",
    ),
    "a decoding time that is not a number": (
        replace(FRENCH_HYPOTHESIS, decoding_time_ms=float("nan")),
        f"{WHERE} took nan ms, outside 0 to 3600000",
    ),
    "an infinite decoding time": (
        replace(FRENCH_HYPOTHESIS, decoding_time_ms=float("inf")),
        f"{WHERE} took inf ms, outside 0 to 3600000",
    ),
    "a boolean decoding time": (
        replace(FRENCH_HYPOTHESIS, decoding_time_ms=True),
        f"{WHERE} took True ms, outside 0 to 3600000",
    ),
    "a decoding time written as a string": (
        replace(FRENCH_HYPOTHESIS, decoding_time_ms=cast(float, "812.5")),
        f"{WHERE} took '812.5' ms, outside 0 to 3600000",
    ),
}


class TestTheHypotheses:
    class TestTheParseRunLinesFunction:
        class TestWhenTheRunWroteTwoLines:
            @pytest.fixture
            def run_lines(self) -> tuple[RunLine, ...]:
                return parse_run_lines(TWO_LINES)

            def test_should_read_each_line_as_an_object_in_order(
                self, run_lines: tuple[RunLine, ...]
            ) -> None:
                assert run_lines == (
                    {"utterance_id": "7105.wav", "tokens": [12, 7]},
                    {"utterance_id": "1274.wav"},
                )

        @pytest.mark.parametrize(
            ("content", "expected_lines"),
            list(ACCEPTED_LINES.values()),
            ids=list(ACCEPTED_LINES.keys()),
        )
        class TestWhenTheLinesReachABound:
            @pytest.fixture
            def run_lines(self, content: bytes) -> tuple[RunLine, ...]:
                return parse_run_lines(content)

            def test_should_read_them(
                self, run_lines: tuple[RunLine, ...], expected_lines: tuple[RunLine, ...]
            ) -> None:
                assert run_lines == expected_lines

        @pytest.mark.parametrize(
            ("content", "reason"), list(REFUSED_LINES.values()), ids=list(REFUSED_LINES.keys())
        )
        class TestWhenTheLinesAreMalformed:
            @pytest.fixture
            def parse_act(self, content: bytes) -> Callable[[], tuple[RunLine, ...]]:
                return partial(parse_run_lines, content)

            def test_should_refuse_them_and_say_why(
                self, parse_act: Callable[[], tuple[RunLine, ...]], reason: str
            ) -> None:
                with pytest.raises(HypothesisError, match=re.escape(reason)):
                    parse_act()

    class TestTheJudgeHypothesesFunction:
        class TestWhenEveryHypothesisHoldsItsRules:
            @pytest.fixture
            def judged_hypotheses(self) -> tuple[Hypothesis, ...]:
                return judge_hypotheses(
                    [ENGLISH_HYPOTHESIS, FRENCH_HYPOTHESIS, FRENCH_UTTERANCE_IN_ENGLISH], LANGUAGES
                )

            def test_should_return_them_unchanged_and_in_order(
                self, judged_hypotheses: tuple[Hypothesis, ...]
            ) -> None:
                assert judged_hypotheses == (
                    ENGLISH_HYPOTHESIS,
                    FRENCH_HYPOTHESIS,
                    FRENCH_UTTERANCE_IN_ENGLISH,
                )

        @pytest.mark.parametrize(
            "hypothesis", list(ACCEPTED_HYPOTHESES.values()), ids=list(ACCEPTED_HYPOTHESES.keys())
        )
        class TestWhenAHypothesisReachesABound:
            @pytest.fixture
            def judged_hypotheses(self, hypothesis: Hypothesis) -> tuple[Hypothesis, ...]:
                return judge_hypotheses([hypothesis], LANGUAGES)

            def test_should_accept_it(
                self, judged_hypotheses: tuple[Hypothesis, ...], hypothesis: Hypothesis
            ) -> None:
                assert judged_hypotheses == (hypothesis,)

        class TestWhenAnUtteranceOfALanguageHasTwoHypotheses:
            @pytest.fixture
            def judge_act(self) -> Callable[[], tuple[Hypothesis, ...]]:
                return partial(
                    judge_hypotheses,
                    [
                        FRENCH_HYPOTHESIS,
                        ENGLISH_HYPOTHESIS,
                        replace(FRENCH_HYPOTHESIS, decoding_time_ms=640.0),
                    ],
                    LANGUAGES,
                )

            def test_should_refuse_them_since_the_score_would_count_it_twice(
                self, judge_act: Callable[[], tuple[Hypothesis, ...]]
            ) -> None:
                with pytest.raises(
                    HypothesisError,
                    match=re.escape("the utterance 7105.wav of fr_fr has two hypotheses"),
                ):
                    judge_act()

        class TestWhenALaterHypothesisBreaksARule:
            @pytest.fixture
            def judge_act(self) -> Callable[[], tuple[Hypothesis, ...]]:
                return partial(
                    judge_hypotheses,
                    [ENGLISH_HYPOTHESIS, replace(FRENCH_HYPOTHESIS, language="de_de")],
                    LANGUAGES,
                )

            def test_should_refuse_the_whole_run(
                self, judge_act: Callable[[], tuple[Hypothesis, ...]]
            ) -> None:
                with pytest.raises(HypothesisError, match=re.escape("is in 'de_de'")):
                    judge_act()

    class TestTheJudgeHypothesisFunction:
        @pytest.mark.parametrize(
            ("hypothesis", "reason"),
            list(REFUSED_HYPOTHESES.values()),
            ids=list(REFUSED_HYPOTHESES.keys()),
        )
        class TestWhenTheHypothesisBreaksARule:
            @pytest.fixture
            def judge_act(self, hypothesis: Hypothesis) -> Callable[[], None]:
                return partial(judge_hypothesis, hypothesis, LANGUAGES)

            def test_should_refuse_it_and_say_why(
                self, judge_act: Callable[[], None], reason: str
            ) -> None:
                with pytest.raises(HypothesisError, match=re.escape(reason)):
                    judge_act()
