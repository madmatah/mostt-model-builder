import re
from collections.abc import Callable
from dataclasses import replace
from functools import partial

import pytest

from mostt_model_builder.domain.bench.hypotheses import Hypothesis
from mostt_model_builder.domain.bench.run_coverage import RunCoverageError, judge_run_coverage

INDEX = {"fr_fr": ("1.wav", "2.wav"), "en_us": ("3.wav",)}
LANGUAGES = ("fr_fr", "en_us")
FIRST = Hypothesis("fr_fr", "1.wav", (4,), "le", "le", 0, 1.0)
SECOND = replace(FIRST, utterance_id="2.wav")
THIRD = replace(FIRST, language="en_us", utterance_id="3.wav")

REFUSED_RUNS = {
    "an utterance missing": (
        (FIRST, THIRD),
        LANGUAGES,
        "the run float32-cpu lacks 1 utterances of fr_fr, such as 2.wav",
    ),
    "a language of the run without any hypothesis": (
        (FIRST, SECOND),
        LANGUAGES,
        "the run float32-cpu lacks 1 utterances of en_us, such as 3.wav",
    ),
    "an utterance twice, across two shards": (
        (FIRST, SECOND, THIRD, FIRST),
        LANGUAGES,
        "the run float32-cpu holds 1.wav of fr_fr twice, across its shards",
    ),
    "an utterance outside the index": (
        (FIRST, SECOND, THIRD, replace(FIRST, utterance_id="9.wav")),
        LANGUAGES,
        "the run float32-cpu holds '9.wav' of fr_fr, which the index of the bench does not hold",
    ),
    "an utterance of the other language's index": (
        (FIRST, SECOND, THIRD, replace(THIRD, language="fr_fr")),
        LANGUAGES,
        "the run float32-cpu holds '3.wav' of fr_fr, which the index of the bench does not hold",
    ),
    "a hypothesis in a language outside the run": (
        (FIRST, SECOND, THIRD),
        ("fr_fr",),
        "the run float32-cpu holds a hypothesis in 'en_us', not one of its languages",
    ),
}


class TestTheRunCoverage:
    class TestTheJudgeRunCoverageFunction:
        class TestWhenTheRunHoldsEveryUtteranceOfItsLanguagesOnce:
            @pytest.fixture
            def judge_act(self) -> Callable[[], object]:
                return partial(
                    judge_run_coverage, "float32-cpu", (THIRD, SECOND, FIRST), LANGUAGES, INDEX
                )

            def test_should_accept_it_in_any_order(self, judge_act: Callable[[], object]) -> None:
                assert judge_act() is None

        class TestWhenTheRunReadsOneLanguageOfTheBench:
            @pytest.fixture
            def judge_act(self) -> Callable[[], object]:
                return partial(
                    judge_run_coverage, "transformers", (FIRST, SECOND), ("fr_fr",), INDEX
                )

            def test_should_ask_for_that_language_alone(
                self, judge_act: Callable[[], object]
            ) -> None:
                assert judge_act() is None

        @pytest.mark.parametrize(
            ("hypotheses", "languages", "reason"),
            list(REFUSED_RUNS.values()),
            ids=list(REFUSED_RUNS.keys()),
        )
        class TestWhenTheRunDoesNotCoverTheBench:
            @pytest.fixture
            def judge_act(
                self, hypotheses: tuple[Hypothesis, ...], languages: tuple[str, ...]
            ) -> Callable[[], object]:
                return partial(judge_run_coverage, "float32-cpu", hypotheses, languages, INDEX)

            def test_should_refuse_it_and_say_why(
                self, judge_act: Callable[[], object], reason: str
            ) -> None:
                with pytest.raises(RunCoverageError, match=re.escape(reason)):
                    judge_act()
