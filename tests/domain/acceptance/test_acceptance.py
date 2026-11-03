import re
from collections.abc import Callable
from functools import partial

import pytest

from mostt_model_builder.domain.acceptance.acceptance import (
    REPORTED_ONLY,
    AcceptanceError,
    ComparedTranscripts,
    ComparisonRule,
    Equivalence,
    NonInferiority,
    find_comparison_failures,
    find_parity_failure,
    judge_comparison_rule,
)
from mostt_model_builder.domain.acceptance.paired_comparison import PairedComparison
from mostt_model_builder.domain.acceptance.parity_reports import ParityReport

SAME_DEVICE = Equivalence(min_identical_ratio=1.0, max_difference_points=0.10)
CROSS_DEVICE = Equivalence(min_identical_ratio=0.99, max_difference_points=0.10)
VARIANT = NonInferiority(max_upper_bound_points=0.20)
NO_DIFFERENCE = PairedComparison(0.0, 0.0, 0.0)
TOLERANCE = 5e-5


def compare(identical_count: int, low_points: float, high_points: float) -> ComparedTranscripts:
    return ComparedTranscripts(
        identical_count=identical_count,
        compared_count=100,
        paired=PairedComparison((low_points + high_points) / 2, low_points, high_points),
    )


ACCEPTED_COMPARISONS = {
    "identical token ids on every utterance, within a tenth of a point": (
        SAME_DEVICE,
        compare(100, -0.10, 0.10),
    ),
    "exactly 99 % of identical token ids across devices": (CROSS_DEVICE, compare(99, 0.0, 0.0)),
    "both bounds of the interval exactly at a tenth of a point": (
        CROSS_DEVICE,
        compare(100, -0.10, 0.10),
    ),
    "a variant whose upper bound is exactly a fifth of a point": (VARIANT, compare(0, -1.0, 0.20)),
    "a variant far better than its reference": (VARIANT, compare(0, -3.0, -1.0)),
    "a comparison reported only, however far apart": (REPORTED_ONLY, compare(0, -5.0, 5.0)),
}

REFUSED_COMPARISONS = {
    "one utterance of 100 whose token ids differ on the same device": (
        SAME_DEVICE,
        compare(99, 0.0, 0.0),
        ("99 of 100 utterances have identical token ids, fewer than 100.0%",),
    ),
    "99 % of identical token ids where 99.5 % are asked": (
        Equivalence(min_identical_ratio=0.995, max_difference_points=0.10),
        compare(99, 0.0, 0.0),
        ("99 of 100 utterances have identical token ids, fewer than 99.5%",),
    ),
    "98 % of identical token ids across devices": (
        CROSS_DEVICE,
        compare(98, 0.0, 0.0),
        ("98 of 100 utterances have identical token ids, fewer than 99.0%",),
    ),
    "a lower bound beyond a tenth of a point": (
        CROSS_DEVICE,
        compare(100, -0.11, 0.0),
        (
            "the interval of the difference of word error rates, [-0.11, +0.00] points, "
            "leaves ±0.10",
        ),
    ),
    "an upper bound beyond a tenth of a point": (
        CROSS_DEVICE,
        compare(100, 0.0, 0.11),
        (
            "the interval of the difference of word error rates, [+0.00, +0.11] points, "
            "leaves ±0.10",
        ),
    ),
    "both the identity and the interval": (
        SAME_DEVICE,
        compare(90, -0.30, 0.30),
        (
            "90 of 100 utterances have identical token ids, fewer than 100.0%",
            "the interval of the difference of word error rates, [-0.30, +0.30] points, "
            "leaves ±0.10",
        ),
    ),
    "a variant whose upper bound is beyond a fifth of a point": (
        VARIANT,
        compare(100, 0.0, 0.21),
        ("the upper bound of the difference of word error rates, +0.21 points, exceeds +0.20",),
    ),
}

REFUSED_RULES = {
    "an equivalence that asks for no identical token id": (
        Equivalence(0.0, 0.10),
        "an equivalence asks for 0.0 of identical token ids, not a ratio above 0 and at most 1",
    ),
    "an equivalence that asks for more than every token id": (
        Equivalence(1.01, 0.10),
        "asks for 1.01 of identical token ids",
    ),
    "an equivalence whose ratio is not a number": (
        Equivalence(float("nan"), 0.10),
        "asks for nan of identical token ids",
    ),
    "an equivalence whose ratio is a boolean": (
        Equivalence(True, 0.10),
        "asks for True of identical token ids",
    ),
    "an equivalence without a bound": (
        Equivalence(1.0, 0.0),
        "an equivalence bounds the difference of word error rates at 0.0 points, not a number "
        "above 0 and at most 10",
    ),
    "a non-inferiority with a negative bound": (
        NonInferiority(-0.20),
        "a non-inferiority bounds the difference of word error rates at -0.2 points",
    ),
    "a non-inferiority with an infinite bound": (
        NonInferiority(float("inf")),
        "a non-inferiority bounds the difference of word error rates at inf points",
    ),
    "a non-inferiority beyond ten points": (
        NonInferiority(10.5),
        "a non-inferiority bounds the difference of word error rates at 10.5 points",
    ),
    "a rule that is no rule": ("equivalence", "'equivalence' is no rule of a comparison"),
}


class TestTheAcceptance:
    class TestTheFindComparisonFailuresFunction:
        @pytest.mark.parametrize(
            ("rule", "compared"),
            list(ACCEPTED_COMPARISONS.values()),
            ids=list(ACCEPTED_COMPARISONS.keys()),
        )
        class TestWhenTheComparisonKeepsWithinItsRule:
            @pytest.fixture
            def failures(
                self, rule: ComparisonRule, compared: ComparedTranscripts
            ) -> tuple[str, ...]:
                return find_comparison_failures(rule, compared)

            def test_should_accept_it(self, failures: tuple[str, ...]) -> None:
                assert failures == ()

        @pytest.mark.parametrize(
            ("rule", "compared", "expected_failures"),
            list(REFUSED_COMPARISONS.values()),
            ids=list(REFUSED_COMPARISONS.keys()),
        )
        class TestWhenTheComparisonBreaksItsRule:
            @pytest.fixture
            def failures(
                self, rule: ComparisonRule, compared: ComparedTranscripts
            ) -> tuple[str, ...]:
                return find_comparison_failures(rule, compared)

            def test_should_refuse_it_and_say_why(
                self, failures: tuple[str, ...], expected_failures: tuple[str, ...]
            ) -> None:
                assert failures == expected_failures

    class TestTheFindParityFailureFunction:
        class TestWhenTheDifferenceIsExactlyTheTolerance:
            @pytest.fixture
            def failure(self) -> str | None:
                return find_parity_failure(ParityReport("encoder", TOLERANCE, 20), TOLERANCE)

            def test_should_accept_it(self, failure: str | None) -> None:
                assert failure is None

        class TestWhenTheDifferenceExceedsTheTolerance:
            @pytest.fixture
            def failure(self) -> str | None:
                return find_parity_failure(ParityReport("encoder", 5.1e-5, 20), TOLERANCE)

            def test_should_refuse_it_and_say_why(self, failure: str | None) -> None:
                assert failure == (
                    "the maximum absolute difference, 5.1e-05, exceeds the tolerance of 5e-05"
                )

    class TestTheJudgeComparisonRuleFunction:
        @pytest.mark.parametrize(
            "rule", [SAME_DEVICE, CROSS_DEVICE, VARIANT, REPORTED_ONLY], ids=repr
        )
        class TestWhenTheRuleKeepsWithinItsBounds:
            @pytest.fixture
            def judge_act(self, rule: ComparisonRule) -> Callable[[], object]:
                return partial(judge_comparison_rule, rule)

            def test_should_accept_it(self, judge_act: Callable[[], object]) -> None:
                assert judge_act() is None

        @pytest.mark.parametrize(
            ("rule", "reason"), list(REFUSED_RULES.values()), ids=list(REFUSED_RULES.keys())
        )
        class TestWhenTheRuleLeavesItsBounds:
            @pytest.fixture
            def judge_act(self, rule: object) -> Callable[[], object]:
                return partial(judge_comparison_rule, rule)

            def test_should_refuse_it_and_say_why(
                self, judge_act: Callable[[], object], reason: str
            ) -> None:
                with pytest.raises(AcceptanceError, match=re.escape(reason)):
                    judge_act()
