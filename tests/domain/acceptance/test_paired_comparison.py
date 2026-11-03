import re
from collections.abc import Callable
from functools import partial

import pytest

from mostt_model_builder.domain.acceptance.paired_comparison import (
    PairedComparison,
    PairedComparisonError,
    PairedUtterance,
    compare_paired_utterances,
)

UTTERANCES = tuple(
    PairedUtterance(error_count, reference_error_count, word_count)
    for error_count, reference_error_count, word_count in (
        (1, 0, 10),
        (0, 0, 12),
        (2, 1, 8),
        (0, 1, 9),
        (3, 3, 15),
        (1, 0, 11),
        (0, 0, 7),
        (2, 0, 10),
        (0, 2, 13),
        (1, 1, 5),
    )
)
# The 2000 resamples that the seed 0 draws from UTTERANCES give these percentiles; another seed, a
# count of resamples or a draw in another order would give others.
SEEDED_INTERVAL_LOW_POINTS = -4.545454545454546
SEEDED_INTERVAL_HIGH_POINTS = 8.433734939759036
IDENTICAL_UTTERANCES = tuple(
    PairedUtterance(utterance.error_count, utterance.error_count, utterance.word_count)
    for utterance in UTTERANCES
)


class TestThePairedComparison:
    class TestTheComparePairedUtterancesFunction:
        class TestWhenTheRunMakesTwoErrorsMoreInAHundredWords:
            @pytest.fixture(scope="class")
            @classmethod
            def paired_comparisons(cls) -> tuple[PairedComparison, PairedComparison]:
                return (
                    compare_paired_utterances(UTTERANCES),
                    compare_paired_utterances(UTTERANCES),
                )

            def test_should_differ_by_two_points(
                self, paired_comparisons: tuple[PairedComparison, PairedComparison]
            ) -> None:
                assert paired_comparisons[0].difference_points == pytest.approx(2.0)

            def test_should_give_the_percentiles_of_the_resamples_that_its_seed_draws(
                self, paired_comparisons: tuple[PairedComparison, PairedComparison]
            ) -> None:
                assert (
                    paired_comparisons[0].interval_low_points,
                    paired_comparisons[0].interval_high_points,
                ) == pytest.approx((SEEDED_INTERVAL_LOW_POINTS, SEEDED_INTERVAL_HIGH_POINTS))

            def test_should_replay_the_same_interval_at_each_comparison(
                self, paired_comparisons: tuple[PairedComparison, PairedComparison]
            ) -> None:
                assert paired_comparisons[0] == paired_comparisons[1]

        class TestWhenBothMakeTheSameErrorsOnEveryUtterance:
            @pytest.fixture(scope="class")
            @classmethod
            def paired_comparison(cls) -> PairedComparison:
                return compare_paired_utterances(IDENTICAL_UTTERANCES)

            def test_should_find_no_difference_in_any_resample(
                self, paired_comparison: PairedComparison
            ) -> None:
                assert paired_comparison == PairedComparison(0.0, 0.0, 0.0)

        class TestWhenThereIsNoUtterance:
            @pytest.fixture
            def compare_act(self) -> Callable[[], PairedComparison]:
                return partial(compare_paired_utterances, ())

            def test_should_refuse_it(self, compare_act: Callable[[], PairedComparison]) -> None:
                with pytest.raises(
                    PairedComparisonError,
                    match=re.escape("a paired comparison needs at least one utterance"),
                ):
                    compare_act()

        class TestWhenAReferenceHoldsNoWord:
            @pytest.fixture
            def compare_act(self) -> Callable[[], PairedComparison]:
                return partial(compare_paired_utterances, (*UTTERANCES, PairedUtterance(1, 0, 0)))

            def test_should_refuse_it_since_a_resample_of_it_alone_has_no_rate(
                self, compare_act: Callable[[], PairedComparison]
            ) -> None:
                with pytest.raises(
                    PairedComparisonError,
                    match=re.escape("a paired comparison needs words in every reference"),
                ):
                    compare_act()
