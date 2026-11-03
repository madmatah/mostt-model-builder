import random
import re
from collections.abc import Callable, Sequence
from functools import cache, partial

import pytest

from mostt_model_builder.domain.bench.word_error_rate import (
    WordErrorRateError,
    compute_word_error_rate_percent,
    count_word_errors,
)

PROPERTY_SEED = 20_261_103
PROPERTY_CASE_COUNT = 500
MAX_PROPERTY_WORD_COUNT = 7
PROPERTY_VOCABULARY = ("le", "chat", "dort", "ici")
REFERENCE = ("the", "cat", "sleeps", "here")

COUNTED_ERRORS = {
    "an identical hypothesis": (REFERENCE, REFERENCE, 0),
    "one word substituted": (REFERENCE, ("the", "dog", "sleeps", "here"), 1),
    "one word deleted": (REFERENCE, ("the", "cat", "sleeps"), 1),
    "one word inserted": (REFERENCE, ("the", "black", "cat", "sleeps", "here"), 1),
    "two words swapped, two substitutions": (REFERENCE, ("the", "sleeps", "cat", "here"), 2),
    "an empty hypothesis, every word deleted": (REFERENCE, (), 4),
    "an empty reference, every word inserted": ((), ("the", "cat"), 2),
    "a deletion and an insertion that a substitution would not spare": (
        ("a", "b", "c"),
        ("b", "c", "d"),
        2,
    ),
    "a hypothesis that shares no word": (REFERENCE, ("a", "dog", "barks"), 4),
}


# The edit distance by its definition, which the dynamic programming of the module must equal.
def count_errors_by_definition(reference: Sequence[str], hypothesis: Sequence[str]) -> int:
    @cache
    def distance(reference_count: int, hypothesis_count: int) -> int:
        if reference_count == 0 or hypothesis_count == 0:
            return reference_count + hypothesis_count
        return min(
            distance(reference_count - 1, hypothesis_count) + 1,
            distance(reference_count, hypothesis_count - 1) + 1,
            distance(reference_count - 1, hypothesis_count - 1)
            + int(reference[reference_count - 1] != hypothesis[hypothesis_count - 1]),
        )

    return distance(len(reference), len(hypothesis))


def build_word_sequences(generator: random.Random) -> list[tuple[tuple[str, ...], ...]]:
    return [
        tuple(
            tuple(
                generator.choice(PROPERTY_VOCABULARY)
                for _ in range(generator.randrange(MAX_PROPERTY_WORD_COUNT + 1))
            )
            for _ in range(2)
        )
        for _ in range(PROPERTY_CASE_COUNT)
    ]


class TestTheWordErrorRate:
    class TestTheCountWordErrorsFunction:
        @pytest.mark.parametrize(
            ("reference", "hypothesis", "error_count"),
            list(COUNTED_ERRORS.values()),
            ids=list(COUNTED_ERRORS.keys()),
        )
        class TestWhenGivenAHandComputedCase:
            @pytest.fixture
            def counted_errors(
                self, reference: tuple[str, ...], hypothesis: tuple[str, ...]
            ) -> int:
                return count_word_errors(reference, hypothesis)

            def test_should_count_the_fewest_substitutions_deletions_and_insertions(
                self, counted_errors: int, error_count: int
            ) -> None:
                assert counted_errors == error_count

        class TestWhenGivenAnyTwoSequencesOfWords:
            def test_should_equal_the_edit_distance_by_its_definition(self) -> None:
                generator = random.Random(PROPERTY_SEED)
                cases = build_word_sequences(generator)
                mismatches = [
                    (reference, hypothesis)
                    for reference, hypothesis in cases
                    if count_word_errors(reference, hypothesis)
                    != count_errors_by_definition(reference, hypothesis)
                ]
                assert mismatches == [], f"seed {PROPERTY_SEED}"
                assert len(cases) == PROPERTY_CASE_COUNT, f"seed {PROPERTY_SEED}"

    class TestTheComputeWordErrorRatePercentFunction:
        class TestWhenOneWordInFourIsWrong:
            @pytest.fixture
            def error_rate_percent(self) -> float:
                return compute_word_error_rate_percent(1, 4)

            def test_should_give_25_percent(self, error_rate_percent: float) -> None:
                assert error_rate_percent == pytest.approx(25.0)

        class TestWhenTheReferenceHoldsNoWord:
            @pytest.fixture
            def compute_act(self) -> Callable[[], float]:
                return partial(compute_word_error_rate_percent, 0, 0)

            def test_should_refuse_it_since_the_rate_has_no_meaning(
                self, compute_act: Callable[[], float]
            ) -> None:
                with pytest.raises(
                    WordErrorRateError,
                    match=re.escape("a word error rate over 0 words has no meaning"),
                ):
                    compute_act()
