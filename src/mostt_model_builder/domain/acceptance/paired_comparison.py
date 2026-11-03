import random
from collections.abc import Sequence
from dataclasses import dataclass

from mostt_model_builder.domain.bench.word_error_rate import PERCENT

RESAMPLE_COUNT = 2000
# Each comparison draws its resamples from a generator of its own, seeded alike, so that its
# interval replays whatever the other comparisons of the build.
RESAMPLE_SEED = 0
# The 2.5th and the 97.5th percentiles of the sorted resamples, each leaving 50 of the 2000 outside
# the interval.
INTERVAL_LOW_INDEX = 50
INTERVAL_HIGH_INDEX = 1949


class PairedComparisonError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class PairedUtterance:
    error_count: int
    reference_error_count: int
    word_count: int


@dataclass(frozen=True, slots=True)
class PairedComparison:
    difference_points: float
    interval_low_points: float
    interval_high_points: float


def compare_paired_utterances(utterances: Sequence[PairedUtterance]) -> PairedComparison:
    if not utterances:
        raise PairedComparisonError("a paired comparison needs at least one utterance")
    if any(utterance.word_count <= 0 for utterance in utterances):
        raise PairedComparisonError("a paired comparison needs words in every reference")
    generator = random.Random(RESAMPLE_SEED)
    resampled_differences = sorted(
        compute_difference_points(
            [utterances[generator.randrange(len(utterances))] for _ in utterances]
        )
        for _ in range(RESAMPLE_COUNT)
    )
    return PairedComparison(
        difference_points=compute_difference_points(utterances),
        interval_low_points=resampled_differences[INTERVAL_LOW_INDEX],
        interval_high_points=resampled_differences[INTERVAL_HIGH_INDEX],
    )


def compute_difference_points(utterances: Sequence[PairedUtterance]) -> float:
    error_difference = sum(
        utterance.error_count - utterance.reference_error_count for utterance in utterances
    )
    return PERCENT * error_difference / sum(utterance.word_count for utterance in utterances)
