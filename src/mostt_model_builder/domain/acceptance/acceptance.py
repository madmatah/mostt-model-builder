import math
from dataclasses import dataclass

from mostt_model_builder.domain.acceptance.paired_comparison import PairedComparison
from mostt_model_builder.domain.acceptance.parity_reports import ParityReport

# A difference of word error rates beyond a few points means a broken build, not a close one: the
# bound only refuses a rule that no comparison of a build could need.
MAX_BOUND_POINTS = 10.0


class AcceptanceError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class Equivalence:
    min_identical_ratio: float
    max_difference_points: float


@dataclass(frozen=True, slots=True)
class NonInferiority:
    max_upper_bound_points: float


@dataclass(frozen=True, slots=True)
class ReportedOnly:
    pass


REPORTED_ONLY = ReportedOnly()
type ComparisonRule = Equivalence | NonInferiority | ReportedOnly


@dataclass(frozen=True, slots=True)
class Comparison:
    run: str
    reference_run: str
    rule: ComparisonRule


@dataclass(frozen=True, slots=True)
class ComparedTranscripts:
    identical_count: int
    compared_count: int
    paired: PairedComparison


def name_rule(rule: ComparisonRule) -> str:
    if isinstance(rule, Equivalence):
        return "equivalence"
    if isinstance(rule, NonInferiority):
        return "non-inferiority"
    return "reported only"


def judge_comparison_rule(rule: object) -> None:
    if isinstance(rule, Equivalence):
        if not is_number_within(rule.min_identical_ratio, 0.0, 1.0) or (
            rule.min_identical_ratio == 0
        ):
            raise AcceptanceError(
                f"an equivalence asks for {rule.min_identical_ratio!a} of identical token ids, "
                "not a ratio above 0 and at most 1"
            )
        judge_bound(rule.max_difference_points, "an equivalence")
    elif isinstance(rule, NonInferiority):
        judge_bound(rule.max_upper_bound_points, "a non-inferiority")
    elif not isinstance(rule, ReportedOnly):
        raise AcceptanceError(f"{rule!a} is no rule of a comparison")


def judge_bound(bound_points: float, where: str) -> None:
    if not is_number_within(bound_points, 0.0, MAX_BOUND_POINTS) or bound_points == 0:
        raise AcceptanceError(
            f"{where} bounds the difference of word error rates at {bound_points!a} points, "
            f"not a number above 0 and at most {MAX_BOUND_POINTS:g}"
        )


def is_number_within(number: object, lowest: float, highest: float) -> bool:
    return (
        isinstance(number, int | float)
        and not isinstance(number, bool)
        and math.isfinite(number)
        and lowest <= number <= highest
    )


def find_comparison_failures(
    rule: ComparisonRule, compared: ComparedTranscripts
) -> tuple[str, ...]:
    paired = compared.paired
    failures = []
    if isinstance(rule, Equivalence):
        if compared.identical_count < rule.min_identical_ratio * compared.compared_count:
            failures.append(
                f"{compared.identical_count} of {compared.compared_count} utterances have "
                f"identical token ids, fewer than {rule.min_identical_ratio:.1%}"
            )
        if not (
            -rule.max_difference_points
            <= paired.interval_low_points
            <= paired.interval_high_points
            <= rule.max_difference_points
        ):
            failures.append(
                f"the interval of the difference of word error rates, "
                f"[{paired.interval_low_points:+.2f}, {paired.interval_high_points:+.2f}] points, "
                f"leaves ±{rule.max_difference_points:.2f}"
            )
    elif (
        isinstance(rule, NonInferiority)
        and paired.interval_high_points > rule.max_upper_bound_points
    ):
        failures.append(
            f"the upper bound of the difference of word error rates, "
            f"{paired.interval_high_points:+.2f} points, exceeds "
            f"{rule.max_upper_bound_points:+.2f}"
        )
    return tuple(failures)


def find_parity_failure(report: ParityReport, absolute_difference_tolerance: float) -> str | None:
    if report.maximum_absolute_difference <= absolute_difference_tolerance:
        return None
    return (
        f"the maximum absolute difference, {report.maximum_absolute_difference:.3g}, exceeds "
        f"the tolerance of {absolute_difference_tolerance:.3g}"
    )
