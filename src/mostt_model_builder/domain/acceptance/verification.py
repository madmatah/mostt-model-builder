import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from mostt_model_builder.domain.acceptance.acceptance import (
    ComparedTranscripts,
    Comparison,
    ComparisonRule,
    Equivalence,
    NonInferiority,
    find_comparison_failures,
    find_parity_failure,
    name_rule,
)
from mostt_model_builder.domain.acceptance.paired_comparison import (
    PairedComparison,
    PairedUtterance,
    compare_paired_utterances,
)
from mostt_model_builder.domain.acceptance.parity_reports import ParityReport
from mostt_model_builder.domain.bench.hypotheses import Hypothesis
from mostt_model_builder.domain.bench.normalization import normalize_words
from mostt_model_builder.domain.bench.run_coverage import judge_run_coverage
from mostt_model_builder.domain.bench.word_error_rate import (
    compute_word_error_rate_percent,
    count_word_errors,
)

VERIFICATION_REPORT_FILE_NAME = "verification.json"
TEXT_ENCODING = "utf-8"

type UtteranceKey = tuple[str, str]
type ReferenceWords = Mapping[str, Mapping[str, tuple[str, ...]]]


class VerificationError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class RunTranscripts:
    name: str
    languages: tuple[str, ...]
    hypotheses: tuple[Hypothesis, ...]


@dataclass(frozen=True, slots=True)
class RunMeasure:
    run: str
    language: str
    utterance_count: int
    word_count: int
    error_count: int
    error_count_with_unknown_tokens: int
    unknown_token_count: int
    decoding_time_ms: float


@dataclass(frozen=True, slots=True)
class ComparisonMeasure:
    run: str
    reference_run: str
    language: str
    rule: ComparisonRule
    identical_count: int
    compared_count: int
    paired: PairedComparison
    failures: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ParityMeasure:
    report: ParityReport
    absolute_difference_tolerance: float
    failure: str | None


@dataclass(frozen=True, slots=True)
class Verification:
    runs: tuple[RunMeasure, ...]
    comparisons: tuple[ComparisonMeasure, ...]
    parities: tuple[ParityMeasure, ...]


@dataclass(frozen=True, slots=True)
class UtteranceErrors:
    error_count: int
    error_count_with_unknown_tokens: int


def collect_reference_words(transcriptions: Mapping[str, Mapping[str, str]]) -> ReferenceWords:
    references: dict[str, dict[str, tuple[str, ...]]] = {}
    for language, transcriptions_by_id in transcriptions.items():
        references[language] = {}
        for utterance_id, transcription in transcriptions_by_id.items():
            words = normalize_words(transcription)
            if not words:
                raise VerificationError(
                    f"the transcription of {utterance_id} of {language} holds no word once "
                    "normalized"
                )
            references[language][utterance_id] = words
    return references


def measure_parity(report: ParityReport, absolute_difference_tolerance: float) -> ParityMeasure:
    return ParityMeasure(
        report,
        absolute_difference_tolerance,
        find_parity_failure(report, absolute_difference_tolerance),
    )


def verify_transcripts(
    runs: Sequence[RunTranscripts],
    references: ReferenceWords,
    comparisons: Sequence[Comparison],
    parities: Sequence[ParityMeasure],
) -> Verification:
    hypotheses_by_run: dict[str, dict[UtteranceKey, Hypothesis]] = {}
    errors_by_run: dict[str, dict[UtteranceKey, UtteranceErrors]] = {}
    run_measures: list[RunMeasure] = []
    for run in runs:
        judge_run_coverage(run.name, run.hypotheses, run.languages, references)
        hypotheses = {
            (hypothesis.language, hypothesis.utterance_id): hypothesis
            for hypothesis in run.hypotheses
        }
        errors = {
            key: count_utterance_errors(references[key[0]][key[1]], hypothesis)
            for key, hypothesis in hypotheses.items()
        }
        hypotheses_by_run[run.name] = hypotheses
        errors_by_run[run.name] = errors
        run_measures.extend(
            measure_run(run.name, language, hypotheses, errors, references[language])
            for language in run.languages
        )
    comparison_measures = [
        measure_comparison(
            comparison, language, hypotheses_by_run, errors_by_run, references[language]
        )
        for comparison in comparisons
        for language in references
        if has_language(hypotheses_by_run, comparison.run, language)
        and has_language(hypotheses_by_run, comparison.reference_run, language)
    ]
    return Verification(tuple(run_measures), tuple(comparison_measures), tuple(parities))


def has_language(
    hypotheses_by_run: Mapping[str, Mapping[UtteranceKey, Hypothesis]], run: str, language: str
) -> bool:
    if run not in hypotheses_by_run:
        raise VerificationError(f"a comparison names the run {run!a}, which was not verified")
    return any(key[0] == language for key in hypotheses_by_run[run])


def count_utterance_errors(
    reference_words: Sequence[str], hypothesis: Hypothesis
) -> UtteranceErrors:
    return UtteranceErrors(
        error_count=count_word_errors(
            reference_words, normalize_words(hypothesis.text_without_unknown_tokens)
        ),
        error_count_with_unknown_tokens=count_word_errors(
            reference_words, normalize_words(hypothesis.text)
        ),
    )


def measure_run(
    run: str,
    language: str,
    hypotheses: Mapping[UtteranceKey, Hypothesis],
    errors: Mapping[UtteranceKey, UtteranceErrors],
    references: Mapping[str, tuple[str, ...]],
) -> RunMeasure:
    keys = [(language, utterance_id) for utterance_id in references]
    return RunMeasure(
        run=run,
        language=language,
        utterance_count=len(keys),
        word_count=sum(len(words) for words in references.values()),
        error_count=sum(errors[key].error_count for key in keys),
        error_count_with_unknown_tokens=sum(
            errors[key].error_count_with_unknown_tokens for key in keys
        ),
        unknown_token_count=sum(hypotheses[key].unknown_token_count for key in keys),
        decoding_time_ms=sum(hypotheses[key].decoding_time_ms for key in keys),
    )


# The utterances are paired in the order of their ids, so that the resamples of a comparison
# depend neither on the order of the index nor on the shards.
def measure_comparison(
    comparison: Comparison,
    language: str,
    hypotheses_by_run: Mapping[str, Mapping[UtteranceKey, Hypothesis]],
    errors_by_run: Mapping[str, Mapping[UtteranceKey, UtteranceErrors]],
    references: Mapping[str, tuple[str, ...]],
) -> ComparisonMeasure:
    keys = [(language, utterance_id) for utterance_id in sorted(references)]
    errors = errors_by_run[comparison.run]
    reference_errors = errors_by_run[comparison.reference_run]
    hypotheses = hypotheses_by_run[comparison.run]
    reference_hypotheses = hypotheses_by_run[comparison.reference_run]
    compared = ComparedTranscripts(
        identical_count=sum(
            1 for key in keys if hypotheses[key].token_ids == reference_hypotheses[key].token_ids
        ),
        compared_count=len(keys),
        paired=compare_paired_utterances(
            [
                PairedUtterance(
                    error_count=errors[key].error_count,
                    reference_error_count=reference_errors[key].error_count,
                    word_count=len(references[key[1]]),
                )
                for key in keys
            ]
        ),
    )
    return ComparisonMeasure(
        run=comparison.run,
        reference_run=comparison.reference_run,
        language=language,
        rule=comparison.rule,
        identical_count=compared.identical_count,
        compared_count=compared.compared_count,
        paired=compared.paired,
        failures=find_comparison_failures(comparison.rule, compared),
    )


def compute_run_word_error_rate_percent(measure: RunMeasure) -> float:
    return compute_word_error_rate_percent(measure.error_count, measure.word_count)


def compute_run_word_error_rate_with_unknown_tokens_percent(measure: RunMeasure) -> float:
    return compute_word_error_rate_percent(
        measure.error_count_with_unknown_tokens, measure.word_count
    )


def list_failures(verification: Verification) -> tuple[str, ...]:
    comparison_failures = [
        f"{measure.run} against {measure.reference_run}, {measure.language}: {failure}"
        for measure in verification.comparisons
        for failure in measure.failures
    ]
    parity_failures = [
        f"the parity {measure.report.name}: {measure.failure}"
        for measure in verification.parities
        if measure.failure is not None
    ]
    return (*parity_failures, *comparison_failures)


def render_verification_report(verification: Verification) -> bytes:
    document = {
        "is_accepted": not list_failures(verification),
        "parities": [
            {
                "name": measure.report.name,
                "maximum_absolute_difference": measure.report.maximum_absolute_difference,
                "compared_count": measure.report.compared_count,
                "absolute_difference_tolerance": measure.absolute_difference_tolerance,
                "failure": measure.failure,
            }
            for measure in verification.parities
        ],
        "runs": [
            {
                "run": measure.run,
                "language": measure.language,
                "utterance_count": measure.utterance_count,
                "word_count": measure.word_count,
                "error_count": measure.error_count,
                "error_count_with_unknown_tokens": measure.error_count_with_unknown_tokens,
                "word_error_rate_percent": compute_run_word_error_rate_percent(measure),
                "word_error_rate_with_unknown_tokens_percent": (
                    compute_run_word_error_rate_with_unknown_tokens_percent(measure)
                ),
                "unknown_token_count": measure.unknown_token_count,
                "decoding_time_ms": measure.decoding_time_ms,
            }
            for measure in verification.runs
        ],
        "comparisons": [
            {
                "run": measure.run,
                "reference_run": measure.reference_run,
                "language": measure.language,
                "rule": render_rule(measure.rule),
                "identical_count": measure.identical_count,
                "compared_count": measure.compared_count,
                "difference_points": measure.paired.difference_points,
                "interval_low_points": measure.paired.interval_low_points,
                "interval_high_points": measure.paired.interval_high_points,
                "failures": list(measure.failures),
            }
            for measure in verification.comparisons
        ],
    }
    return (json.dumps(document, indent=2, allow_nan=False) + "\n").encode(TEXT_ENCODING)


def render_rule(rule: ComparisonRule) -> dict[str, object]:
    if isinstance(rule, Equivalence):
        return {
            "name": name_rule(rule),
            "min_identical_ratio": rule.min_identical_ratio,
            "max_difference_points": rule.max_difference_points,
        }
    if isinstance(rule, NonInferiority):
        return {"name": name_rule(rule), "max_upper_bound_points": rule.max_upper_bound_points}
    return {"name": name_rule(rule)}
