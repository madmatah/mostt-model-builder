import json
import re
from collections.abc import Callable
from dataclasses import replace
from functools import partial
from typing import cast

import pytest

from mostt_model_builder.domain.acceptance.acceptance import (
    REPORTED_ONLY,
    Comparison,
    Equivalence,
    NonInferiority,
)
from mostt_model_builder.domain.acceptance.parity_reports import ParityReport
from mostt_model_builder.domain.acceptance.verification import (
    ParityMeasure,
    ReferenceWords,
    RunMeasure,
    RunTranscripts,
    Verification,
    VerificationError,
    collect_reference_words,
    compute_run_word_error_rate_percent,
    compute_run_word_error_rate_with_unknown_tokens_percent,
    list_failures,
    measure_parity,
    render_verification_report,
    verify_transcripts,
)
from mostt_model_builder.domain.bench.hypotheses import Hypothesis
from mostt_model_builder.domain.bench.run_coverage import RunCoverageError

TRANSCRIPTIONS = {
    "fr_fr": {
        "1.wav": "Le chat dort.",
        "2.wav": "Il pleut aujourd\N{RIGHT SINGLE QUOTATION MARK}hui",
    },
    "en_us": {"3.wav": "The cat sleeps here"},
}
REFERENCES = collect_reference_words(TRANSCRIPTIONS)
NEMO = RunTranscripts(
    "nemo",
    ("fr_fr", "en_us"),
    (
        Hypothesis("fr_fr", "1.wav", (1, 2, 3), "le chat dort", "le chat dort", 0, 100.0),
        Hypothesis(
            "fr_fr", "2.wav", (4, 0, 0, 5), "il<unk><unk> aujourd'hui", "il aujourd'hui", 2, 50.0
        ),
        Hypothesis(
            "en_us", "3.wav", (6, 7, 8, 9), "the cat sleeps here", "the cat sleeps here", 0, 25.0
        ),
    ),
)
FLOAT32 = RunTranscripts(
    "float32-cpu",
    ("fr_fr", "en_us"),
    (
        replace(NEMO.hypotheses[0], token_ids=(1, 2, 10, 11)),
        NEMO.hypotheses[1],
        Hypothesis(
            "en_us", "3.wav", (6, 7, 12, 9), "the cat sleep here", "the cat sleep here", 0, 30.0
        ),
    ),
)
TRANSFORMERS = RunTranscripts("transformers", ("fr_fr",), NEMO.hypotheses[:2])
ENCODER_PARITY = measure_parity(ParityReport("encoder-onnx-against-nemo", 4.18e-6, 20), 5e-5)
COMPARISONS = (
    Comparison("float32-cpu", "nemo", NonInferiority(max_upper_bound_points=100.0)),
    Comparison("transformers", "nemo", REPORTED_ONLY),
)


def verify_runs(comparisons: tuple[Comparison, ...] = COMPARISONS) -> Verification:
    return verify_transcripts(
        (NEMO, FLOAT32, TRANSFORMERS), REFERENCES, comparisons, (ENCODER_PARITY,)
    )


class TestTheVerification:
    class TestTheCollectReferenceWordsFunction:
        class TestWhenGivenTheTranscriptionsOfTheBench:
            @pytest.fixture
            def reference_words(self) -> ReferenceWords:
                return collect_reference_words(TRANSCRIPTIONS)

            def test_should_normalize_each_transcription_into_its_words(
                self, reference_words: ReferenceWords
            ) -> None:
                assert reference_words == {
                    "fr_fr": {
                        "1.wav": ("le", "chat", "dort"),
                        "2.wav": ("il", "pleut", "aujourd'hui"),
                    },
                    "en_us": {"3.wav": ("the", "cat", "sleeps", "here")},
                }

        class TestWhenATranscriptionHoldsNoWord:
            @pytest.fixture
            def collect_act(self) -> Callable[[], ReferenceWords]:
                return partial(collect_reference_words, {"fr_fr": {"1.wav": " ... "}})

            def test_should_refuse_it_since_its_errors_could_not_be_rated(
                self, collect_act: Callable[[], ReferenceWords]
            ) -> None:
                with pytest.raises(
                    VerificationError,
                    match=re.escape(
                        "the transcription of 1.wav of fr_fr holds no word once normalized"
                    ),
                ):
                    collect_act()

    class TestTheVerifyTranscriptsFunction:
        class TestWhenEveryRunCoversItsLanguages:
            @pytest.fixture(scope="class")
            @classmethod
            def verification(cls) -> Verification:
                return verify_runs()

            def test_should_measure_each_run_on_each_of_its_languages(
                self, verification: Verification
            ) -> None:
                assert verification.runs[:2] == (
                    RunMeasure("nemo", "fr_fr", 2, 6, 1, 2, 2, 150.0),
                    RunMeasure("nemo", "en_us", 1, 4, 0, 0, 0, 25.0),
                )

            def test_should_rate_the_errors_without_and_with_the_unknown_tokens(
                self, verification: Verification
            ) -> None:
                assert (
                    compute_run_word_error_rate_percent(verification.runs[0]),
                    compute_run_word_error_rate_with_unknown_tokens_percent(verification.runs[0]),
                ) == pytest.approx((100 / 6, 200 / 6))

            def test_should_measure_the_runs_in_their_order(
                self, verification: Verification
            ) -> None:
                assert [(measure.run, measure.language) for measure in verification.runs] == [
                    ("nemo", "fr_fr"),
                    ("nemo", "en_us"),
                    ("float32-cpu", "fr_fr"),
                    ("float32-cpu", "en_us"),
                    ("transformers", "fr_fr"),
                ]

            def test_should_compare_two_runs_on_each_language_they_share_alone(
                self, verification: Verification
            ) -> None:
                assert [
                    (measure.run, measure.reference_run, measure.language)
                    for measure in verification.comparisons
                ] == [
                    ("float32-cpu", "nemo", "fr_fr"),
                    ("float32-cpu", "nemo", "en_us"),
                    ("transformers", "nemo", "fr_fr"),
                ]

            def test_should_count_as_identical_only_the_same_token_ids_not_the_same_words(
                self, verification: Verification
            ) -> None:
                assert [
                    (measure.identical_count, measure.compared_count)
                    for measure in verification.comparisons
                ] == [(1, 2), (0, 1), (2, 2)]

            def test_should_give_the_difference_of_word_error_rates_in_points(
                self, verification: Verification
            ) -> None:
                assert verification.comparisons[1].paired.difference_points == pytest.approx(25.0)

            def test_should_keep_the_parities_it_was_given(
                self, verification: Verification
            ) -> None:
                assert verification.parities == (ENCODER_PARITY,)

            def test_should_find_no_failure(self, verification: Verification) -> None:
                assert list_failures(verification) == ()

        class TestWhenAComparisonBreaksItsRule:
            @pytest.fixture(scope="class")
            @classmethod
            def verification(cls) -> Verification:
                return verify_runs((Comparison("float32-cpu", "nemo", Equivalence(1.0, 0.10)),))

            def test_should_name_the_comparison_its_language_and_each_rule_it_breaks(
                self, verification: Verification
            ) -> None:
                assert list_failures(verification) == (
                    "float32-cpu against nemo, fr_fr: 1 of 2 utterances have identical token ids, "
                    "fewer than 100.0%",
                    "float32-cpu against nemo, en_us: 0 of 1 utterances have identical token ids, "
                    "fewer than 100.0%",
                    "float32-cpu against nemo, en_us: the interval of the difference of word error "
                    "rates, [+25.00, +25.00] points, leaves ±0.10",
                )

        class TestWhenARunLacksAnUtterance:
            @pytest.fixture
            def verify_act(self) -> Callable[[], Verification]:
                return partial(
                    verify_transcripts,
                    (NEMO, replace(FLOAT32, hypotheses=FLOAT32.hypotheses[1:])),
                    REFERENCES,
                    (),
                    (),
                )

            def test_should_refuse_the_run(self, verify_act: Callable[[], Verification]) -> None:
                with pytest.raises(
                    RunCoverageError,
                    match=re.escape(
                        "the run float32-cpu lacks 1 utterances of fr_fr, such as 1.wav"
                    ),
                ):
                    verify_act()

        class TestWhenAComparisonNamesARunNotVerified:
            @pytest.fixture
            def verify_act(self) -> Callable[[], Verification]:
                return partial(
                    verify_transcripts,
                    (NEMO,),
                    REFERENCES,
                    (Comparison("float32-cpu", "nemo", REPORTED_ONLY),),
                    (),
                )

            def test_should_refuse_it(self, verify_act: Callable[[], Verification]) -> None:
                with pytest.raises(
                    VerificationError,
                    match=re.escape(
                        "a comparison names the run 'float32-cpu', which was not verified"
                    ),
                ):
                    verify_act()

    class TestTheMeasureParityFunction:
        class TestWhenTheDifferenceExceedsItsTolerance:
            @pytest.fixture
            def parity_measure(self) -> ParityMeasure:
                return measure_parity(ParityReport("encoder-onnx-against-nemo", 6e-5, 20), 5e-5)

            def test_should_keep_why_it_fails(self, parity_measure: ParityMeasure) -> None:
                assert parity_measure.failure == (
                    "the maximum absolute difference, 6e-05, exceeds the tolerance of 5e-05"
                )

        class TestWhenAFailingParityIsVerified:
            @pytest.fixture
            def failures(self) -> tuple[str, ...]:
                return list_failures(
                    Verification(
                        (),
                        (),
                        (
                            measure_parity(
                                ParityReport("encoder-onnx-against-nemo", 6e-5, 20), 5e-5
                            ),
                        ),
                    )
                )

            def test_should_name_the_parity_among_the_failures(
                self, failures: tuple[str, ...]
            ) -> None:
                assert failures == (
                    "the parity encoder-onnx-against-nemo: the maximum absolute difference, "
                    "6e-05, exceeds the tolerance of 5e-05",
                )

    class TestTheRenderVerificationReportFunction:
        class TestWhenTheBuildIsAccepted:
            @pytest.fixture(scope="class")
            @classmethod
            def report(cls) -> dict[str, object]:
                return cast(
                    dict[str, object], json.loads(render_verification_report(verify_runs()))
                )

            def test_should_say_that_the_build_is_accepted(self, report: dict[str, object]) -> None:
                assert report["is_accepted"] is True

            def test_should_give_each_run_its_rates_and_counts(
                self, report: dict[str, object]
            ) -> None:
                assert cast(list[object], report["runs"])[0] == {
                    "run": "nemo",
                    "language": "fr_fr",
                    "utterance_count": 2,
                    "word_count": 6,
                    "error_count": 1,
                    "error_count_with_unknown_tokens": 2,
                    "word_error_rate_percent": pytest.approx(100 / 6),
                    "word_error_rate_with_unknown_tokens_percent": pytest.approx(200 / 6),
                    "unknown_token_count": 2,
                    "decoding_time_ms": 150.0,
                }

            def test_should_give_each_comparison_its_rule_its_counts_and_its_interval(
                self, report: dict[str, object]
            ) -> None:
                assert cast(list[object], report["comparisons"])[1] == {
                    "run": "float32-cpu",
                    "reference_run": "nemo",
                    "language": "en_us",
                    "rule": {"name": "non-inferiority", "max_upper_bound_points": 100.0},
                    "identical_count": 0,
                    "compared_count": 1,
                    "difference_points": 25.0,
                    "interval_low_points": 25.0,
                    "interval_high_points": 25.0,
                    "failures": [],
                }

            def test_should_give_each_parity_its_difference_and_its_tolerance(
                self, report: dict[str, object]
            ) -> None:
                assert report["parities"] == [
                    {
                        "name": "encoder-onnx-against-nemo",
                        "maximum_absolute_difference": 4.18e-6,
                        "compared_count": 20,
                        "absolute_difference_tolerance": 5e-5,
                        "failure": None,
                    }
                ]

        class TestWhenAComparisonFails:
            @pytest.fixture
            def report(self) -> dict[str, object]:
                return cast(
                    dict[str, object],
                    json.loads(
                        render_verification_report(
                            verify_runs(
                                (Comparison("float32-cpu", "nemo", Equivalence(1.0, 0.10)),)
                            )
                        )
                    ),
                )

            def test_should_say_that_the_build_is_refused(self, report: dict[str, object]) -> None:
                assert report["is_accepted"] is False
