from dataclasses import replace

import pytest

from mostt_model_builder.domain.acceptance.acceptance import (
    REPORTED_ONLY,
    ComparisonRule,
    NonInferiority,
)
from mostt_model_builder.domain.acceptance.paired_comparison import PairedComparison
from mostt_model_builder.domain.acceptance.verification import (
    CandidateDigest,
    ComparisonMeasure,
    RunMeasure,
    Verification,
)
from mostt_model_builder.domain.containers.devices import Device
from mostt_model_builder.domain.contract.recipe_contract import REFERENCE_VARIANT, JudgedRun
from mostt_model_builder.domain.formats.file_formats import FileFormat
from mostt_model_builder.domain.publication.variant_summaries import (
    LanguageDifference,
    LanguageRate,
    VariantComparison,
    VariantSummary,
    list_shared_files,
    summarize_variants,
)

LANGUAGES = ("fr_fr", "en_us")
NON_INFERIORITY = NonInferiority(0.2)
ASYMMETRIC_DIFFERENCE_POINTS = 0.25


def read_no_lines(lines: bytes, candidate_texts: object) -> tuple[()]:  # noqa: ARG001  # a run reader takes both, and no test here reads the lines of a run
    return ()


def build_run(name: str, variant: str, device: Device) -> JudgedRun:
    return JudgedRun(name, "transcribe", device, variant, LANGUAGES, 1, read_no_lines)


def build_digest(name: str, variants: tuple[str, ...], size_bytes: int) -> CandidateDigest:
    return CandidateDigest(name, FileFormat.ONNX, variants, size_bytes, "0" * 64)


def compare(
    run: str, reference_run: str, rule: ComparisonRule = NON_INFERIORITY
) -> tuple[ComparisonMeasure, ...]:
    return tuple(
        ComparisonMeasure(
            run=run,
            reference_run=reference_run,
            language=language,
            rule=rule,
            identical_count=1,
            compared_count=2,
            paired=PairedComparison(ASYMMETRIC_DIFFERENCE_POINTS, -0.5, 1.0),
            failures=(),
        )
        for language in LANGUAGES
    )


NEMO = build_run("nemo", REFERENCE_VARIANT, Device.CPU)
FLOAT32_CPU = build_run("float32-cpu", "float32", Device.CPU)
FLOAT16_CPU = build_run("float16-cpu", "float16", Device.CPU)
FLOAT16_GPU = build_run("float16-gpu", "float16", Device.GPU)
RUNS = (NEMO, FLOAT32_CPU, FLOAT16_CPU, FLOAT16_GPU)
CANDIDATE = (
    build_digest("encoder.onnx", ("float32",), 100),
    build_digest("encoder.fp16.onnx", ("float16",), 50),
    build_digest("vocab.txt", ("float32", "float16"), 10),
    build_digest("config.json", ("float32", "float16"), 5),
)
RUN_MEASURES = tuple(
    RunMeasure(run.name, language, 2, 100, error_count, error_count + 2, 1, 10.0)
    for run, error_count in ((NEMO, 4), (FLOAT32_CPU, 5), (FLOAT16_CPU, 6), (FLOAT16_GPU, 7))
    for language in LANGUAGES
)
VERIFICATION = Verification(runs=RUN_MEASURES, comparisons=(), parities=(), candidate=CANDIDATE)


def summarize_float16(
    comparisons: tuple[ComparisonMeasure, ...] = (), runs: tuple[JudgedRun, ...] = RUNS
) -> VariantSummary:
    summaries = summarize_variants(replace(VERIFICATION, comparisons=comparisons), runs)
    return next(summary for summary in summaries if summary.variant == "float16")


def build_comparison(
    other_run: JudgedRun, difference_points: float, is_reported_only: bool = False
) -> VariantComparison:
    return VariantComparison(
        other_run=other_run.name,
        other_variant=other_run.variant,
        other_device=other_run.device,
        is_reported_only=is_reported_only,
        differences=tuple(
            LanguageDifference(language, difference_points) for language in LANGUAGES
        ),
    )


RATED_RUNS = {
    "runs on the cpu alone, which give the rates": (
        (NEMO, FLOAT32_CPU, FLOAT16_CPU),
        "float16-cpu",
    ),
    "runs on the gpu alone, which give the rates": (
        (NEMO, FLOAT32_CPU, FLOAT16_GPU),
        "float16-gpu",
    ),
    "runs on both devices, of which the cpu's gives the rates": (RUNS, "float16-cpu"),
    "runs on both devices, the gpu's listed first, of which the cpu's gives the rates": (
        (NEMO, FLOAT32_CPU, FLOAT16_GPU, FLOAT16_CPU),
        "float16-cpu",
    ),
}
CLAUSES = {
    "a comparison that judges the rated run against another variant's": (
        compare("float16-cpu", "float32-cpu"),
        (build_comparison(FLOAT32_CPU, ASYMMETRIC_DIFFERENCE_POINTS),),
    ),
    "a reference run judged against the rated run, its difference turned around": (
        compare("nemo", "float16-cpu"),
        (build_comparison(NEMO, -ASYMMETRIC_DIFFERENCE_POINTS),),
    ),
    "the variant's other device judged against the rated run, which gives no clause": (
        compare("float16-gpu", "float16-cpu"),
        (),
    ),
    "two comparisons, which give two clauses in their order": (
        (*compare("float16-cpu", "float32-cpu"), *compare("nemo", "float16-cpu")),
        (
            build_comparison(FLOAT32_CPU, ASYMMETRIC_DIFFERENCE_POINTS),
            build_comparison(NEMO, -ASYMMETRIC_DIFFERENCE_POINTS),
        ),
    ),
    "a figure reported only, read as such": (
        compare("nemo", "float16-cpu", REPORTED_ONLY),
        (build_comparison(NEMO, -ASYMMETRIC_DIFFERENCE_POINTS, is_reported_only=True),),
    ),
}


class TestTheVariantSummaries:
    class TestTheSummarizeVariantsFunction:
        @pytest.mark.parametrize(
            ("runs", "rated_run"), list(RATED_RUNS.values()), ids=list(RATED_RUNS.keys())
        )
        class TestWhenAVariantRunsOnItsDevices:
            @pytest.fixture
            def summary(self, runs: tuple[JudgedRun, ...]) -> VariantSummary:
                return summarize_float16(runs=runs)

            def test_should_rate_it_by_the_run_of_the_cpu_or_else_of_the_gpu(
                self, summary: VariantSummary, rated_run: str
            ) -> None:
                assert summary.rated_run == rated_run

        @pytest.mark.parametrize(
            ("comparisons", "variant_comparisons"),
            list(CLAUSES.values()),
            ids=list(CLAUSES.keys()),
        )
        class TestWhenTheVerificationComparesTheRuns:
            @pytest.fixture
            def summary(self, comparisons: tuple[ComparisonMeasure, ...]) -> VariantSummary:
                return summarize_float16(comparisons)

            def test_should_give_the_variant_the_comparisons_of_its_rated_run(
                self, summary: VariantSummary, variant_comparisons: tuple[VariantComparison, ...]
            ) -> None:
                assert summary.comparisons == variant_comparisons

        class TestWhenTwoVariantsShareFiles:
            @pytest.fixture
            def summaries(self) -> tuple[VariantSummary, ...]:
                return summarize_variants(VERIFICATION, RUNS)

            def test_should_count_the_shared_files_in_each_size(
                self, summaries: tuple[VariantSummary, ...]
            ) -> None:
                assert [summary.size_bytes for summary in summaries] == [115, 65]

            def test_should_leave_the_shared_files_out_of_each_variant_s_own(
                self, summaries: tuple[VariantSummary, ...]
            ) -> None:
                assert [summary.own_files for summary in summaries] == [
                    ("encoder.onnx",),
                    ("encoder.fp16.onnx",),
                ]

            def test_should_rate_each_language_of_the_rated_run(
                self, summaries: tuple[VariantSummary, ...]
            ) -> None:
                assert summaries[1].rates == (
                    LanguageRate("fr_fr", 6.0),
                    LanguageRate("en_us", 6.0),
                )

    class TestTheListSharedFilesFunction:
        class TestWhenTheCandidateHoldsASingleVariant:
            @pytest.fixture
            def shared_files(self) -> tuple[str, ...]:
                return list_shared_files(
                    [digest for digest in CANDIDATE if digest.variants == ("float32",)]
                    + [build_digest("vocab.txt", ("float32",), 10)]
                )

            def test_should_share_no_file_since_no_other_variant_loads_one(
                self, shared_files: tuple[str, ...]
            ) -> None:
                assert shared_files == ()
