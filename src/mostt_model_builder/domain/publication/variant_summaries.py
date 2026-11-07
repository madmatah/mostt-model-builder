from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from mostt_model_builder.domain.acceptance.verification import (
    CandidateDigest,
    ComparisonMeasure,
    Verification,
    compute_run_word_error_rate_percent,
)
from mostt_model_builder.domain.containers.devices import Device
from mostt_model_builder.domain.contract.recipe_contract import REFERENCE_VARIANT, JudgedRun

# The runtime and the hardware, not the files, make two devices differ, so one run speaks for a
# variant.
RATED_DEVICES = (Device.CPU, Device.GPU)


class VariantSummaryError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class LanguageRate:
    language: str
    word_error_rate_percent: float


@dataclass(frozen=True, slots=True)
class LanguageDifference:
    language: str
    difference_points: float


@dataclass(frozen=True, slots=True)
class VariantComparison:
    other_run: str
    other_variant: str
    other_device: Device
    differences: tuple[LanguageDifference, ...]


@dataclass(frozen=True, slots=True)
class VariantSummary:
    variant: str
    devices: tuple[Device, ...]
    size_bytes: int
    own_files: tuple[str, ...]
    rated_run: str
    rates: tuple[LanguageRate, ...]
    comparisons: tuple[VariantComparison, ...]


def list_variants(candidate: Sequence[CandidateDigest]) -> list[str]:
    variants: list[str] = []
    for digest in candidate:
        variants.extend(variant for variant in digest.variants if variant not in variants)
    return variants


def list_shared_files(candidate: Sequence[CandidateDigest]) -> tuple[str, ...]:
    variants = list_variants(candidate)
    if len(variants) < 2:
        return ()
    return tuple(
        digest.name
        for digest in candidate
        if all(variant in digest.variants for variant in variants)
    )


def select_rated_run(variant: str, runs: Sequence[JudgedRun]) -> JudgedRun:
    for device in RATED_DEVICES:
        for run in runs:
            if run.variant == variant and run.device is device:
                return run
    raise VariantSummaryError(
        f"the variant {variant} has no run on the cpu or the gpu, so the card has no rate for it"
    )


def summarize_variants(
    verification: Verification, runs: Sequence[JudgedRun]
) -> tuple[VariantSummary, ...]:
    shared_files = list_shared_files(verification.candidate)
    runs_by_name = {run.name: run for run in runs}
    rated_runs = {
        variant: select_rated_run(variant, runs)
        for variant in list_variants(verification.candidate)
    }
    summaries = []
    for variant, rated_run in rated_runs.items():
        variant_devices = {run.device for run in runs if run.variant == variant}
        variant_files = [digest for digest in verification.candidate if variant in digest.variants]
        summaries.append(
            VariantSummary(
                variant=variant,
                devices=tuple(device for device in RATED_DEVICES if device in variant_devices),
                size_bytes=sum(digest.size_bytes for digest in variant_files),
                own_files=tuple(
                    digest.name for digest in variant_files if digest.name not in shared_files
                ),
                rated_run=rated_run.name,
                rates=tuple(
                    LanguageRate(measure.language, compute_run_word_error_rate_percent(measure))
                    for measure in verification.runs
                    if measure.run == rated_run.name
                ),
                comparisons=compare_rated_run(rated_run, verification.comparisons, runs_by_name),
            )
        )
    return tuple(summaries)


def compare_rated_run(
    rated_run: JudgedRun,
    comparisons: Sequence[ComparisonMeasure],
    runs_by_name: Mapping[str, JudgedRun],
) -> tuple[VariantComparison, ...]:
    measures_by_pair: dict[tuple[str, str], list[ComparisonMeasure]] = {}
    for measure in comparisons:
        measures_by_pair.setdefault((measure.run, measure.reference_run), []).append(measure)
    variant_comparisons = []
    for (run_name, reference_run_name), measures in measures_by_pair.items():
        run = runs_by_name[run_name]
        reference_run = runs_by_name[reference_run_name]
        if run.name == rated_run.name and reference_run.variant != rated_run.variant:
            variant_comparisons.append(
                build_variant_comparison(reference_run, measures, is_turned_around=False)
            )
        elif reference_run.name == rated_run.name and run.variant == REFERENCE_VARIANT:
            variant_comparisons.append(
                build_variant_comparison(run, measures, is_turned_around=True)
            )
    return tuple(variant_comparisons)


def build_variant_comparison(
    other_run: JudgedRun, measures: Sequence[ComparisonMeasure], is_turned_around: bool
) -> VariantComparison:
    sign = -1.0 if is_turned_around else 1.0
    return VariantComparison(
        other_run=other_run.name,
        other_variant=other_run.variant,
        other_device=other_run.device,
        differences=tuple(
            LanguageDifference(measure.language, sign * measure.paired.difference_points)
            for measure in measures
        ),
    )
