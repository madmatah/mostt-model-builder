from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from string import Template

from mostt_model_builder.domain.acceptance.verification import Verification
from mostt_model_builder.domain.containers.container_files import FileFacts, FileKind
from mostt_model_builder.domain.containers.devices import Device
from mostt_model_builder.domain.contract.model_description import ModelDescription
from mostt_model_builder.domain.contract.recipe_contract import (
    REFERENCE_VARIANT,
    JudgedRun,
    VariantConversion,
)
from mostt_model_builder.domain.onnx.conversion_request import FloatConversion
from mostt_model_builder.domain.onnx.precision_decisions import list_kept_float32_operators
from mostt_model_builder.domain.publication.measuring_libraries import (
    DEVICE_ORDER,
    MeasuringLibrary,
)
from mostt_model_builder.domain.publication.variant_summaries import (
    LanguageDifference,
    VariantComparison,
    VariantSummary,
    list_shared_files,
    summarize_variants,
)
from mostt_model_builder.domain.sources.source_pins import RepositoryType, SourcePin

CARD_ENCODING = "utf-8"
MODEL_CARD_MAX_SIZE_BYTES = 2**20
MODEL_CARD_FILE_MODE = 0o444
HUB_URL = "https://huggingface.co/"
BYTES_PER_GIGABYTE = 10**9


class CardError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class CardInputs:
    template: str
    description: ModelDescription
    pins: tuple[SourcePin, ...]
    verification: Verification
    runs: tuple[JudgedRun, ...]
    conversions: tuple[VariantConversion, ...]
    measuring_libraries: tuple[MeasuringLibrary, ...]


def render_model_card(inputs: CardInputs) -> bytes:
    summaries = summarize_variants(inputs.verification, inputs.runs)
    revisions = {pin.repository: pin.revision for pin in inputs.pins}
    fills = {
        "title": inputs.description.title,
        "variant_table": render_variant_table(inputs, summaries),
        "variant_sections": "\n\n".join(
            render_variant_section(summary, inputs) for summary in summaries
        ),
        "shared_files": ", ".join(
            f"`{name}`" for name in list_shared_files(inputs.verification.candidate)
        ),
        "measured_languages": " and ".join(list_measured_languages(inputs.verification)),
        "sources": render_sources(inputs.pins),
        "measured_with": describe_measuring_libraries(inputs.measuring_libraries),
    }
    for pinned in inputs.description.pinned_revisions:
        if pinned.placeholder in fills:
            raise CardError(
                f"the description gives the placeholder {pinned.placeholder} a pinned revision, "
                "which the core already fills"
            )
        fills[pinned.placeholder] = find_revision(pinned.repository, revisions)
    template = Template(inputs.template)
    unfilled = [name for name in template.get_identifiers() if name not in fills]
    if unfilled:
        raise CardError(f"the template holds the placeholder {unfilled[0]}, which nothing fills")
    return template.substitute(fills).encode(CARD_ENCODING)


def is_written_card(facts: FileFacts, user_id: int) -> bool:
    return (
        facts.kind is FileKind.REGULAR
        and facts.owner_id == user_id
        and facts.permission_bits == MODEL_CARD_FILE_MODE
        and facts.link_count == 1
        and facts.size_bytes <= MODEL_CARD_MAX_SIZE_BYTES
    )


def refuse_card_of_changed_clone(changed_paths: Sequence[str]) -> None:
    if changed_paths:
        raise CardError(
            f"the clone holds {len(changed_paths)} uncommitted changes or untracked files, such "
            f"as {changed_paths[0]!a}: the card states the libraries and the code of HEAD's "
            "commit"
        )


def find_revision(repository: str, revisions: Mapping[str, str]) -> str:
    if repository not in revisions:
        raise CardError(f"the card gives the revision of {repository}, which no pin names")
    return revisions[repository]


def render_table(header: Sequence[str], rows: Iterable[Sequence[str]]) -> str:
    lines = [
        "| " + " | ".join(header) + " |",
        "|" + "---|" * len(header),
        *("| " + " | ".join(row) + " |" for row in rows),
    ]
    return "\n".join(lines)


def link_repository(repository: str, revision: str, repository_type: RepositoryType) -> str:
    prefix = "datasets/" if repository_type is RepositoryType.DATASET else ""
    return f"{HUB_URL}{prefix}{repository}/tree/{revision}"


def name_device(device: Device) -> str:
    return device.value.upper()


def format_size(size_bytes: int) -> str:
    return f"{size_bytes / BYTES_PER_GIGABYTE:.2f} GB"


def format_difference(difference_points: float) -> str:
    text = f"{difference_points:+.2f}"
    return "+0.00" if text == "-0.00" else text


def join_words(words: Sequence[str]) -> str:
    if len(words) < 2:
        return "".join(words)
    return f"{', '.join(words[:-1])} and {words[-1]}"


def name_variant(variant: str, inputs: CardInputs) -> str:
    names = {description.variant: description.name for description in inputs.description.variants}
    if variant not in names:
        raise CardError(f"the description gives no name to the variant {variant}")
    return names[variant]


def title_reference_run(run: str, inputs: CardInputs) -> str:
    titles = {
        description.run: description.title for description in inputs.description.reference_runs
    }
    if run not in titles:
        raise CardError(
            f"the description gives no title to the reference run {run}, which the card compares "
            "a variant with"
        )
    return titles[run]


def list_rated_languages(summaries: Sequence[VariantSummary]) -> list[str]:
    languages: list[str] = []
    for summary in summaries:
        languages.extend(rate.language for rate in summary.rates if rate.language not in languages)
    return languages


def render_variant_table(inputs: CardInputs, summaries: Sequence[VariantSummary]) -> str:
    languages = list_rated_languages(summaries)
    rows = []
    for summary in summaries:
        rates = {rate.language: rate.word_error_rate_percent for rate in summary.rates}
        rows.append(
            (
                name_variant(summary.variant, inputs),
                ", ".join(name_device(device) for device in summary.devices),
                format_size(summary.size_bytes),
                *(
                    f"{rates[language]:.2f}%" if language in rates else "not measured"
                    for language in languages
                ),
            )
        )
    return render_table(("Variant", "Tested on", "Size", *languages), rows)


def render_variant_section(summary: VariantSummary, inputs: CardInputs) -> str:
    conversions = {conversion.variant: conversion for conversion in inputs.conversions}
    notes = {description.variant: description.note for description in inputs.description.variants}
    paragraphs = [f"Files: {', '.join(f'`{name}`' for name in summary.own_files)}."]
    note = notes.get(summary.variant)
    if note is not None:
        paragraphs.append(note)
    paragraphs.append(describe_comparisons(summary.comparisons, inputs))
    paragraphs.append(f"Conversion: {describe_conversion(conversions.get(summary.variant))}.")
    return f"### {name_variant(summary.variant, inputs)}\n\n" + "\n\n".join(paragraphs)


def describe_comparisons(comparisons: Sequence[VariantComparison], inputs: CardInputs) -> str:
    if not comparisons:
        return "Compared with no other variant and no reference run."
    clauses = "; ".join(describe_comparison(comparison, inputs) for comparison in comparisons)
    return clauses[0].upper() + clauses[1:] + "."


def describe_comparison(comparison: VariantComparison, inputs: CardInputs) -> str:
    device = name_device(comparison.other_device)
    if comparison.other_variant == REFERENCE_VARIANT:
        against = f"against {title_reference_run(comparison.other_run, inputs)}, on the {device}"
    else:
        against = f"against {name_variant(comparison.other_variant, inputs)} on the {device}"
    verdict = "reported only" if comparison.is_reported_only else "accepted"
    return f"{against}: {describe_differences(comparison.differences)}, {verdict}"


def describe_differences(differences: Sequence[LanguageDifference]) -> str:
    words = [
        f"{format_difference(difference.difference_points)}"
        + (" points" if index == 0 else "")
        + f" in {difference.language}"
        for index, difference in enumerate(differences)
    ]
    return join_words(words)


def describe_conversion(variant_conversion: VariantConversion | None) -> str:
    if variant_conversion is None:
        return "none, the exported graph"
    conversion = variant_conversion.conversion
    if isinstance(conversion, FloatConversion):
        kept = join_words(
            list_kept_float32_operators(conversion.kind, conversion.added_float32_operators)
        )
        return f"{conversion.kind.value}" + (f", with {kept} kept in float32" if kept else "")
    symmetry = "symmetric" if conversion.is_symmetric else "asymmetric"
    return (
        f"8-bit blocks of {conversion.block_size} {symmetry} weights, at the accuracy "
        f"level {conversion.accuracy_level}, {conversion.quantized_product_count} products "
        "quantized"
    )


def list_measured_languages(verification: Verification) -> list[str]:
    languages: list[str] = []
    for measure in verification.runs:
        if measure.language not in languages:
            languages.append(measure.language)
    return languages


def render_sources(pins: Sequence[SourcePin]) -> str:
    repositories = dict.fromkeys(
        (pin.repository_type, pin.repository, pin.revision) for pin in pins
    )
    return "\n".join(
        f"- [{repository}]({link_repository(repository, revision, repository_type)})"
        for repository_type, repository, revision in repositories
    )


def describe_measuring_libraries(measuring_libraries: Sequence[MeasuringLibrary]) -> str:
    by_device = [
        (
            device,
            [
                f"{library.name} {library.version}"
                for library in measuring_libraries
                if library.device is device
            ],
        )
        for device in DEVICE_ORDER
    ]
    return ", and ".join(
        f"with {join_words(libraries)} on the {name_device(device)}"
        for device, libraries in by_device
        if libraries
    )
