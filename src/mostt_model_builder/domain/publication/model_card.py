from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from string import Template

from mostt_model_builder.domain.acceptance.verification import Verification
from mostt_model_builder.domain.bench.bench_languages import LANGUAGE_NAMES
from mostt_model_builder.domain.containers.container_files import FileFacts, FileKind
from mostt_model_builder.domain.containers.devices import Device
from mostt_model_builder.domain.contract.model_description import (
    ModelDescription,
    VariantDescription,
)
from mostt_model_builder.domain.contract.recipe_contract import REFERENCE_VARIANT, JudgedRun
from mostt_model_builder.domain.publication.measuring_libraries import (
    DEVICE_ORDER,
    MeasuringLibrary,
)
from mostt_model_builder.domain.publication.variant_summaries import (
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
    measuring_libraries: tuple[MeasuringLibrary, ...]


def render_model_card(inputs: CardInputs) -> bytes:
    summaries = summarize_variants(inputs.verification, inputs.runs)
    shared_files = list_shared_files(inputs.verification.candidate)
    revisions = {pin.repository: pin.revision for pin in inputs.pins}
    measured_languages = list_measured_languages(inputs.verification)
    fills = {
        "title": inputs.description.title,
        "variant_sections": "\n\n".join(
            render_variant_section(summary, shared_files, inputs) for summary in summaries
        ),
        "measured_languages": join_words(
            [name_language(language) for language in measured_languages]
        ),
        "measured_utterances": describe_measured_utterances(
            inputs.verification, measured_languages
        ),
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


def format_word_error_rate(word_error_rate_percent: float) -> str:
    return f"{word_error_rate_percent:.2f} %"


def join_words(words: Sequence[str]) -> str:
    if len(words) < 2:
        return "".join(words)
    return f"{', '.join(words[:-1])} and {words[-1]}"


def name_language(language: str) -> str:
    if language not in LANGUAGE_NAMES:
        raise CardError(f"the card has no name for the language {language}, which a run measured")
    return LANGUAGE_NAMES[language]


def describe_variant(variant: str, inputs: CardInputs) -> VariantDescription:
    descriptions = {description.variant: description for description in inputs.description.variants}
    if variant not in descriptions:
        raise CardError(f"the description gives no name to the variant {variant}")
    return descriptions[variant]


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


def render_variant_section(
    summary: VariantSummary, shared_files: Sequence[str], inputs: CardInputs
) -> str:
    description = describe_variant(summary.variant, inputs)
    heading = f"### {description.name}" + (
        f" ({description.abbreviation})" if description.abbreviation is not None else ""
    )
    devices = ", ".join(name_device(device) for device in summary.devices)
    facts = (
        f"- **Runs on:** {devices}"
        + (f"; {description.note}" if description.note is not None else "")
        + f"\n- **Size:** {format_size(summary.size_bytes)}"
    )
    files = "\n".join(
        [
            *(f"- `{name}`" for name in summary.own_files),
            *(f"- `{name}`, shared by every variant" for name in shared_files),
        ]
    )
    return "\n\n".join(
        (
            heading,
            description.role_line,
            facts,
            render_rates(summary, inputs),
            "Files:",
            files,
        )
    )


def render_rates(summary: VariantSummary, inputs: CardInputs) -> str:
    differences_by_comparison = [
        {difference.language: difference.difference_points for difference in comparison.differences}
        for comparison in summary.comparisons
    ]
    rows = [
        (
            f"{name_language(rate.language)} ({rate.language})",
            format_word_error_rate(rate.word_error_rate_percent),
            *(
                f"{format_difference(differences_by_language[rate.language])} points"
                if rate.language in differences_by_language
                else "not measured"
                for differences_by_language in differences_by_comparison
            ),
        )
        for rate in summary.rates
    ]
    header = (
        "Language",
        "WER",
        *(head_comparison(comparison, inputs) for comparison in summary.comparisons),
    )
    return render_table(header, rows)


def head_comparison(comparison: VariantComparison, inputs: CardInputs) -> str:
    if comparison.other_variant == REFERENCE_VARIANT:
        other = title_reference_run(comparison.other_run, inputs)
    else:
        other = describe_variant(comparison.other_variant, inputs).name
    return f"Against {other}, {name_device(comparison.other_device)}"


def describe_measured_utterances(verification: Verification, languages: Sequence[str]) -> str:
    utterance_counts: dict[str, int] = {}
    for measure in verification.runs:
        utterance_counts.setdefault(measure.language, measure.utterance_count)
    return ", ".join(
        f"{utterance_counts[language]}"
        + (" utterances" if index == 0 else "")
        + f" in {name_language(language)}"
        for index, language in enumerate(languages)
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
