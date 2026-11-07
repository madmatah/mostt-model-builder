import re
from collections.abc import Callable
from dataclasses import replace
from functools import partial
from pathlib import PurePosixPath

import pytest

from mostt_model_builder.domain.acceptance.acceptance import (
    REPORTED_ONLY,
    ComparisonRule,
    Equivalence,
    NonInferiority,
)
from mostt_model_builder.domain.acceptance.paired_comparison import PairedComparison
from mostt_model_builder.domain.acceptance.parity_reports import ParityReport
from mostt_model_builder.domain.acceptance.verification import (
    CandidateDigest,
    ComparisonMeasure,
    RunMeasure,
    Verification,
    measure_parity,
)
from mostt_model_builder.domain.containers.devices import Device
from mostt_model_builder.domain.contract.model_description import (
    ModelDescription,
    PinnedRevision,
    ReferenceRunDescription,
    VariantDescription,
)
from mostt_model_builder.domain.contract.recipe_contract import REFERENCE_VARIANT, JudgedRun
from mostt_model_builder.domain.formats.file_formats import FileFormat
from mostt_model_builder.domain.publication.measuring_libraries import MeasuringLibrary
from mostt_model_builder.domain.publication.model_card import (
    CardError,
    CardInputs,
    join_words,
    render_model_card,
)
from mostt_model_builder.domain.publication.variant_summaries import VariantSummaryError
from mostt_model_builder.domain.sources.source_pins import RepositoryType, SourcePin

WEIGHTS_REVISION = "0123456789abcdef0123456789abcdef01234567"
FLEURS_REVISION = "70bb2e84b976b7e960aa89f1c648e09c59f894dd"
TEMPLATE = (
    "# $title\n\n$variant_sections\n\n"
    "Measured $measured_languages ($measured_utterances) $measured_with, the weights at "
    "$weights_revision.\n\n$sources\n"
)
PINS = (
    SourcePin(
        RepositoryType.DATASET, "google/fleurs", FLEURS_REVISION, "data/fr_fr/test.tsv", 9, "1" * 64
    ),
    SourcePin(
        RepositoryType.MODEL, "maker/weights", WEIGHTS_REVISION, "model.safetensors", 7, "2" * 64
    ),
    SourcePin(RepositoryType.MODEL, "maker/weights", WEIGHTS_REVISION, "config.json", 5, "3" * 64),
)
NEMO = ReferenceRunDescription("nemo", "the original model in NeMo")
TRANSFORMERS = ReferenceRunDescription("transformers", "the original model in Transformers")
FLOAT32 = VariantDescription(
    "float32",
    "float32",
    "The exported graph, unconverted.",
    note="the reference of the other variants",
    abbreviation="FP32",
)
EIGHT_BIT_BLOCKS = VariantDescription("eight-bit-blocks", "int8", "The encoder in 8-bit blocks.")
DESCRIPTION = ModelDescription(
    title="Weights in ONNX",
    template_path=PurePosixPath("model-card.md"),
    measuring_library_names=("onnx-asr", "onnxruntime", "onnxruntime-gpu"),
    variants=(FLOAT32, EIGHT_BIT_BLOCKS),
    reference_runs=(NEMO, TRANSFORMERS),
    pinned_revisions=(PinnedRevision("weights_revision", "maker/weights"),),
)


def read_no_lines(lines: bytes, candidate_texts: object) -> tuple[()]:  # noqa: ARG001  # a run reader takes both, and no test here reads the lines of a run
    return ()


def compare(
    run: str, reference_run: str, language: str, rule: ComparisonRule, difference_points: float
) -> ComparisonMeasure:
    return ComparisonMeasure(
        run=run,
        reference_run=reference_run,
        language=language,
        rule=rule,
        identical_count=1,
        compared_count=2,
        paired=PairedComparison(
            difference_points, difference_points - 0.5, difference_points + 0.5
        ),
        failures=(),
    )


EQUIVALENCE = Equivalence(0.5, 0.5)
NON_INFERIORITY = NonInferiority(0.2)
INT8_AGAINST_FLOAT32 = compare(
    "eight-bit-blocks-cpu", "float32-cpu", "fr_fr", NON_INFERIORITY, 0.05
)
NEMO_AGAINST_FLOAT32_IN_ENGLISH = compare("nemo", "float32-cpu", "en_us", EQUIVALENCE, 0.0)
VERIFICATION = Verification(
    runs=(
        RunMeasure("nemo", "fr_fr", 2, 100, 5, 7, 2, 1000.0),
        RunMeasure("nemo", "en_us", 3, 100, 4, 4, 0, 1000.0),
        RunMeasure("float32-cpu", "fr_fr", 2, 100, 5, 7, 2, 1000.0),
        RunMeasure("float32-cpu", "en_us", 3, 100, 4, 4, 0, 1000.0),
        RunMeasure("float32-gpu", "fr_fr", 2, 100, 6, 8, 2, 400.0),
        RunMeasure("float32-gpu", "en_us", 3, 100, 3, 3, 0, 400.0),
        RunMeasure("eight-bit-blocks-cpu", "fr_fr", 2, 100, 6, 8, 2, 900.0),
        RunMeasure("transformers", "fr_fr", 2, 100, 7, 9, 2, 1200.0),
    ),
    comparisons=(
        compare("nemo", "float32-cpu", "fr_fr", EQUIVALENCE, 0.25),
        NEMO_AGAINST_FLOAT32_IN_ENGLISH,
        compare("float32-gpu", "float32-cpu", "fr_fr", EQUIVALENCE, 1.0),
        compare("float32-gpu", "float32-cpu", "en_us", EQUIVALENCE, -1.0),
        INT8_AGAINST_FLOAT32,
        compare("transformers", "eight-bit-blocks-cpu", "fr_fr", REPORTED_ONLY, 0.3),
    ),
    parities=(measure_parity(ParityReport("encoder-onnx-against-nemo", 4.18e-6, 20), 5e-5),),
    candidate=(
        CandidateDigest("encoder.onnx", FileFormat.ONNX, ("float32",), 40_000_000, "a" * 64),
        CandidateDigest(
            "encoder.onnx.data",
            FileFormat.ONNX_EXTERNAL_DATA,
            ("float32",),
            2_480_000_000,
            "b" * 64,
        ),
        CandidateDigest(
            "encoder.int8.onnx", FileFormat.ONNX, ("eight-bit-blocks",), 740_000_000, "c" * 64
        ),
        CandidateDigest(
            "vocab.txt", FileFormat.TEXT, ("float32", "eight-bit-blocks"), 30_000_000, "d" * 64
        ),
    ),
)
BOTH_LANGUAGES = ("fr_fr", "en_us")
RUNS = (
    JudgedRun("nemo", "reference", Device.CPU, REFERENCE_VARIANT, BOTH_LANGUAGES, 1, read_no_lines),
    JudgedRun(
        "transformers", "transformers", Device.CPU, REFERENCE_VARIANT, ("fr_fr",), 1, read_no_lines
    ),
    JudgedRun("float32-cpu", "transcribe", Device.CPU, "float32", BOTH_LANGUAGES, 1, read_no_lines),
    JudgedRun("float32-gpu", "transcribe", Device.GPU, "float32", BOTH_LANGUAGES, 1, read_no_lines),
    JudgedRun(
        "eight-bit-blocks-cpu",
        "transcribe",
        Device.CPU,
        "eight-bit-blocks",
        ("fr_fr",),
        1,
        read_no_lines,
    ),
)
INPUTS = CardInputs(
    template=TEMPLATE,
    description=DESCRIPTION,
    pins=PINS,
    verification=VERIFICATION,
    runs=RUNS,
    measuring_libraries=(
        MeasuringLibrary(Device.CPU, "onnx-asr", "0.12.0"),
        MeasuringLibrary(Device.CPU, "onnxruntime", "1.31.0"),
        MeasuringLibrary(Device.GPU, "onnx-asr", "0.12.0"),
        MeasuringLibrary(Device.GPU, "onnxruntime-gpu", "1.31.0"),
    ),
)
FLOAT32_RATES = """| Language | WER | Against the original model in NeMo, CPU |
|---|---|---|
| French (fr_fr) | 5.00 % | -0.25 points |
| English (en_us) | 4.00 % | +0.00 points |
"""
FLOAT32_FILES = """Files:

- `encoder.onnx`
- `encoder.onnx.data`
- `vocab.txt`, shared by every variant
"""
EIGHT_BIT_BLOCKS_RATES = (
    "| Language | WER | Against float32, CPU | Against the original model in Transformers, CPU |\n"
    "|---|---|---|---|\n"
    "| French (fr_fr) | 6.00 % | +0.05 points | -0.30 points |\n"
)
ITALIAN_RUN_MEASURE = RunMeasure("float32-cpu", "it_it", 2, 100, 5, 7, 2, 1000.0)
THREE_WORDS = ("French", "English", "German")


def change_description(**fields: object) -> CardInputs:
    return replace(INPUTS, description=replace(DESCRIPTION, **fields))  # type: ignore[arg-type]  # each row changes one field of the description, by its name


REFUSED_INPUTS = {
    "a pinned revision of a repository that no pin names": (
        change_description(pinned_revisions=(PinnedRevision("weights_revision", "maker/other"),)),
        CardError,
        "the card gives the revision of maker/other, which no pin names",
    ),
    "a pinned revision in place of a placeholder that the core fills": (
        change_description(pinned_revisions=(PinnedRevision("title", "maker/weights"),)),
        CardError,
        "the description gives the placeholder title a pinned revision, which the core already "
        "fills",
    ),
    "a placeholder that nothing fills": (
        replace(INPUTS, template=f"{TEMPLATE}$license\n"),
        CardError,
        "the template holds the placeholder license, which nothing fills",
    ),
    "the commit of the build, which the card no longer names": (
        replace(INPUTS, template=f"{TEMPLATE}$build_commit\n"),
        CardError,
        "the template holds the placeholder build_commit, which nothing fills",
    ),
    "a variant without a name": (
        change_description(variants=(EIGHT_BIT_BLOCKS,)),
        CardError,
        "the description gives no name to the variant float32",
    ),
    "a measured language without a name": (
        replace(
            INPUTS,
            verification=replace(VERIFICATION, runs=(*VERIFICATION.runs, ITALIAN_RUN_MEASURE)),
        ),
        CardError,
        "the card has no name for the language it_it, which a run measured",
    ),
    "the table of the variants, which the card no longer gives": (
        replace(INPUTS, template=f"{TEMPLATE}$variant_table\n"),
        CardError,
        "the template holds the placeholder variant_table, which nothing fills",
    ),
    "the files of every variant, which each variant's section now lists": (
        replace(INPUTS, template=f"{TEMPLATE}$shared_files\n"),
        CardError,
        "the template holds the placeholder shared_files, which nothing fills",
    ),
    "a compared reference run without a title": (
        change_description(reference_runs=(NEMO,)),
        CardError,
        "the description gives no title to the reference run transformers",
    ),
    "a variant without a run on the cpu or the gpu": (
        replace(INPUTS, runs=tuple(run for run in RUNS if run.variant != "float32")),
        VariantSummaryError,
        "the variant float32 has no run on the cpu or the gpu",
    ),
}


class TestTheModelCard:
    class TestTheRenderModelCardFunction:
        class TestWhenGivenTheRecordsOfAnAcceptedBuild:
            @pytest.fixture(scope="class")
            @classmethod
            def card(cls) -> str:
                return render_model_card(INPUTS).decode("utf-8")

            def test_should_head_a_variant_with_its_name_and_abbreviation_then_its_role_line(
                self, card: str
            ) -> None:
                assert (
                    "# Weights in ONNX\n\n### float32 (FP32)\n\nThe exported graph, "
                    "unconverted.\n\n- **Runs on:**" in card
                )

            def test_should_give_the_devices_of_its_runs_with_its_note_then_its_whole_size(
                self, card: str
            ) -> None:
                assert (
                    "- **Runs on:** CPU, GPU; the reference of the other variants\n"
                    "- **Size:** 2.55 GB\n\n| Language |" in card
                )

            def test_should_rate_it_by_language_against_a_reference_run_turned_around(
                self, card: str
            ) -> None:
                assert f"2.55 GB\n\n{FLOAT32_RATES}\nFiles:" in card

            def test_should_list_its_own_files_then_the_shared_ones_each_marked(
                self, card: str
            ) -> None:
                assert f"{FLOAT32_RATES}\n{FLOAT32_FILES}" in card

            def test_should_show_a_variant_without_abbreviation_nor_note_by_name_and_devices_alone(
                self, card: str
            ) -> None:
                assert (
                    "### int8\n\nThe encoder in 8-bit blocks.\n\n- **Runs on:** CPU\n"
                    "- **Size:** 0.77 GB\n\n" in card
                )

            def test_should_give_one_column_per_comparison_in_their_order(self, card: str) -> None:
                assert f"0.77 GB\n\n{EIGHT_BIT_BLOCKS_RATES}\nFiles:" in card

            def test_should_name_the_measured_languages_their_utterances_and_the_versions(
                self, card: str
            ) -> None:
                assert (
                    "Measured French and English (2 utterances in French, 3 in English) with "
                    "onnx-asr 0.12.0 and onnxruntime 1.31.0 on the CPU, and with onnx-asr 0.12.0 "
                    "and onnxruntime-gpu 1.31.0 on the GPU, the " in card
                )

            def test_should_fill_a_pinned_revision_with_the_revision_its_pin_gives(
                self, card: str
            ) -> None:
                assert f" the weights at {WEIGHTS_REVISION}.\n" in card

            def test_should_list_each_pinned_repository_once_at_its_revision(
                self, card: str
            ) -> None:
                assert card.endswith(
                    f"- [google/fleurs](https://huggingface.co/datasets/google/fleurs/tree/"
                    f"{FLEURS_REVISION})\n"
                    f"- [maker/weights](https://huggingface.co/maker/weights/tree/{WEIGHTS_REVISION})"
                    "\n"
                )

        class TestWhenNoComparisonJudgesTheRatedRuns:
            @pytest.fixture
            def card(self) -> str:
                return render_model_card(
                    replace(INPUTS, verification=replace(VERIFICATION, comparisons=()))
                ).decode("utf-8")

            def test_should_give_each_variant_its_rates_alone(self, card: str) -> None:
                assert (
                    "| Language | WER |\n|---|---|\n| French (fr_fr) | 5.00 % |\n"
                    "| English (en_us) | 4.00 % |\n\nFiles:" in card
                )

        class TestWhenAComparisonLacksALanguageThatTheRatedRunMeasured:
            @pytest.fixture
            def card(self) -> str:
                comparisons = tuple(
                    measure
                    for measure in VERIFICATION.comparisons
                    if measure != NEMO_AGAINST_FLOAT32_IN_ENGLISH
                )
                return render_model_card(
                    replace(INPUTS, verification=replace(VERIFICATION, comparisons=comparisons))
                ).decode("utf-8")

            def test_should_say_the_comparison_did_not_measure_it(self, card: str) -> None:
                assert "| English (en_us) | 4.00 % | not measured |\n" in card

        @pytest.mark.parametrize(
            ("inputs", "error", "message"),
            list(REFUSED_INPUTS.values()),
            ids=list(REFUSED_INPUTS.keys()),
        )
        class TestWhenTheCardHasNoWordForAnInput:
            @pytest.fixture
            def render(self, inputs: CardInputs) -> Callable[[], bytes]:
                return partial(render_model_card, inputs)

            def test_should_refuse_to_render_it_naming_the_input(
                self, render: Callable[[], bytes], error: type[ValueError], message: str
            ) -> None:
                with pytest.raises(error, match=re.escape(message)):
                    render()

    class TestTheJoinWordsFunction:
        class TestWhenGivenThreeLanguages:
            @pytest.fixture
            def joined_words(self) -> str:
                return join_words(THREE_WORDS)

            def test_should_separate_them_by_commas_and_the_last_by_and(
                self, joined_words: str
            ) -> None:
                assert joined_words == "French, English and German"
