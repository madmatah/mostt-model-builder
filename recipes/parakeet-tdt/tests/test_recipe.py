from pathlib import Path

import pytest

from mostt_model_builder.domain.contract.recipe_contract import (
    CANDIDATE_MOUNT,
    EarlierOutputMount,
    JudgedRecipe,
    RecipeFacts,
    SourceMount,
    StepOutputOrigin,
    judge_recipe,
)
from mostt_model_builder.domain.sources.source_pins import SourcePin, parse_source_pins
from mostt_model_builder.host_main import CLONE_ROOT, find_present_entry_modules, read_pins
from mostt_recipe_parakeet_tdt.domain.transcripts.run_readers import (
    read_onnx_asr_lines,
    read_token_id_lines,
)
from mostt_recipe_parakeet_tdt.recipe import (
    DESCRIBE_STEP,
    EXPORT_IMAGE_NAME,
    EXPORT_STEP,
    RECIPE,
    REFERENCE_STEP,
    TRANSCRIBE_STEP,
    TRANSFORMERS_STEP,
)

RECIPE_SOURCES_PATH = Path(__file__).resolve().parent.parent / "sources.toml"
ULTRA_REPOSITORY = "moondream/parakeet-ultra"
ULTRA_REVISION = "73175eb7aeb0d82f1e2a6b53b3aabc10a90bcd0b"
NEMO_REPOSITORY = "nvidia/parakeet-tdt-0.6b-v3"
NEMO_REVISION = "541d1f99c6b0c3cd0b11a95167540bb8edefd82b"
NEMO_ARCHIVE = (NEMO_REPOSITORY, NEMO_REVISION, "parakeet-tdt-0.6b-v3.nemo")
PINNED_FILES = {
    (ULTRA_REPOSITORY, ULTRA_REVISION, "config.json"),
    (ULTRA_REPOSITORY, ULTRA_REVISION, "tokenizer.json"),
    (ULTRA_REPOSITORY, ULTRA_REVISION, "model.safetensors"),
    NEMO_ARCHIVE,
    (NEMO_REPOSITORY, NEMO_REVISION, "processor_config.json"),
    (NEMO_REPOSITORY, NEMO_REVISION, "tokenizer_config.json"),
    (NEMO_REPOSITORY, NEMO_REVISION, "generation_config.json"),
}
FLEURS_REPOSITORY = "google/fleurs"
EXPORTED_FILES = {
    SourceMount(ULTRA_REPOSITORY, "tokenizer.json"),
    SourceMount(ULTRA_REPOSITORY, "model.safetensors"),
    SourceMount(NEMO_REPOSITORY, "parakeet-tdt-0.6b-v3.nemo"),
    SourceMount(FLEURS_REPOSITORY, "data/fr_fr/test.tsv"),
    SourceMount(FLEURS_REPOSITORY, "data/fr_fr/audio/test.tar.gz"),
}
EXPORT_OUTPUTS = [
    "encoder-model.onnx",
    "encoder-model.onnx.data",
    "decoder_joint-model.onnx",
    "vocab.txt",
    "encoder-parity.json",
    "peaks.json",
]
BENCH_FILES = {
    SourceMount(FLEURS_REPOSITORY, "data/fr_fr/test.tsv"),
    SourceMount(FLEURS_REPOSITORY, "data/fr_fr/audio/test.tar.gz"),
    SourceMount(FLEURS_REPOSITORY, "data/en_us/test.tsv"),
    SourceMount(FLEURS_REPOSITORY, "data/en_us/audio/test.tar.gz"),
}
REFERENCE_FILES = {
    SourceMount(ULTRA_REPOSITORY, "tokenizer.json"),
    SourceMount(ULTRA_REPOSITORY, "model.safetensors"),
    SourceMount(NEMO_REPOSITORY, "parakeet-tdt-0.6b-v3.nemo"),
    *BENCH_FILES,
}
TRANSFORMERS_FILES = {
    SourceMount(ULTRA_REPOSITORY, "config.json"),
    SourceMount(ULTRA_REPOSITORY, "tokenizer.json"),
    SourceMount(ULTRA_REPOSITORY, "model.safetensors"),
    SourceMount(NEMO_REPOSITORY, "processor_config.json"),
    SourceMount(NEMO_REPOSITORY, "tokenizer_config.json"),
    SourceMount(NEMO_REPOSITORY, "generation_config.json"),
    SourceMount(FLEURS_REPOSITORY, "data/fr_fr/test.tsv"),
    SourceMount(FLEURS_REPOSITORY, "data/fr_fr/audio/test.tar.gz"),
    EarlierOutputMount("reference", "reference.safetensors"),
    EarlierOutputMount("reference", "reference.json"),
}
TRANSFORMERS_OUTPUTS = [
    "transformers-encoder-parity.json",
    "transformers-joint-parity.json",
    "peaks.json",
]
TRANSFORMERS_ROW_SOURCES = {
    (ULTRA_REPOSITORY, "config.json"),
    (ULTRA_REPOSITORY, "tokenizer.json"),
    (ULTRA_REPOSITORY, "model.safetensors"),
    (NEMO_REPOSITORY, "processor_config.json"),
    (NEMO_REPOSITORY, "tokenizer_config.json"),
    (NEMO_REPOSITORY, "generation_config.json"),
    (FLEURS_REPOSITORY, "data/fr_fr/test.tsv"),
    (FLEURS_REPOSITORY, "data/fr_fr/audio/test.tar.gz"),
}
TRANSFORMERS_ROW_INPUTS = {
    ("reference", "reference.safetensors"),
    ("reference", "reference.json"),
}
PARITIES = [
    ("export", "encoder-parity.json", 5e-5),
    ("transformers-reference", "transformers-encoder-parity.json", 1.5e-5),
    ("transformers-reference", "transformers-joint-parity.json", 5.3e-4),
]
BOTH_LANGUAGES = ("fr_fr", "en_us")
RUNS = [
    ("nemo", "reference", "cpu", "reference", BOTH_LANGUAGES, 1, read_token_id_lines),
    (
        "transformers",
        "transformers-reference",
        "cpu",
        "reference",
        ("fr_fr",),
        1,
        read_token_id_lines,
    ),
    ("float32-cpu", "transcribe", "cpu", "float32", BOTH_LANGUAGES, 4, read_onnx_asr_lines),
    ("float32-gpu", "transcribe", "gpu", "float32", BOTH_LANGUAGES, 1, read_onnx_asr_lines),
]
DERIVED_ROW_NAMES = [
    "reference",
    "transformers-reference",
    "transcribe-float32-cpu-0",
    "transcribe-float32-cpu-1",
    "transcribe-float32-cpu-2",
    "transcribe-float32-cpu-3",
    "transcribe-float32-gpu-0",
]
FLOAT32_CANDIDATE = [
    ("encoder-model.onnx", "onnx", None),
    ("encoder-model.onnx.data", "onnx-external-data", "encoder-model.onnx"),
    ("decoder_joint-model.onnx", "onnx", None),
    ("vocab.txt", "text", None),
    ("config.json", "text", None),
]
HOST_CPU_COUNT = 20


def judge_the_recipe() -> JudgedRecipe:
    return judge_recipe(
        RECIPE,
        RecipeFacts(
            core_pins=read_pins(CLONE_ROOT),
            recipe_pins=read_pins(RECIPE_SOURCES_PATH.parent),
            present_entry_modules=find_present_entry_modules(RECIPE),
            host_cpu_count=HOST_CPU_COUNT,
        ),
    )


DESCRIBED_FILES = {
    SourceMount(ULTRA_REPOSITORY, "model.safetensors"),
    SourceMount(NEMO_REPOSITORY, "parakeet-tdt-0.6b-v3.nemo"),
}


class TestTheParakeetTdtRecipe:
    class TestItsSources:
        @pytest.fixture
        def recipe_pins(self) -> tuple[SourcePin, ...]:
            return parse_source_pins(RECIPE_SOURCES_PATH.read_text(encoding="utf-8"))

        def test_should_pin_moondreams_three_files_and_v3s_nemo_and_processor_at_their_revisions(
            self, recipe_pins: tuple[SourcePin, ...]
        ) -> None:
            assert {(pin.repository, pin.revision, pin.path) for pin in recipe_pins} == (
                PINNED_FILES
            )

        def test_should_mark_the_nemo_alone_as_a_file_that_runs_code_when_it_loads(
            self, recipe_pins: tuple[SourcePin, ...]
        ) -> None:
            assert [
                (pin.repository, pin.revision, pin.path)
                for pin in recipe_pins
                if pin.runs_code_when_loaded
            ] == [NEMO_ARCHIVE]

    class TestItsImages:
        def test_should_let_the_export_image_alone_read_the_nemo(self) -> None:
            assert [image.name for image in RECIPE.images if image.reads_code_running_sources] == [
                EXPORT_IMAGE_NAME
            ]

    class TestItsDescribeStep:
        def test_should_bind_moondreams_weights_and_the_nemo_alone(self) -> None:
            assert set(DESCRIBE_STEP.mounts) == DESCRIBED_FILES

        def test_should_bind_each_file_once(self) -> None:
            assert len(DESCRIBE_STEP.mounts) == len(DESCRIBED_FILES)

        def test_should_run_in_the_export_image(self) -> None:
            assert DESCRIBE_STEP.image == EXPORT_IMAGE_NAME

    class TestItsExportStep:
        def test_should_bind_moondreams_tokenizer_and_weights_the_nemo_and_the_french_bench_alone(
            self,
        ) -> None:
            assert set(EXPORT_STEP.mounts) == EXPORTED_FILES

        def test_should_bind_each_file_once(self) -> None:
            assert len(EXPORT_STEP.mounts) == len(EXPORTED_FILES)

        def test_should_run_in_the_export_image(self) -> None:
            assert EXPORT_STEP.image == EXPORT_IMAGE_NAME

        def test_should_write_the_files_of_onnx_asr_its_parity_and_its_peaks(self) -> None:
            assert [output.name for output in EXPORT_STEP.outputs] == EXPORT_OUTPUTS

    class TestItsParities:
        def test_should_hold_the_onnx_encoder_and_transformers_encoder_and_joint_against_nemo(
            self,
        ) -> None:
            assert [
                (parity.step, parity.output, parity.absolute_difference_tolerance)
                for parity in RECIPE.parities
            ] == PARITIES

    class TestItsReferenceStep:
        def test_should_bind_moondreams_tokenizer_and_weights_the_nemo_and_both_benches_alone(
            self,
        ) -> None:
            assert set(REFERENCE_STEP.mounts) == REFERENCE_FILES

        def test_should_bind_each_file_once(self) -> None:
            assert len(REFERENCE_STEP.mounts) == len(REFERENCE_FILES)

        def test_should_run_on_the_cpu_in_the_export_image_since_it_reads_the_nemo(self) -> None:
            assert [(template.device, template.image) for template in REFERENCE_STEP.devices] == [
                ("cpu", EXPORT_IMAGE_NAME)
            ]

    class TestItsTransformersReferenceStep:
        def test_should_bind_moondreams_files_v3s_processor_french_and_the_references_dump_alone(
            self,
        ) -> None:
            assert set(TRANSFORMERS_STEP.mounts) == TRANSFORMERS_FILES

        def test_should_bind_each_file_once(self) -> None:
            assert len(TRANSFORMERS_STEP.mounts) == len(TRANSFORMERS_FILES)

        def test_should_run_on_the_cpu_in_its_own_image(self) -> None:
            assert [
                (template.device, template.image) for template in TRANSFORMERS_STEP.devices
            ] == [("cpu", "transformers-reference")]

        def test_should_write_its_two_parities_and_its_peaks(self) -> None:
            assert [output.name for output in TRANSFORMERS_STEP.outputs] == TRANSFORMERS_OUTPUTS

    class TestItsTranscribeStep:
        def test_should_bind_the_candidate_and_both_benches_alone(self) -> None:
            assert set(TRANSCRIBE_STEP.mounts) == {CANDIDATE_MOUNT, *BENCH_FILES}

        def test_should_run_each_device_in_its_own_image(self) -> None:
            assert [(template.device, template.image) for template in TRANSCRIBE_STEP.devices] == [
                ("cpu", "transcribe-cpu"),
                ("gpu", "transcribe-gpu"),
            ]

    class TestItsRuns:
        def test_should_run_nemo_transformers_in_french_and_float32_on_both_devices(self) -> None:
            assert [
                (
                    run.name,
                    run.step,
                    run.device,
                    run.variant,
                    run.languages,
                    run.shard_count,
                    run.reader,
                )
                for run in RECIPE.runs
            ] == RUNS

    class TestItsCandidate:
        def test_should_hold_the_float32_files_of_onnx_asr(self) -> None:
            assert [
                (candidate_file.name, candidate_file.file_format, candidate_file.external_data_of)
                for candidate_file in RECIPE.candidate
            ] == FLOAT32_CANDIDATE

        def test_should_convert_the_exported_encoder(self) -> None:
            assert RECIPE.conversion_input == StepOutputOrigin("export", "encoder-model.onnx")

    class TestItsJudgment:
        @pytest.fixture(scope="class")
        @classmethod
        def judged_recipe(cls) -> JudgedRecipe:
            return judge_the_recipe()

        def test_should_derive_one_reference_row_and_a_row_per_shard_of_the_transcriptions(
            self, judged_recipe: JudgedRecipe
        ) -> None:
            assert [
                derived_row.row.name for derived_row in judged_recipe.derived_rows
            ] == DERIVED_ROW_NAMES

        def test_should_build_from_the_bench_to_the_transcriptions(
            self, judged_recipe: JudgedRecipe
        ) -> None:
            assert judged_recipe.build_order == (
                "fleurs-check",
                "export",
                "inspect",
                "reference",
                "transformers-reference",
                "transcribe",
            )

        def test_should_bind_each_pinned_file_of_the_transformers_row_by_itself_and_no_nemo(
            self, judged_recipe: JudgedRecipe
        ) -> None:
            assert {
                (source.repository, source.path)
                for row in judged_recipe.rows
                if row.name == "transformers-reference"
                for source in row.sources
            } == TRANSFORMERS_ROW_SOURCES

        def test_should_bind_each_dumped_file_of_the_reference_to_the_transformers_row_by_itself(
            self, judged_recipe: JudgedRecipe
        ) -> None:
            assert {
                (earlier_output.step, earlier_output.output.name)
                for row in judged_recipe.rows
                if row.name == "transformers-reference"
                for earlier_output in row.earlier_outputs
            } == TRANSFORMERS_ROW_INPUTS
