from pathlib import Path

import pytest

from mostt_model_builder.domain.contract.recipe_contract import (
    BENCH_MOUNT,
    CANDIDATE_MOUNT,
    EarlierOutputMount,
    HostContentOrigin,
    JudgedRecipe,
    RecipeFacts,
    SourceMount,
    StepOutputOrigin,
    judge_recipe,
)
from mostt_model_builder.domain.sources.source_pins import SourcePin, parse_source_pins
from mostt_model_builder.host_main import (
    CLONE_ROOT,
    find_present_entry_modules,
    is_card_template_present,
    read_pins,
)
from mostt_recipe_parakeet_tdt.domain.publication.model_description import MODEL_DESCRIPTION
from mostt_recipe_parakeet_tdt.domain.transcripts.run_readers import (
    read_onnx_asr_lines,
    read_token_id_lines,
)
from mostt_recipe_parakeet_tdt.domain.transcripts.vocabulary import VOCABULARY_SHA256
from mostt_recipe_parakeet_tdt.domain.variants.candidate_layout import read_quantization
from mostt_recipe_parakeet_tdt.domain.variants.comparisons import COMPARISONS
from mostt_recipe_parakeet_tdt.domain.variants.quantizations import find_quantization
from mostt_recipe_parakeet_tdt.recipe import (
    DESCRIBE_STEP,
    EXPORT_IMAGE_NAME,
    EXPORT_STEP,
    RECIPE,
    REFERENCE_DUMP_STEP,
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
REFERENCE_FILES = {
    SourceMount(ULTRA_REPOSITORY, "tokenizer.json"),
    SourceMount(ULTRA_REPOSITORY, "model.safetensors"),
    SourceMount(NEMO_REPOSITORY, "parakeet-tdt-0.6b-v3.nemo"),
    BENCH_MOUNT,
}
DUMPED_FILES = {
    SourceMount(ULTRA_REPOSITORY, "tokenizer.json"),
    SourceMount(ULTRA_REPOSITORY, "model.safetensors"),
    SourceMount(NEMO_REPOSITORY, "parakeet-tdt-0.6b-v3.nemo"),
    SourceMount(FLEURS_REPOSITORY, "data/fr_fr/test.tsv"),
    SourceMount(FLEURS_REPOSITORY, "data/fr_fr/audio/test.tar.gz"),
}
DUMP_OUTPUTS = ["reference.safetensors", "reference.json", "peaks.json"]
TRANSFORMERS_FILES = {
    SourceMount(ULTRA_REPOSITORY, "config.json"),
    SourceMount(ULTRA_REPOSITORY, "tokenizer.json"),
    SourceMount(ULTRA_REPOSITORY, "model.safetensors"),
    SourceMount(NEMO_REPOSITORY, "processor_config.json"),
    SourceMount(NEMO_REPOSITORY, "tokenizer_config.json"),
    SourceMount(NEMO_REPOSITORY, "generation_config.json"),
    BENCH_MOUNT,
    EarlierOutputMount("reference-dump", "reference.safetensors"),
    EarlierOutputMount("reference-dump", "reference.json"),
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
    ("reference-dump", "reference.safetensors"),
    ("reference-dump", "reference.json"),
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
    ("float16-gpu", "transcribe", "gpu", "float16", BOTH_LANGUAGES, 1, read_onnx_asr_lines),
    ("bfloat16-gpu", "transcribe", "gpu", "bfloat16", BOTH_LANGUAGES, 1, read_onnx_asr_lines),
    (
        "eight-bit-blocks-gpu",
        "transcribe",
        "gpu",
        "eight-bit-blocks",
        BOTH_LANGUAGES,
        1,
        read_onnx_asr_lines,
    ),
    (
        "eight-bit-blocks-cpu",
        "transcribe",
        "cpu",
        "eight-bit-blocks",
        BOTH_LANGUAGES,
        4,
        read_onnx_asr_lines,
    ),
]
TRANSCRIPTION_SHARD_COUNTS = {"cpu": 4, "gpu": 1}
DERIVED_ROW_NAMES = [
    "reference-nemo-fr_fr-0",
    "reference-nemo-en_us-0",
    "transformers-reference-transformers-fr_fr-0",
    *(
        f"transcribe-{run}-{language}-{shard}"
        for run, device in (
            ("float32-cpu", "cpu"),
            ("float32-gpu", "gpu"),
            ("float16-gpu", "gpu"),
            ("bfloat16-gpu", "gpu"),
            ("eight-bit-blocks-gpu", "gpu"),
            ("eight-bit-blocks-cpu", "cpu"),
        )
        for language in BOTH_LANGUAGES
        for shard in range(TRANSCRIPTION_SHARD_COUNTS[device])
    ),
]
TRANSFORMERS_ROW_NAME = "transformers-reference-transformers-fr_fr-0"
ALL_VARIANTS = ("float32", "float16", "bfloat16", "eight-bit-blocks")
EXPORTED_DECODER = StepOutputOrigin("export", "decoder_joint-model.onnx")
PUBLISHED_LAYOUT = [
    (
        "encoder-model.onnx",
        "onnx",
        ("float32",),
        StepOutputOrigin("export", "encoder-model.onnx"),
        None,
    ),
    (
        "encoder-model.onnx.data",
        "onnx-external-data",
        ("float32",),
        StepOutputOrigin("export", "encoder-model.onnx.data"),
        "encoder-model.onnx",
    ),
    (
        "encoder-model.fp16.onnx",
        "onnx",
        ("float16",),
        StepOutputOrigin("convert", "encoder-model.fp16.onnx"),
        None,
    ),
    (
        "encoder-model.bf16.onnx",
        "onnx",
        ("bfloat16",),
        StepOutputOrigin("convert", "encoder-model.bf16.onnx"),
        None,
    ),
    (
        "encoder-model.int8.onnx",
        "onnx",
        ("eight-bit-blocks",),
        StepOutputOrigin("convert", "encoder-model.int8.onnx"),
        None,
    ),
    ("decoder_joint-model.onnx", "onnx", ("float32",), EXPORTED_DECODER, None),
    ("decoder_joint-model.fp16.onnx", "onnx", ("float16",), EXPORTED_DECODER, None),
    ("decoder_joint-model.bf16.onnx", "onnx", ("bfloat16",), EXPORTED_DECODER, None),
    ("decoder_joint-model.int8.onnx", "onnx", ("eight-bit-blocks",), EXPORTED_DECODER, None),
    ("vocab.txt", "text", ALL_VARIANTS, StepOutputOrigin("export", "vocab.txt"), None),
    (
        "config.json",
        "text",
        ALL_VARIANTS,
        HostContentOrigin(
            b'{"model_type": "nemo-conformer-tdt", "features_size": 128, '
            b'"subsampling_factor": 8, "max_tokens_per_step": 10}'
        ),
        None,
    ),
]
PUBLISHED_VOCABULARY_SHA256 = "d58544679ea4bc6ac563d1f545eb7d474bd6cfa467f0a6e2c1dc1c7d37e3c35d"
HOST_CPU_COUNT = 20


def judge_the_recipe() -> JudgedRecipe:
    return judge_recipe(
        RECIPE,
        RecipeFacts(
            core_pins=read_pins(CLONE_ROOT),
            recipe_pins=read_pins(RECIPE_SOURCES_PATH.parent),
            present_entry_modules=find_present_entry_modules(RECIPE),
            host_cpu_count=HOST_CPU_COUNT,
            has_card_template=is_card_template_present(RECIPE, CLONE_ROOT),
        ),
    )


TRANSCRIPTION_ROW_COUNT = len(
    [name for name in DERIVED_ROW_NAMES if name.startswith("transcribe-")]
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
        def test_should_bind_moondreams_tokenizer_and_weights_the_nemo_and_its_rows_bench_alone(
            self,
        ) -> None:
            assert set(REFERENCE_STEP.mounts) == REFERENCE_FILES

        def test_should_bind_each_file_once(self) -> None:
            assert len(REFERENCE_STEP.mounts) == len(REFERENCE_FILES)

        def test_should_run_on_the_cpu_in_the_export_image_since_it_reads_the_nemo(self) -> None:
            assert [(template.device, template.image) for template in REFERENCE_STEP.devices] == [
                ("cpu", EXPORT_IMAGE_NAME)
            ]

        def test_should_write_its_peaks_alone_beside_the_lines_of_its_runs_so_it_dumps_nothing(
            self,
        ) -> None:
            assert [output.name for output in REFERENCE_STEP.outputs] == ["peaks.json"]

    class TestItsReferenceDumpStep:
        def test_should_bind_moondreams_tokenizer_and_weights_the_nemo_and_the_french_bench_alone(
            self,
        ) -> None:
            assert set(REFERENCE_DUMP_STEP.mounts) == DUMPED_FILES

        def test_should_bind_each_file_once(self) -> None:
            assert len(REFERENCE_DUMP_STEP.mounts) == len(DUMPED_FILES)

        def test_should_run_in_the_export_image_since_it_reads_the_nemo(self) -> None:
            assert REFERENCE_DUMP_STEP.image == EXPORT_IMAGE_NAME

        def test_should_write_the_tensors_and_the_steps_of_the_dump_and_its_peaks(self) -> None:
            assert [output.name for output in REFERENCE_DUMP_STEP.outputs] == DUMP_OUTPUTS

    class TestItsTransformersReferenceStep:
        def test_should_bind_moondreams_files_v3s_processor_its_rows_bench_and_the_dump_alone(
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
        def test_should_bind_the_candidate_and_its_rows_bench_alone(self) -> None:
            assert set(TRANSCRIBE_STEP.mounts) == {CANDIDATE_MOUNT, BENCH_MOUNT}

        def test_should_run_four_shards_of_two_cpus_on_the_cpu_and_one_of_eight_on_the_gpu(
            self,
        ) -> None:
            shards_by_device = {
                run.device: run.shard_count for run in RECIPE.runs if run.step == "transcribe"
            }
            assert [
                (template.device, shards_by_device[template.device], template.limits.cpu_count)
                for template in TRANSCRIBE_STEP.devices
            ] == [("cpu", 4, 2), ("gpu", 1, 8)]

        def test_should_run_each_device_in_its_own_image(self) -> None:
            assert [(template.device, template.image) for template in TRANSCRIBE_STEP.devices] == [
                ("cpu", "transcribe-cpu"),
                ("gpu", "transcribe-gpu"),
            ]

    class TestItsRuns:
        def test_should_run_nemo_transformers_in_french_and_each_variant_on_its_devices(
            self,
        ) -> None:
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
        def test_should_hold_the_published_layout_of_onnx_asr_file_by_file(self) -> None:
            assert [
                (
                    candidate_file.name,
                    candidate_file.file_format,
                    candidate_file.variants,
                    candidate_file.origin,
                    candidate_file.external_data_of,
                )
                for candidate_file in RECIPE.candidate
            ] == PUBLISHED_LAYOUT

        def test_should_accept_the_vocabulary_at_the_sha256_of_v3s_alone(self) -> None:
            assert VOCABULARY_SHA256 == PUBLISHED_VOCABULARY_SHA256

        def test_should_convert_the_exported_encoder(self) -> None:
            assert RECIPE.conversion_input == StepOutputOrigin("export", "encoder-model.onnx")

    class TestItsJudgment:
        @pytest.fixture(scope="class")
        @classmethod
        def judged_recipe(cls) -> JudgedRecipe:
            return judge_the_recipe()

        def test_should_derive_a_row_per_language_and_shard_of_each_run(
            self, judged_recipe: JudgedRecipe
        ) -> None:
            assert [
                derived_row.row.name for derived_row in judged_recipe.derived_rows
            ] == DERIVED_ROW_NAMES

        def test_should_name_in_each_transcription_s_request_the_encoder_of_its_quantization(
            self, judged_recipe: JudgedRecipe
        ) -> None:
            transcriptions = [
                derived_row.request
                for derived_row in judged_recipe.derived_rows
                if derived_row.step == TRANSCRIBE_STEP.name
            ]
            assert len(transcriptions) == TRANSCRIPTION_ROW_COUNT
            assert [
                request.variant
                for request in transcriptions
                if read_quantization(request.files) != find_quantization(request.variant)
            ] == []

        def test_should_bind_each_row_of_a_run_to_the_bench_of_its_own_language_alone(
            self, judged_recipe: JudgedRecipe
        ) -> None:
            mismatches = [
                derived_row.row.name
                for derived_row in judged_recipe.derived_rows
                if {
                    source.path
                    for source in derived_row.row.sources
                    if source.repository == FLEURS_REPOSITORY
                }
                != {
                    f"data/{derived_row.language}/test.tsv",
                    f"data/{derived_row.language}/audio/test.tar.gz",
                }
            ]
            assert len(judged_recipe.derived_rows) == len(DERIVED_ROW_NAMES)
            assert mismatches == []

        def test_should_read_the_parities_of_transformers_from_the_one_row_of_its_run(
            self, judged_recipe: JudgedRecipe
        ) -> None:
            assert [parity.step for parity in judged_recipe.parities] == [
                "export",
                TRANSFORMERS_ROW_NAME,
                TRANSFORMERS_ROW_NAME,
            ]

        def test_should_carry_the_description_of_its_model(
            self, judged_recipe: JudgedRecipe
        ) -> None:
            assert judged_recipe.model == MODEL_DESCRIPTION

        def test_should_keep_every_comparison_of_its_runs(
            self, judged_recipe: JudgedRecipe
        ) -> None:
            assert judged_recipe.comparisons == COMPARISONS

        def test_should_build_from_the_bench_to_the_transcriptions(
            self, judged_recipe: JudgedRecipe
        ) -> None:
            assert judged_recipe.build_order == (
                "fleurs-check",
                "export",
                "convert",
                "inspect",
                "reference",
                "reference-dump",
                "transformers-reference",
                "transcribe",
            )

        def test_should_convert_the_exported_encoder_bound_beside_its_external_data(
            self, judged_recipe: JudgedRecipe
        ) -> None:
            assert [
                (earlier_output.step, earlier_output.output.name)
                for row in judged_recipe.rows
                if row.name == "convert"
                for earlier_output in row.earlier_outputs
            ] == [("export", "encoder-model.onnx"), ("export", "encoder-model.onnx.data")]

        def test_should_bind_each_pinned_file_of_the_transformers_row_by_itself_and_no_nemo(
            self, judged_recipe: JudgedRecipe
        ) -> None:
            assert {
                (source.repository, source.path)
                for row in judged_recipe.rows
                if row.name == TRANSFORMERS_ROW_NAME
                for source in row.sources
            } == TRANSFORMERS_ROW_SOURCES

        def test_should_bind_each_dumped_file_of_the_reference_to_the_transformers_row_by_itself(
            self, judged_recipe: JudgedRecipe
        ) -> None:
            assert {
                (earlier_output.step, earlier_output.output.name)
                for row in judged_recipe.rows
                if row.name == TRANSFORMERS_ROW_NAME
                for earlier_output in row.earlier_outputs
            } == TRANSFORMERS_ROW_INPUTS
