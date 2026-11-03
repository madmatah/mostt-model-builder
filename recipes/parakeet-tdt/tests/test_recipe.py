from pathlib import Path

import pytest

from mostt_model_builder.domain.contract.recipe_contract import SourceMount
from mostt_model_builder.domain.sources.source_pins import SourcePin, parse_source_pins
from mostt_recipe_parakeet_tdt.recipe import DESCRIBE_STEP, EXPORT_IMAGE_NAME, RECIPE

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
}
DESCRIBED_FILES = {
    SourceMount(ULTRA_REPOSITORY, "model.safetensors"),
    SourceMount(NEMO_REPOSITORY, "parakeet-tdt-0.6b-v3.nemo"),
}


class TestTheParakeetTdtRecipe:
    class TestItsSources:
        @pytest.fixture
        def recipe_pins(self) -> tuple[SourcePin, ...]:
            return parse_source_pins(RECIPE_SOURCES_PATH.read_text(encoding="utf-8"))

        def test_should_pin_moondreams_three_files_and_the_v3_nemo_at_their_revisions(
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
