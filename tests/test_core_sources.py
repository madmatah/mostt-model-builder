from pathlib import Path

import pytest

from mostt_model_builder.domain.bench.bench_languages import BENCH_LANGUAGES
from mostt_model_builder.domain.bench.fleurs_index import locate_bench_files
from mostt_model_builder.domain.sources.source_pins import (
    RepositoryType,
    SourcePin,
    parse_source_pins,
)

CORE_SOURCES_PATH = Path(__file__).resolve().parent.parent / "sources.toml"
FLEURS_REPOSITORY = "google/fleurs"
FLEURS_REVISION = "70bb2e84b976b7e960aa89f1c648e09c59f894dd"
BENCH_FILES = {path for language in BENCH_LANGUAGES for path in locate_bench_files(language)}


class TestTheCoreSources:
    class TestWhenTheCommittedSourcesAreRead:
        @pytest.fixture
        def core_pins(self) -> tuple[SourcePin, ...]:
            return parse_source_pins(CORE_SOURCES_PATH.read_text(encoding="utf-8"))

        def test_should_pin_the_index_and_the_audio_of_every_language_of_the_bench(
            self, core_pins: tuple[SourcePin, ...]
        ) -> None:
            assert {pin.path for pin in core_pins} == BENCH_FILES

        def test_should_pin_one_file_each(self, core_pins: tuple[SourcePin, ...]) -> None:
            assert len(core_pins) == len(BENCH_FILES)

        def test_should_read_them_from_the_fleurs_dataset_at_its_measured_revision(
            self, core_pins: tuple[SourcePin, ...]
        ) -> None:
            assert {(pin.repository_type, pin.repository, pin.revision) for pin in core_pins} == {
                (RepositoryType.DATASET, FLEURS_REPOSITORY, FLEURS_REVISION)
            }
