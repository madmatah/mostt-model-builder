import re
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import pytest

from mostt_model_builder.steps.bound_sources import BoundSourceError, find_bound_source

REPOSITORY = "moondream/parakeet-ultra"
PATH = "model.safetensors"
REVISION = "73175eb7aeb0d82f1e2a6b53b3aabc10a90bcd0b"
OTHER_REVISION = "0123456789abcdef0123456789abcdef01234567"


def bind_source(sources_directory: Path, revision: str) -> Path:
    bound_path = sources_directory / REPOSITORY / revision / PATH
    bound_path.parent.mkdir(parents=True)
    bound_path.write_bytes(b"weights")
    return bound_path


@dataclass(frozen=True)
class FoundSource:
    found_path: Path
    bound_path: Path


class TestTheBoundSources:
    class TestTheFindBoundSourceFunction:
        class TestWhenTheRowBindsTheFileAtItsRevision:
            @pytest.fixture
            def found_source(self, tmp_path: Path) -> FoundSource:
                bound_path = bind_source(tmp_path, REVISION)
                return FoundSource(find_bound_source(REPOSITORY, PATH, tmp_path), bound_path)

            def test_should_find_it_under_its_revision(self, found_source: FoundSource) -> None:
                assert found_source.found_path == found_source.bound_path

        class TestWhenNoFileIsBound:
            @pytest.fixture
            def find_act(self, tmp_path: Path) -> Callable[[], Path]:
                return partial(find_bound_source, REPOSITORY, PATH, tmp_path)

            def test_should_refuse_it_and_count_none(self, find_act: Callable[[], Path]) -> None:
                with pytest.raises(
                    BoundSourceError,
                    match=re.escape(
                        f"0 files are bound for {PATH} of {REPOSITORY}, where its step expects one"
                    ),
                ):
                    find_act()

        class TestWhenTwoRevisionsOfTheFileAreBound:
            @pytest.fixture
            def find_act(self, tmp_path: Path) -> Callable[[], Path]:
                bind_source(tmp_path, REVISION)
                bind_source(tmp_path, OTHER_REVISION)
                return partial(find_bound_source, REPOSITORY, PATH, tmp_path)

            def test_should_refuse_it_since_its_step_expects_one(
                self, find_act: Callable[[], Path]
            ) -> None:
                with pytest.raises(BoundSourceError, match=r"^2 files are bound for"):
                    find_act()
