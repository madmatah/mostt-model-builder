import re
from collections.abc import Callable
from functools import partial

import pytest

from mostt_model_builder.domain.publication.repository_files import (
    RepositoryFilesError,
    judge_repository_confirmation,
    judge_repository_files,
    judge_repository_name,
)

REPOSITORY = "example-user/parakeet-ultra-onnx"
PUBLISHED_NAMES = ("encoder-model.onnx", "vocab.txt", "README.md")
ACCEPTED_LISTINGS = {
    "a repository the Hub just created": [".gitattributes"],
    "an earlier publication of the build": [".gitattributes", *PUBLISHED_NAMES],
}


class TestTheRepositoryFiles:
    class TestTheJudgeRepositoryFilesFunction:
        @pytest.mark.parametrize(
            "listed_files", list(ACCEPTED_LISTINGS.values()), ids=list(ACCEPTED_LISTINGS.keys())
        )
        class TestWhenTheRepositoryHoldsNothingElse:
            @pytest.fixture
            def judge_act(self, listed_files: list[str]) -> Callable[[], object]:
                return partial(judge_repository_files, listed_files, PUBLISHED_NAMES)

            def test_should_accept_it(self, judge_act: Callable[[], object]) -> None:
                assert judge_act() is None

        class TestWhenTheRepositoryHoldsAFileOfNoBuild:
            @pytest.fixture
            def judge_act(self) -> Callable[[], None]:
                return partial(
                    judge_repository_files,
                    [".gitattributes", "model.safetensors", "config.json"],
                    PUBLISHED_NAMES,
                )

            def test_should_refuse_it_since_it_may_be_another_repository(
                self, judge_act: Callable[[], None]
            ) -> None:
                with pytest.raises(
                    RepositoryFilesError,
                    match=re.escape(
                        "the repository holds 2 files that are not the build's, such as "
                        "'config.json'"
                    ),
                ):
                    judge_act()

    class TestTheJudgeRepositoryConfirmationFunction:
        class TestWhenTheTypedRepositoryDiffers:
            @pytest.fixture
            def judge_act(self) -> Callable[[], None]:
                return partial(
                    judge_repository_confirmation, REPOSITORY, "example-user/parakeet-ultra-onx"
                )

            def test_should_stop_before_anything_is_published(
                self, judge_act: Callable[[], None]
            ) -> None:
                with pytest.raises(
                    RepositoryFilesError,
                    match=re.escape(
                        f"the repository typed again differs from {REPOSITORY}: nothing was "
                        "published"
                    ),
                ):
                    judge_act()

        class TestWhenTheTypedRepositoryIsTheSame:
            @pytest.fixture
            def judge_act(self) -> Callable[[], object]:
                return partial(judge_repository_confirmation, REPOSITORY, REPOSITORY)

            def test_should_go_on(self, judge_act: Callable[[], object]) -> None:
                assert judge_act() is None

    class TestTheJudgeRepositoryNameFunction:
        class TestWhenTheNameIsAPlaceholder:
            @pytest.fixture
            def judge_act(self) -> Callable[[], None]:
                return partial(judge_repository_name, "<user>/parakeet-ultra-onnx")

            def test_should_refuse_it(self, judge_act: Callable[[], None]) -> None:
                with pytest.raises(
                    RepositoryFilesError,
                    match=re.escape("the repository '<user>/parakeet-ultra-onnx' is not"),
                ):
                    judge_act()
