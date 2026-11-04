import re
from collections.abc import Callable
from dataclasses import replace
from functools import partial

import pytest

from mostt_model_builder.domain.publication.publication_request import (
    PublicationRequest,
    PublishedFile,
)
from mostt_model_builder.domain.publication.requested_files import (
    RequestedFilesError,
    judge_requested_files,
)

FILES = (
    PublishedFile("encoder-model.onnx", 4096, "a" * 64),
    PublishedFile("README.md", 900, "c" * 64),
)
REQUEST = PublicationRequest("example-user/parakeet-ultra-onnx", "Publish it", FILES[:1], FILES[1])
SUBJECT = "the upload"
DISAGREEING_FILES = {
    "a missing file": (FILES[:1], "the upload lacks README.md, which the request names"),
    "an extra file": (
        (*FILES, PublishedFile("model.safetensors", 7, "d" * 64)),
        "the upload names model.safetensors, which the request does not",
    ),
    "a file named twice": ((*FILES, FILES[0]), "the upload names one file twice"),
    "a size that differs": (
        (replace(FILES[0], size_bytes=4095), FILES[1]),
        f"the upload gives 4095 bytes of SHA-256 {'a' * 64} for encoder-model.onnx, where the "
        f"verification accepted 4096 bytes of SHA-256 {'a' * 64}",
    ),
    "a hash that differs": (
        (FILES[0], replace(FILES[1], sha256="e" * 64)),
        f"for README.md, where the verification accepted 900 bytes of SHA-256 {'c' * 64}",
    ),
}


class TestTheRequestedFiles:
    class TestTheJudgeRequestedFilesFunction:
        class TestWhenTheFilesAreThoseOfTheRequestInAnotherOrder:
            @pytest.fixture
            def judge_act(self) -> Callable[[], object]:
                return partial(judge_requested_files, FILES[::-1], REQUEST, SUBJECT)

            def test_should_accept_them(self, judge_act: Callable[[], object]) -> None:
                assert judge_act() is None

        @pytest.mark.parametrize(
            ("files", "reason"),
            list(DISAGREEING_FILES.values()),
            ids=list(DISAGREEING_FILES.keys()),
        )
        class TestWhenTheFilesDisagreeWithTheRequest:
            @pytest.fixture
            def judge_act(self, files: tuple[PublishedFile, ...]) -> Callable[[], None]:
                return partial(judge_requested_files, files, REQUEST, SUBJECT)

            def test_should_refuse_them_and_say_why(
                self, judge_act: Callable[[], None], reason: str
            ) -> None:
                with pytest.raises(RequestedFilesError, match=re.escape(reason)):
                    judge_act()
