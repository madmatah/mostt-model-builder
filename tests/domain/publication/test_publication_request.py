import json
import re
from collections.abc import Callable
from functools import partial

import pytest

from mostt_model_builder.domain.publication.publication_request import (
    PublicationRequest,
    PublicationRequestError,
    PublishedFile,
    describe_commit_message,
    parse_publication_request,
    render_publication_request,
)

REPOSITORY = "example-user/parakeet-ultra-onnx"
REQUEST = PublicationRequest(
    repository=REPOSITORY,
    commit_message="Publish Parakeet Ultra in ONNX",
    candidate=(
        PublishedFile("encoder-model.onnx", 4096, "a" * 64),
        PublishedFile("vocab.txt", 21, "b" * 64),
    ),
    card=PublishedFile("README.md", 900, "c" * 64),
)
DOCUMENT = json.loads(render_publication_request(REQUEST))


def change_document(**fields: object) -> bytes:
    return json.dumps({**DOCUMENT, **fields}).encode()


REFUSED_REQUESTS = {
    "a document beyond its bound": (b" " * (64 * 1024 + 1), "holds more than 65536 bytes"),
    "a document that is not ASCII": ('{"repository": "é"}'.encode(), "is not ASCII JSON"),
    "a key twice": (
        b'{"repository": "a/b", "repository": "a/b"}',
        "a key appears twice in an object",
    ),
    "a missing key": (
        json.dumps({key: DOCUMENT[key] for key in ("repository", "candidate", "card")}).encode(),
        "the request is not an object of exactly",
    ),
    "a repository outside its grammar": (
        change_document(repository="<user>/parakeet-ultra-onnx"),
        "the request names the repository '<user>/parakeet-ultra-onnx'",
    ),
    "an empty commit message": (change_document(commit_message=""), "commit message is ''"),
    "a commit message on two lines": (
        change_document(commit_message="Publish\nit"),
        "commit message is 'Publish\\nit'",
    ),
    "no candidate": (change_document(candidate=[]), "is not a list of 1 to 256 files"),
    "a file outside the grammar of names": (
        change_document(candidate=[{"name": "../x", "size_bytes": 1, "sha256": "a" * 64}]),
        "is named '../x'",
    ),
    "an empty file": (
        change_document(candidate=[{"name": "x.onnx", "size_bytes": 0, "sha256": "a" * 64}]),
        "holds 0 bytes",
    ),
    "a size that is a boolean": (
        change_document(candidate=[{"name": "x.onnx", "size_bytes": True, "sha256": "a" * 64}]),
        "holds True bytes",
    ),
    "a hash outside its grammar": (
        change_document(candidate=[{"name": "x.onnx", "size_bytes": 1, "sha256": "A" * 64}]),
        "has the SHA-256",
    ),
    "a card under another name": (
        change_document(card={"name": "card.md", "size_bytes": 1, "sha256": "a" * 64}),
        "the request's card is not named README.md",
    ),
    "a file named twice": (
        change_document(
            card={"name": "README.md", "size_bytes": 1, "sha256": "a" * 64},
            candidate=[{"name": "README.md", "size_bytes": 1, "sha256": "a" * 64}],
        ),
        "the request names one file twice",
    ),
}


class TestThePublicationRequest:
    class TestTheParsePublicationRequestFunction:
        class TestWhenGivenARenderedRequest:
            @pytest.fixture
            def publication_request(self) -> PublicationRequest:
                return parse_publication_request(render_publication_request(REQUEST))

            def test_should_give_it_back_unchanged(
                self, publication_request: PublicationRequest
            ) -> None:
                assert publication_request == REQUEST

            def test_should_list_the_candidate_then_the_card(
                self, publication_request: PublicationRequest
            ) -> None:
                assert [published.name for published in publication_request.files] == [
                    "encoder-model.onnx",
                    "vocab.txt",
                    "README.md",
                ]

        @pytest.mark.parametrize(
            ("content", "reason"),
            list(REFUSED_REQUESTS.values()),
            ids=list(REFUSED_REQUESTS.keys()),
        )
        class TestWhenTheRequestIsMalformed:
            @pytest.fixture
            def parse_act(self, content: bytes) -> Callable[[], PublicationRequest]:
                return partial(parse_publication_request, content)

            def test_should_refuse_it_and_say_why(
                self, parse_act: Callable[[], PublicationRequest], reason: str
            ) -> None:
                with pytest.raises(PublicationRequestError, match=re.escape(reason)):
                    parse_act()

    class TestTheDescribeCommitMessageFunction:
        class TestWhenGivenTheTitleOfTheCard:
            @pytest.fixture
            def commit_message(self) -> str:
                return describe_commit_message("Parakeet Ultra in ONNX")

            def test_should_say_what_it_publishes(self, commit_message: str) -> None:
                assert commit_message == "Publish Parakeet Ultra in ONNX"
