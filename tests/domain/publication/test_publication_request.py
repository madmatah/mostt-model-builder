import json
import re
from collections.abc import Callable
from dataclasses import replace
from functools import partial

import pytest

from mostt_model_builder.domain.publication.publication_request import (
    COMMIT_MESSAGE_CHARACTERS,
    PublicationRequest,
    PublicationRequestError,
    PublishedFile,
    describe_commit_title,
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
COMMIT_DESCRIPTION = (
    "The float16 and bfloat16 encoders of revision 9dcab459 computed the encoder's output\n"
    "lengths (encoded_lengths), and the lengths of its subsampling stages, in their own\n"
    "reduced precision. The decoder then ignored the last frames of most recordings, and\n"
    "onnx-asr could fail on a long one. This revision keeps that computation in float32;\n"
    "the float32 and int8 files are unchanged.\n"
    "\n"
    "The card now measures English, Latin American Spanish, French, Brazilian Portuguese\n"
    "and German on FLEURS."
)
LONGEST_COMMIT_DESCRIPTION = "d" * 4096
PRINTABLE_ASCII = range(0x20, 0x7F)
LINE_FEED = 0x0A
CODE_POINTS_OUTSIDE_ASCII = (0x80, 0xA0, 0x2019)


def change_document(**fields: object) -> bytes:
    return json.dumps({**DOCUMENT, **fields}).encode()


def find_refusal_message(content: bytes) -> str:
    try:
        parse_publication_request(content)
    except PublicationRequestError as error:
        return str(error)
    raise LookupError("the request was accepted")


REFUSED_COMMIT_DESCRIPTIONS = {
    "a description with a tab": "The float16\tencoders kept their lengths",
    "a description with a character outside ASCII": "The encoder\u2019s lengths stay in float32",
}

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
    "no commit description": (
        json.dumps(
            {key: value for key, value in DOCUMENT.items() if key != "commit_description"}
        ).encode(),
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
    "a commit description that is not a string": (
        change_document(commit_description=["The float16 encoders"]),
        "commit description is not a text of at most 4096 printable ASCII characters",
    ),
    "a commit description beyond its bound": (
        change_document(commit_description="d" * 4097),
        "commit description is not a text of at most 4096",
    ),
    "a commit description with a tab": (
        change_document(commit_description="The float16\tencoders"),
        "commit description is not a text of at most 4096",
    ),
    "a commit description whose ASCII JSON escapes a character outside ASCII": (
        change_document(commit_description="The encoder\u2019s lengths"),
        "commit description is not a text of at most 4096",
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

            def test_should_carry_an_empty_commit_description(
                self, publication_request: PublicationRequest
            ) -> None:
                assert publication_request.commit_description == ""

            def test_should_list_the_candidate_then_the_card(
                self, publication_request: PublicationRequest
            ) -> None:
                assert [published.name for published in publication_request.files] == [
                    "encoder-model.onnx",
                    "vocab.txt",
                    "README.md",
                ]

        @pytest.mark.parametrize(
            "commit_description",
            [COMMIT_DESCRIPTION, LONGEST_COMMIT_DESCRIPTION],
            ids=["a description of two paragraphs", "a description of 4096 characters"],
        )
        class TestWhenGivenARenderedRequestWithACommitDescription:
            @pytest.fixture
            def publication_request(self, commit_description: str) -> PublicationRequest:
                return parse_publication_request(
                    render_publication_request(
                        replace(REQUEST, commit_description=commit_description)
                    )
                )

            def test_should_give_it_back_unchanged(
                self, publication_request: PublicationRequest, commit_description: str
            ) -> None:
                assert publication_request == replace(
                    REQUEST, commit_description=commit_description
                )

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

        @pytest.mark.parametrize(
            "commit_description",
            list(REFUSED_COMMIT_DESCRIPTIONS.values()),
            ids=list(REFUSED_COMMIT_DESCRIPTIONS.keys()),
        )
        class TestWhenTheCommitDescriptionIsRefused:
            @pytest.fixture
            def refusal_message(self, commit_description: str) -> str:
                return find_refusal_message(change_document(commit_description=commit_description))

            def test_should_not_quote_the_description_which_can_be_thousands_of_characters_long(
                self, refusal_message: str, commit_description: str
            ) -> None:
                assert all(
                    quoted not in refusal_message
                    for quoted in (commit_description, ascii(commit_description))
                )

    class TestTheDescribeCommitTitleFunction:
        class TestWhenGivenTheTitleOfTheCard:
            @pytest.fixture
            def commit_title(self) -> str:
                return describe_commit_title("Parakeet Ultra in ONNX")

            def test_should_say_what_it_publishes(self, commit_title: str) -> None:
                assert commit_title == "Publish Parakeet Ultra in ONNX"

    class TestTheCommitMessageCharactersConstant:
        class TestWhenEveryAsciiCodePointAndSomeBeyondAreChecked:
            def test_should_hold_exactly_printable_ascii_and_the_line_feed(self) -> None:
                code_points = [*range(0x80), *CODE_POINTS_OUTSIDE_ASCII]
                expected = {*PRINTABLE_ASCII, LINE_FEED}
                mismatches = [
                    code_point
                    for code_point in code_points
                    if (chr(code_point) in COMMIT_MESSAGE_CHARACTERS) != (code_point in expected)
                ]
                assert (mismatches, len(code_points)) == ([], 131)
