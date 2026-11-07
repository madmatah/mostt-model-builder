import re
from collections.abc import Callable
from functools import partial

import pytest

from mostt_model_builder.domain.publication.commit_message import (
    CommitMessage,
    CommitMessageError,
    describe_default_commit_message,
    parse_commit_message_file,
)

MESSAGE_FILE = "the message file /data/work/publication-message.txt"
TITLE = "Keep the encoder's lengths in float32 in float16 and bfloat16; measure five languages"
DESCRIPTION = (
    "The float16 and bfloat16 encoders of revision 9dcab459 computed the encoder's output\n"
    "lengths (encoded_lengths), and the lengths of its subsampling stages, in their own\n"
    "reduced precision. The decoder then ignored the last frames of most recordings, and\n"
    "onnx-asr could fail on a long one. This revision keeps that computation in float32;\n"
    "the float32 and int8 files are unchanged.\n"
    "\n"
    "The card now measures English, Latin American Spanish, French, Brazilian Portuguese\n"
    "and German on FLEURS."
)
MESSAGE = CommitMessage(TITLE, DESCRIPTION)
FILE_TEXT = f"{TITLE}\n\n{DESCRIPTION}\n"
FILE_WITH_TRAILING_WHITESPACE = (
    f"{TITLE}  \n \n{DESCRIPTION.replace('unchanged.', 'unchanged.   ')}\n\n  \n\n"
)
FILE_WITH_A_TYPOGRAPHIC_APOSTROPHE = FILE_TEXT.replace(
    "the encoder's output", "the encoder\u2019s output"
)
TITLE_ON_ITS_LIMIT = "t" * 200
DESCRIPTION_ON_ITS_LIMIT = "d" * 4096
FILE_ON_ITS_LIMIT = FILE_TEXT + "\n" * (8192 - len(FILE_TEXT))
DESCRIPTION_OF_4213_CHARACTERS_ON_43_LINES = ("d" * 99 + "\n") * 42 + "d" * 13
LINE_FEED = 0x0A
PRINTABLE_ASCII = range(0x20, 0x7F)
SHORT_TITLE = b"Keep the encoder lengths in float32"

ACCEPTED_FILES = {
    "a title, a blank line and two paragraphs give the title and the description": (
        FILE_TEXT.encode(),
        MESSAGE,
    ),
    "trailing spaces and trailing blank lines are removed": (
        FILE_WITH_TRAILING_WHITESPACE.encode(),
        MESSAGE,
    ),
    "a title alone without a final line feed gives an empty description": (
        TITLE.encode(),
        CommitMessage(TITLE, ""),
    ),
    "a title alone with a final line feed gives an empty description": (
        f"{TITLE}\n".encode(),
        CommitMessage(TITLE, ""),
    ),
    "a title followed by a blank line alone gives an empty description": (
        f"{TITLE}\n\n".encode(),
        CommitMessage(TITLE, ""),
    ),
    "a title of 200 characters is accepted": (
        TITLE_ON_ITS_LIMIT.encode(),
        CommitMessage(TITLE_ON_ITS_LIMIT, ""),
    ),
    "a description of 4096 characters is accepted": (
        f"{TITLE}\n\n{DESCRIPTION_ON_ITS_LIMIT}\n".encode(),
        CommitMessage(TITLE, DESCRIPTION_ON_ITS_LIMIT),
    ),
    "a file of 8192 bytes is accepted": (FILE_ON_ITS_LIMIT.encode(), MESSAGE),
}

REFUSED_FILES = {
    "a file of 8193 bytes": (
        (FILE_ON_ITS_LIMIT + "\n").encode(),
        f"{MESSAGE_FILE} holds more than 8192 bytes",
    ),
    "a typographic apostrophe in UTF-8 on line 3": (
        FILE_WITH_A_TYPOGRAPHIC_APOSTROPHE.encode(),
        f"{MESSAGE_FILE} is not printable ASCII text: line 3 holds '\\u2019'",
    ),
    "a tab": (
        f"{TITLE}\n\nThe float16\tencoders\n".encode(),
        f"{MESSAGE_FILE} is not printable ASCII text: line 3 holds '\\t'",
    ),
    "a carriage return, as a CRLF file holds": (
        FILE_TEXT.replace("\n", "\r\n").encode(),
        f"{MESSAGE_FILE} is not printable ASCII text: line 1 holds '\\r'",
    ),
    "a byte that is not UTF-8": (
        f"{TITLE}\n\nThe float16 encoders\n".encode() + b"\xff\n",
        f"{MESSAGE_FILE} is not printable ASCII text: line 4 holds '\\ufffd'",
    ),
    "a title of 201 characters": (
        ("t" * 201).encode(),
        f"{MESSAGE_FILE} has a title of 201 characters, beyond 200",
    ),
    "a title of 213 characters above a description": (
        ("t" * 213 + f"\n\n{DESCRIPTION}\n").encode(),
        f"{MESSAGE_FILE} has a title of 213 characters, beyond 200",
    ),
    "a description of 4097 characters": (
        f"{TITLE}\n\n{'d' * 4097}\n".encode(),
        f"{MESSAGE_FILE} has a description of 4097 characters, beyond 4096",
    ),
    "a description of 4213 characters whose line feeds count": (
        f"{TITLE}\n\n{DESCRIPTION_OF_4213_CHARACTERS_ON_43_LINES}\n".encode(),
        f"{MESSAGE_FILE} has a description of 4213 characters, beyond 4096",
    ),
    "an empty file": (b"", f"{MESSAGE_FILE} has no title on its first line"),
    "an empty first line": (
        f"\n{TITLE}\n\n{DESCRIPTION}\n".encode(),
        f"{MESSAGE_FILE} has no title on its first line",
    ),
    "a first line of spaces": (
        f"   \n\n{DESCRIPTION}\n".encode(),
        f"{MESSAGE_FILE} has no title on its first line",
    ),
    "a description right under the title": (
        f"{TITLE}\n{DESCRIPTION}\n".encode(),
        f"{MESSAGE_FILE} has no blank line between its title and its description",
    ),
}


def is_accepted(content: bytes) -> bool:
    try:
        parse_commit_message_file(content, MESSAGE_FILE)
    except CommitMessageError:
        return False
    return True


class TestTheCommitMessage:
    class TestTheParseCommitMessageFileFunction:
        @pytest.mark.parametrize(
            ("content", "expected_message"),
            list(ACCEPTED_FILES.values()),
            ids=list(ACCEPTED_FILES.keys()),
        )
        class TestWhenTheFileIsWellFormed:
            @pytest.fixture
            def commit_message(self, content: bytes) -> CommitMessage:
                return parse_commit_message_file(content, MESSAGE_FILE)

            def test_should_give_its_title_and_its_description(
                self, commit_message: CommitMessage, expected_message: CommitMessage
            ) -> None:
                assert commit_message == expected_message

        @pytest.mark.parametrize(
            ("content", "reason"),
            list(REFUSED_FILES.values()),
            ids=list(REFUSED_FILES.keys()),
        )
        class TestWhenTheFileIsMalformed:
            @pytest.fixture
            def parse_act(self, content: bytes) -> Callable[[], CommitMessage]:
                return partial(parse_commit_message_file, content, MESSAGE_FILE)

            def test_should_refuse_it_after_its_label_and_say_why(
                self, parse_act: Callable[[], CommitMessage], reason: str
            ) -> None:
                with pytest.raises(CommitMessageError, match=f"^{re.escape(reason)}$"):
                    parse_act()

        class TestWhenEachByteValueFollowsATitle:
            @pytest.fixture
            def acceptance_by_byte_value(self) -> dict[int, bool]:
                return {
                    byte_value: is_accepted(SHORT_TITLE + bytes([byte_value]))
                    for byte_value in range(256)
                }

            def test_should_accept_exactly_printable_ascii_and_the_line_feed(
                self, acceptance_by_byte_value: dict[int, bool]
            ) -> None:
                expected = {*PRINTABLE_ASCII, LINE_FEED}
                mismatches = [
                    byte_value
                    for byte_value, is_byte_accepted in acceptance_by_byte_value.items()
                    if is_byte_accepted != (byte_value in expected)
                ]
                assert (mismatches, len(acceptance_by_byte_value)) == ([], 256)

    class TestTheDescribeDefaultCommitMessageFunction:
        class TestWhenGivenTheTitleOfTheCard:
            @pytest.fixture
            def commit_message(self) -> CommitMessage:
                return describe_default_commit_message("Parakeet Ultra in ONNX")

            def test_should_say_what_it_publishes_without_a_description(
                self, commit_message: CommitMessage
            ) -> None:
                assert commit_message == CommitMessage("Publish Parakeet Ultra in ONNX", "")
