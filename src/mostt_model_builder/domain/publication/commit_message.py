from dataclasses import dataclass

from mostt_model_builder.domain.publication.publication_request import (
    COMMIT_MESSAGE_CHARACTERS,
    MAX_COMMIT_DESCRIPTION_CHARACTERS,
    MAX_COMMIT_TITLE_CHARACTERS,
    describe_commit_title,
)

MAX_COMMIT_MESSAGE_FILE_BYTES = 8 * 1024
QUOTING_ENCODING = "utf-8"
LINE_FEED = "\n"
TRAILING_SPACE = " "


class CommitMessageError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class CommitMessage:
    title: str
    description: str


def describe_default_commit_message(card_title: str) -> CommitMessage:
    return CommitMessage(describe_commit_title(card_title), "")


def parse_commit_message_file(content: bytes, what: str) -> CommitMessage:
    if len(content) > MAX_COMMIT_MESSAGE_FILE_BYTES:
        raise CommitMessageError(f"{what} holds more than {MAX_COMMIT_MESSAGE_FILE_BYTES} bytes")
    text = decode_for_quoting(content)
    refuse_characters_outside_the_message(text, what)
    lines = [line.rstrip(TRAILING_SPACE) for line in text.split(LINE_FEED)]
    while lines and not lines[-1]:
        lines.pop()
    title, rest = (lines[0], lines[1:]) if lines else ("", [])
    if not title:
        raise CommitMessageError(f"{what} has no title on its first line")
    if len(title) > MAX_COMMIT_TITLE_CHARACTERS:
        raise CommitMessageError(
            f"{what} has a title of {len(title)} characters, beyond {MAX_COMMIT_TITLE_CHARACTERS}"
        )
    if rest and rest[0]:
        raise CommitMessageError(f"{what} has no blank line between its title and its description")
    description = LINE_FEED.join(rest[1:])
    if len(description) > MAX_COMMIT_DESCRIPTION_CHARACTERS:
        raise CommitMessageError(
            f"{what} has a description of {len(description)} characters, "
            f"beyond {MAX_COMMIT_DESCRIPTION_CHARACTERS}"
        )
    return CommitMessage(title, description)


def decode_for_quoting(content: bytes) -> str:
    return content.decode(QUOTING_ENCODING, errors="replace")


def refuse_characters_outside_the_message(text: str, what: str) -> None:
    for index, character in enumerate(text):
        if character not in COMMIT_MESSAGE_CHARACTERS:
            line_number = text.count(LINE_FEED, 0, index) + 1
            raise CommitMessageError(
                f"{what} is not printable ASCII text: line {line_number} holds {character!a}"
            )
