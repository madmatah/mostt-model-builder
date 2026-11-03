import hashlib
import re
from collections.abc import Callable, Sequence
from functools import partial

import pytest

from mostt_recipe_parakeet_tdt.domain.transcripts.vocabulary import (
    VOCABULARY_SHA256,
    Vocabulary,
    VocabularyError,
    parse_vocabulary,
)

PIECES = ("<unk>", "\u2581the", "\u2581cat", "s", "\u2581sleep", "<blk>")
MAX_PIECE_COUNT = 65_536


def render_vocabulary(pieces: Sequence[str]) -> bytes:
    return "".join(f"{piece} {index}\n" for index, piece in enumerate(pieces)).encode()


def hash_content(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def render_lines(*lines: str) -> bytes:
    return "".join(f"{line}\n" for line in lines).encode()


VOCABULARY_CONTENT = render_vocabulary(PIECES)
VOCABULARY_CONTENT_SHA256 = hash_content(VOCABULARY_CONTENT)
OVERSIZED_PIECES = (
    "<unk>",
    *(f"p{index}" for index in range(1, MAX_PIECE_COUNT)),
    "<blk>",
)
REFUSED_VOCABULARIES = {
    "a byte that is not UTF-8": (
        b"<unk> 0\n\xff 1\n<blk> 2\n",
        "the vocabulary is not UTF-8",
    ),
    "no line feed at the end": (
        b"<unk> 0\n<blk> 1",
        "the vocabulary does not end with a line feed",
    ),
    "an empty vocabulary": (b"", "the vocabulary does not end with a line feed"),
    "a line without a space": (
        render_lines("<unk> 0", "cat", "<blk> 2"),
        "line 2 of the vocabulary is not '<piece> <index>': 'cat'",
    ),
    "an empty line": (
        render_lines("<unk> 0", "", "<blk> 2"),
        "line 2 of the vocabulary is not '<piece> <index>': ''",
    ),
    "an empty piece": (
        render_lines("<unk> 0", " 1", "<blk> 2"),
        "line 2 of the vocabulary holds an empty piece",
    ),
    "a piece holding a space": (
        render_lines("<unk> 0", "a cat 1", "<blk> 2"),
        "line 2 of the vocabulary: the piece 'a cat' holds a space",
    ),
    "two spaces before the index": (
        render_lines("<unk> 0", "cat  1", "<blk> 2"),
        "line 2 of the vocabulary: the piece 'cat ' holds a space",
    ),
    "an index in words": (
        render_lines("<unk> 0", "cat one", "<blk> 2"),
        "line 2 of the vocabulary: the index 'one' is not decimal",
    ),
    "an index with a sign": (
        render_lines("<unk> 0", "cat +1", "<blk> 2"),
        "line 2 of the vocabulary: the index '+1' is not decimal",
    ),
    "an index with a leading zero": (
        render_lines("<unk> 0", "cat 01", "<blk> 2"),
        "line 2 of the vocabulary: the index '01' is not decimal",
    ),
    "an index in Arabic-Indic digits": (
        render_lines("<unk> 0", "cat \u0661", "<blk> 2"),
        "line 2 of the vocabulary: the index '\\u0661' is not decimal",
    ),
    "a carriage return before the line feed": (
        render_lines("<unk> 0", "cat 1\r", "<blk> 2"),
        "line 2 of the vocabulary: the index '1\\r' is not decimal",
    ),
    "an index out of its position": (
        render_lines("<unk> 0", "cat 2", "<blk> 2"),
        "line 2 of the vocabulary holds the index 2, not 1",
    ),
    "a first piece other than the unknown piece": (
        render_vocabulary(("cat", "<unk>", "<blk>")),
        "the vocabulary starts with 'cat', not <unk>",
    ),
    "a last piece other than the blank": (
        render_vocabulary(("<unk>", "<blk>", "cat")),
        "the vocabulary ends with 'cat', not <blk>",
    ),
    "a piece listed twice": (
        render_vocabulary(("<unk>", "\u2581cat", "\u2581cat", "<blk>")),
        "the vocabulary holds the piece '\\u2581cat' twice",
    ),
    "more lines than the bound": (
        render_vocabulary(OVERSIZED_PIECES),
        f"the vocabulary holds {MAX_PIECE_COUNT + 1} lines, more than {MAX_PIECE_COUNT}",
    ),
}


class TestTheVocabulary:
    class TestTheParseVocabularyFunction:
        class TestWhenAWellFormedVocabularyOfTheExpectedHashIsRead:
            @pytest.fixture
            def vocabulary(self) -> Vocabulary:
                return parse_vocabulary(VOCABULARY_CONTENT, VOCABULARY_CONTENT_SHA256)

            def test_should_read_every_piece_in_the_order_of_its_index(
                self, vocabulary: Vocabulary
            ) -> None:
                assert vocabulary.pieces == PIECES

            def test_should_give_the_last_id_to_the_blank(self, vocabulary: Vocabulary) -> None:
                assert vocabulary.blank_id == len(PIECES) - 1

            def test_should_give_the_first_id_to_the_unknown_piece(
                self, vocabulary: Vocabulary
            ) -> None:
                assert vocabulary.unknown_id == 0

            def test_should_map_each_piece_with_spaces_for_word_markers_to_its_id(
                self, vocabulary: Vocabulary
            ) -> None:
                assert dict(vocabulary.ids_by_rendered_piece) == {
                    "<unk>": 0,
                    " the": 1,
                    " cat": 2,
                    "s": 3,
                    " sleep": 4,
                    "<blk>": 5,
                }

        class TestWhenTheHashIsNotTheExpectedOne:
            @pytest.fixture
            def parse_act(self) -> Callable[[], Vocabulary]:
                return partial(parse_vocabulary, VOCABULARY_CONTENT, hash_content(b"another"))

            def test_should_refuse_it_before_reading_it(
                self, parse_act: Callable[[], Vocabulary]
            ) -> None:
                with pytest.raises(
                    VocabularyError,
                    match=re.escape(
                        f"the vocabulary's sha256 is {VOCABULARY_CONTENT_SHA256}, not "
                        f"{hash_content(b'another')}"
                    ),
                ):
                    parse_act()

        class TestWhenNoHashIsGiven:
            @pytest.fixture
            def parse_act(self) -> Callable[[], Vocabulary]:
                return partial(parse_vocabulary, VOCABULARY_CONTENT)

            def test_should_refuse_any_vocabulary_but_the_pinned_one(
                self, parse_act: Callable[[], Vocabulary]
            ) -> None:
                with pytest.raises(
                    VocabularyError,
                    match=re.escape(
                        f"the vocabulary's sha256 is {VOCABULARY_CONTENT_SHA256}, not "
                        f"{VOCABULARY_SHA256}"
                    ),
                ):
                    parse_act()

        @pytest.mark.parametrize(
            ("content", "reason"),
            list(REFUSED_VOCABULARIES.values()),
            ids=list(REFUSED_VOCABULARIES.keys()),
        )
        class TestWhenTheVocabularyIsMalformed:
            @pytest.fixture
            def parse_act(self, content: bytes) -> Callable[[], Vocabulary]:
                return partial(parse_vocabulary, content, hash_content(content))

            def test_should_refuse_it_and_say_why(
                self, parse_act: Callable[[], Vocabulary], reason: str
            ) -> None:
                with pytest.raises(VocabularyError, match=re.escape(reason)):
                    parse_act()
