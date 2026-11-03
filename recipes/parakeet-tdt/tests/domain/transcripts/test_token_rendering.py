import re
from collections.abc import Callable, Sequence
from functools import partial
from typing import cast

import pytest

from mostt_recipe_parakeet_tdt.domain.transcripts.token_rendering import (
    TokenRenderingError,
    count_unknown_tokens,
    map_pieces_to_ids,
    render_text,
)
from mostt_recipe_parakeet_tdt.domain.transcripts.vocabulary import Vocabulary

VOCABULARY = Vocabulary(("<unk>", "\u2581the", "\u2581cat", "s", "\u2581sleep", "<blk>"))
UNKNOWN_ID = 0
THE_ID = 1
CAT_ID = 2
PLURAL_ID = 3
SLEEP_ID = 4
BLANK_ID = 5
SENTENCE_IDS = (THE_ID, CAT_ID, PLURAL_ID, SLEEP_ID)
SENTENCE_WITH_UNKNOWN_IDS = (THE_ID, UNKNOWN_ID, CAT_ID, UNKNOWN_ID)
SENTENCE_PIECES = (" the", " cat", "s", " sleep", "<unk>")
REFUSED_TOKEN_IDS: dict[str, tuple[Sequence[object], str]] = {
    "a negative id": ((THE_ID, -1), "the token id -1 is outside the vocabulary, 0 to 5"),
    "an id beyond the vocabulary": (
        (THE_ID, 6),
        "the token id 6 is outside the vocabulary, 0 to 5",
    ),
    "the blank's id": ((THE_ID, BLANK_ID), "the token id 5 is the blank, never emitted"),
    "a boolean": ((THE_ID, True), "the token id True is not an integer"),
    "a float": ((THE_ID, 1.0), "the token id 1.0 is not an integer"),
    "a string": ((THE_ID, "1"), "the token id '1' is not an integer"),
}
REFUSED_PIECES = {
    "a piece outside the vocabulary": (
        (" the", " dog"),
        "the piece ' dog' is not in the vocabulary",
    ),
    "a piece that still holds its word marker": (
        ("\u2581the",),
        "the piece '\\u2581the' is not in the vocabulary",
    ),
    "the blank's piece": ((" the", "<blk>"), "the piece '<blk>' is the blank, <blk>"),
}


class TestTheTokenRendering:
    class TestTheRenderTextFunction:
        class TestWhenTheIdsStartWordsWithTheirMarker:
            @pytest.fixture
            def text(self) -> str:
                return render_text(SENTENCE_IDS, VOCABULARY)

            def test_should_join_the_pieces_with_a_space_for_each_marker_and_strip_the_ends(
                self, text: str
            ) -> None:
                assert text == "the cats sleep"

        class TestWhenTheIdsHoldTheUnknownId:
            @pytest.fixture
            def text(self) -> str:
                return render_text(SENTENCE_WITH_UNKNOWN_IDS, VOCABULARY)

            def test_should_render_each_unknown_token_as_its_piece(self, text: str) -> None:
                assert text == "the<unk> cat<unk>"

        class TestWhenThereIsNoId:
            @pytest.fixture
            def text(self) -> str:
                return render_text((), VOCABULARY)

            def test_should_render_an_empty_text(self, text: str) -> None:
                assert text == ""

        @pytest.mark.parametrize(
            ("token_ids", "reason"),
            list(REFUSED_TOKEN_IDS.values()),
            ids=list(REFUSED_TOKEN_IDS.keys()),
        )
        class TestWhenAnIdIsNoTokenTheModelEmits:
            @pytest.fixture
            def render_act(self, token_ids: Sequence[object]) -> Callable[[], str]:
                return partial(render_text, cast("Sequence[int]", token_ids), VOCABULARY)

            def test_should_refuse_it_and_say_why(
                self, render_act: Callable[[], str], reason: str
            ) -> None:
                with pytest.raises(TokenRenderingError, match=re.escape(reason)):
                    render_act()

    class TestTheCountUnknownTokensFunction:
        class TestWhenTheIdsHoldTheUnknownIdTwice:
            @pytest.fixture
            def unknown_token_count(self) -> int:
                return count_unknown_tokens(SENTENCE_WITH_UNKNOWN_IDS, VOCABULARY)

            def test_should_count_two_unknown_tokens(self, unknown_token_count: int) -> None:
                assert unknown_token_count == 2

        class TestWhenAnIdIsBeyondTheVocabulary:
            @pytest.fixture
            def count_act(self) -> Callable[[], int]:
                return partial(count_unknown_tokens, (THE_ID, 6), VOCABULARY)

            def test_should_refuse_it(self, count_act: Callable[[], int]) -> None:
                with pytest.raises(
                    TokenRenderingError,
                    match=re.escape("the token id 6 is outside the vocabulary"),
                ):
                    count_act()

    class TestTheMapPiecesToIdsFunction:
        class TestWhenEachPieceIsInTheVocabulary:
            @pytest.fixture
            def token_ids(self) -> tuple[int, ...]:
                return map_pieces_to_ids(SENTENCE_PIECES, VOCABULARY)

            def test_should_map_each_piece_with_spaces_for_markers_back_to_its_id(
                self, token_ids: tuple[int, ...]
            ) -> None:
                assert token_ids == (THE_ID, CAT_ID, PLURAL_ID, SLEEP_ID, UNKNOWN_ID)

        @pytest.mark.parametrize(
            ("pieces", "reason"),
            list(REFUSED_PIECES.values()),
            ids=list(REFUSED_PIECES.keys()),
        )
        class TestWhenAPieceIsNoTokenTheModelEmits:
            @pytest.fixture
            def map_act(self, pieces: Sequence[str]) -> Callable[[], tuple[int, ...]]:
                return partial(map_pieces_to_ids, pieces, VOCABULARY)

            def test_should_refuse_it_and_say_why(
                self, map_act: Callable[[], tuple[int, ...]], reason: str
            ) -> None:
                with pytest.raises(TokenRenderingError, match=re.escape(reason)):
                    map_act()
