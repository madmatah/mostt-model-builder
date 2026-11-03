from collections.abc import Sequence

from mostt_recipe_parakeet_tdt.domain.transcripts.vocabulary import (
    BLANK_PIECE,
    SPACE,
    Vocabulary,
    render_piece,
)


class TokenRenderingError(ValueError):
    pass


def render_text(token_ids: Sequence[int], vocabulary: Vocabulary) -> str:
    return "".join(
        render_piece(vocabulary.pieces[judge_token_id(token_id, vocabulary)])
        for token_id in token_ids
    ).strip(SPACE)


def render_text_without_unknown_tokens(token_ids: Sequence[int], vocabulary: Vocabulary) -> str:
    return render_text(
        [
            token_id
            for token_id in token_ids
            if judge_token_id(token_id, vocabulary) != vocabulary.unknown_id
        ],
        vocabulary,
    )


def count_unknown_tokens(token_ids: Sequence[int], vocabulary: Vocabulary) -> int:
    return sum(
        1 for token_id in token_ids if judge_token_id(token_id, vocabulary) == vocabulary.unknown_id
    )


def map_pieces_to_ids(pieces: Sequence[str], vocabulary: Vocabulary) -> tuple[int, ...]:
    return tuple(map_piece_to_id(piece, vocabulary) for piece in pieces)


def map_piece_to_id(piece: str, vocabulary: Vocabulary) -> int:
    token_id = vocabulary.ids_by_rendered_piece.get(piece)
    if token_id is None:
        raise TokenRenderingError(f"the piece {piece!a} is not in the vocabulary")
    if token_id == vocabulary.blank_id:
        raise TokenRenderingError(f"the piece {piece!a} is the blank, {BLANK_PIECE}")
    return token_id


def judge_token_id(token_id: int, vocabulary: Vocabulary) -> int:
    if isinstance(token_id, bool) or not isinstance(token_id, int):
        raise TokenRenderingError(f"the token id {token_id!a} is not an integer")
    if not 0 <= token_id < len(vocabulary.pieces):
        raise TokenRenderingError(
            f"the token id {token_id} is outside the vocabulary, 0 to {len(vocabulary.pieces) - 1}"
        )
    if token_id == vocabulary.blank_id:
        raise TokenRenderingError(f"the token id {token_id} is the blank, never emitted")
    return token_id
