import hashlib
import re
from collections.abc import Mapping
from dataclasses import dataclass, field

# The vocabulary of Parakeet TDT 0.6B v3, the one mostt pins: 8192 pieces and the blank.
VOCABULARY_SHA256 = "d58544679ea4bc6ac563d1f545eb7d474bd6cfa467f0a6e2c1dc1c7d37e3c35d"
UNKNOWN_PIECE = "<unk>"
BLANK_PIECE = "<blk>"
WORD_MARKER = "\u2581"
SPACE = " "
LINE_SEPARATOR = "\n"
FIELD_SEPARATOR = " "
TEXT_ENCODING = "utf-8"
MAX_PIECE_COUNT = 65_536
INDEX_GRAMMAR = re.compile(r"0|[1-9][0-9]{0,5}")


class VocabularyError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class Vocabulary:
    pieces: tuple[str, ...]
    ids_by_rendered_piece: Mapping[str, int] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "ids_by_rendered_piece",
            {render_piece(piece): token_id for token_id, piece in enumerate(self.pieces)},
        )

    @property
    def unknown_id(self) -> int:
        return 0

    @property
    def blank_id(self) -> int:
        return len(self.pieces) - 1


def render_piece(piece: str) -> str:
    return piece.replace(WORD_MARKER, SPACE)


def parse_vocabulary(content: bytes, expected_sha256: str = VOCABULARY_SHA256) -> Vocabulary:
    sha256 = hashlib.sha256(content).hexdigest()
    if sha256 != expected_sha256:
        raise VocabularyError(f"the vocabulary's sha256 is {sha256}, not {expected_sha256}")
    try:
        text = content.decode(TEXT_ENCODING)
    except UnicodeDecodeError as error:
        raise VocabularyError(f"the vocabulary is not UTF-8: {error}") from error
    if not text.endswith(LINE_SEPARATOR):
        raise VocabularyError("the vocabulary does not end with a line feed")
    lines = text[: -len(LINE_SEPARATOR)].split(LINE_SEPARATOR)
    if len(lines) > MAX_PIECE_COUNT:
        raise VocabularyError(
            f"the vocabulary holds {len(lines)} lines, more than {MAX_PIECE_COUNT}"
        )
    pieces = tuple(parse_piece(line, position) for position, line in enumerate(lines))
    judge_special_pieces(pieces)
    judge_uniqueness(pieces)
    return Vocabulary(pieces)


def parse_piece(line: str, position: int) -> str:
    where = f"line {position + 1} of the vocabulary"
    piece, separator, index = line.rpartition(FIELD_SEPARATOR)
    if not separator:
        raise VocabularyError(f"{where} is not '<piece> <index>': {line!a}")
    if not piece:
        raise VocabularyError(f"{where} holds an empty piece")
    # onnx-asr gives its pieces with a space in place of the word marker: a piece without a space
    # of its own maps back to a single id.
    if SPACE in piece:
        raise VocabularyError(f"{where}: the piece {piece!a} holds a space")
    if not INDEX_GRAMMAR.fullmatch(index):
        raise VocabularyError(f"{where}: the index {index!a} is not decimal")
    if int(index) != position:
        raise VocabularyError(f"{where} holds the index {index}, not {position}")
    return piece


def judge_special_pieces(pieces: tuple[str, ...]) -> None:
    if pieces[0] != UNKNOWN_PIECE:
        raise VocabularyError(f"the vocabulary starts with {pieces[0]!a}, not {UNKNOWN_PIECE}")
    if pieces[-1] != BLANK_PIECE:
        raise VocabularyError(f"the vocabulary ends with {pieces[-1]!a}, not {BLANK_PIECE}")


def judge_uniqueness(pieces: tuple[str, ...]) -> None:
    seen: set[str] = set()
    for piece in pieces:
        if piece in seen:
            raise VocabularyError(f"the vocabulary holds the piece {piece!a} twice")
        seen.add(piece)
