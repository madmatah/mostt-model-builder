from collections.abc import Callable, Mapping
from functools import partial

from mostt_model_builder.domain.bench.fleurs_index import BENCH_LANGUAGES
from mostt_model_builder.domain.bench.hypotheses import (
    Hypothesis,
    RunLine,
    judge_hypotheses,
    parse_run_lines,
)
from mostt_recipe_parakeet_tdt.domain.transcripts.token_rendering import (
    TokenRenderingError,
    count_unknown_tokens,
    map_pieces_to_ids,
    render_text,
    render_text_without_unknown_tokens,
)
from mostt_recipe_parakeet_tdt.domain.transcripts.vocabulary import (
    VOCABULARY_SHA256,
    Vocabulary,
    VocabularyError,
    parse_vocabulary,
)
from mostt_recipe_parakeet_tdt.domain.variants.candidate_layout import VOCABULARY_FILE_NAME

# An utterance of FLEURS lasts under 30 s, a few hundred tokens at most.
MAX_TOKEN_COUNT = 4096
LANGUAGE_KEY = "language"
UTTERANCE_ID_KEY = "utterance_id"
TOKEN_IDS_KEY = "token_ids"
PIECES_KEY = "pieces"
DECODING_TIME_KEY = "decoding_time_ms"
TOKEN_ID_LINE_KEYS = frozenset({LANGUAGE_KEY, UTTERANCE_ID_KEY, TOKEN_IDS_KEY, DECODING_TIME_KEY})
ONNX_ASR_LINE_KEYS = frozenset({LANGUAGE_KEY, UTTERANCE_ID_KEY, PIECES_KEY, DECODING_TIME_KEY})

type TokenIdReader = Callable[[RunLine, str], tuple[int, ...]]


class RunLineError(ValueError):
    pass


def read_token_id_lines(
    lines: bytes, candidate_texts: Mapping[str, bytes]
) -> tuple[Hypothesis, ...]:
    return read_token_id_lines_with(lines, candidate_texts, VOCABULARY_SHA256)


def read_onnx_asr_lines(
    lines: bytes, candidate_texts: Mapping[str, bytes]
) -> tuple[Hypothesis, ...]:
    return read_onnx_asr_lines_with(lines, candidate_texts, VOCABULARY_SHA256)


def read_token_id_lines_with(
    lines: bytes, candidate_texts: Mapping[str, bytes], expected_sha256: str
) -> tuple[Hypothesis, ...]:
    vocabulary = read_vocabulary(candidate_texts, expected_sha256)
    return read_run_lines(lines, vocabulary, TOKEN_ID_LINE_KEYS, read_listed_token_ids)


def read_onnx_asr_lines_with(
    lines: bytes, candidate_texts: Mapping[str, bytes], expected_sha256: str
) -> tuple[Hypothesis, ...]:
    vocabulary = read_vocabulary(candidate_texts, expected_sha256)
    return read_run_lines(
        lines,
        vocabulary,
        ONNX_ASR_LINE_KEYS,
        partial(read_onnx_asr_token_ids, vocabulary=vocabulary),
    )


def read_run_lines(
    lines: bytes,
    vocabulary: Vocabulary,
    line_keys: frozenset[str],
    read_token_ids: TokenIdReader,
) -> tuple[Hypothesis, ...]:
    hypotheses = tuple(
        read_hypothesis(line, f"line {line_number}", line_keys, read_token_ids, vocabulary)
        for line_number, line in enumerate(parse_run_lines(lines), start=1)
    )
    return judge_hypotheses(hypotheses, BENCH_LANGUAGES)


def read_vocabulary(candidate_texts: Mapping[str, bytes], expected_sha256: str) -> Vocabulary:
    content = candidate_texts.get(VOCABULARY_FILE_NAME)
    if content is None:
        raise VocabularyError(f"the candidate holds no {VOCABULARY_FILE_NAME}")
    return parse_vocabulary(content, expected_sha256)


def read_hypothesis(
    line: RunLine,
    where: str,
    line_keys: frozenset[str],
    read_token_ids: TokenIdReader,
    vocabulary: Vocabulary,
) -> Hypothesis:
    judge_keys(line, where, line_keys)
    language = read_string(line, LANGUAGE_KEY, where)
    utterance_id = read_string(line, UTTERANCE_ID_KEY, where)
    decoding_time_ms = read_number(line, DECODING_TIME_KEY, where)
    try:
        token_ids = read_token_ids(line, where)
        text = render_text(token_ids, vocabulary)
        text_without_unknown_tokens = render_text_without_unknown_tokens(token_ids, vocabulary)
        unknown_token_count = count_unknown_tokens(token_ids, vocabulary)
    except TokenRenderingError as error:
        raise RunLineError(f"{where}: {error}") from error
    return Hypothesis(
        language=language,
        utterance_id=utterance_id,
        token_ids=token_ids,
        text=text,
        text_without_unknown_tokens=text_without_unknown_tokens,
        unknown_token_count=unknown_token_count,
        decoding_time_ms=decoding_time_ms,
    )


def read_listed_token_ids(line: RunLine, where: str) -> tuple[int, ...]:
    entries = read_bounded_list(line, TOKEN_IDS_KEY, where)
    token_ids: list[int] = []
    for entry in entries:
        if isinstance(entry, bool) or not isinstance(entry, int):
            raise RunLineError(f"{where}: {TOKEN_IDS_KEY} holds {entry!a}, not an integer")
        token_ids.append(entry)
    return tuple(token_ids)


def read_onnx_asr_token_ids(line: RunLine, where: str, vocabulary: Vocabulary) -> tuple[int, ...]:
    entries = read_bounded_list(line, PIECES_KEY, where)
    pieces: list[str] = []
    for entry in entries:
        if not isinstance(entry, str):
            raise RunLineError(f"{where}: {PIECES_KEY} holds {entry!a}, not a string")
        pieces.append(entry)
    return map_pieces_to_ids(pieces, vocabulary)


def judge_keys(line: RunLine, where: str, line_keys: frozenset[str]) -> None:
    missing_keys = sorted(line_keys - line.keys())
    if missing_keys:
        raise RunLineError(f"{where} lacks {', '.join(missing_keys)}")
    extra_keys = sorted(line.keys() - line_keys)
    if extra_keys:
        raise RunLineError(
            f"{where} holds the unexpected {', '.join(ascii(key) for key in extra_keys)}"
        )


def read_string(line: RunLine, key: str, where: str) -> str:
    entry = line[key]
    if not isinstance(entry, str):
        raise RunLineError(f"{where}: {key} is {entry!a}, not a string")
    return entry


def read_number(line: RunLine, key: str, where: str) -> int | float:
    entry = line[key]
    if isinstance(entry, bool) or not isinstance(entry, int | float):
        raise RunLineError(f"{where}: {key} is {entry!a}, not a number")
    return entry


def read_bounded_list(line: RunLine, key: str, where: str) -> list[object]:
    entries = line[key]
    if not isinstance(entries, list):
        raise RunLineError(f"{where}: {key} is {entries!a}, not a list")
    if len(entries) > MAX_TOKEN_COUNT:
        raise RunLineError(
            f"{where}: {key} holds {len(entries)} tokens, more than {MAX_TOKEN_COUNT}"
        )
    return entries
