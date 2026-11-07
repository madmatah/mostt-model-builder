import json
import math
import unicodedata
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass

from mostt_model_builder.domain.bench.fleurs_index import RECORDING_NAME_GRAMMAR
from mostt_model_builder.domain.bench.run_lines import RUN_LINES_MAX_SIZE_BYTES
from mostt_model_builder.domain.json_documents.json_depth import MAX_JSON_DEPTH, refuse_deep_json

# A run writes one line per utterance of FLEURS's test sets, under 1 000 per language, each a few
# hundred bytes; the bounds only refuse lines that no run of the bench could write.
MAX_LINE_COUNT = 20_000
MAX_LINE_SIZE_BYTES = 64 * 1024
MAX_TEXT_LENGTH = 4096
MAX_TOKEN_COUNT = 4096
# Token ids are compared, never read: the bound only refuses an id that no vocabulary holds.
MAX_TOKEN_ID = 2**31 - 1
MAX_DECODING_TIME_MS = 3_600_000.0
LINE_SEPARATOR = b"\n"
TEXT_ENCODING = "utf-8"
# Every character a reader cannot see, and the line and paragraph separators, are refused in a
# hypothesis's text, which reaches a terminal and a report.
INVISIBLE_CATEGORIES = frozenset({"Cc", "Cf", "Co", "Cn", "Cs", "Zl", "Zp"})


class HypothesisError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class Hypothesis:
    language: str
    utterance_id: str
    token_ids: tuple[int, ...]
    text: str
    text_without_unknown_tokens: str
    unknown_token_count: int
    decoding_time_ms: float


type RunLine = Mapping[str, object]
# The reader of a run is the recipe's, so that the core never reads a token of the model: it
# compares the ids of two runs, and measures the texts.
type RunReader = Callable[[bytes, Mapping[str, bytes]], tuple[Hypothesis, ...]]


def refuse_constant(name: str) -> object:
    raise HypothesisError(f"a line holds {name}, which is not a JSON number")


def parse_run_lines(content: bytes) -> tuple[RunLine, ...]:
    if len(content) > RUN_LINES_MAX_SIZE_BYTES:
        raise HypothesisError(
            f"the lines hold {len(content)} bytes, beyond {RUN_LINES_MAX_SIZE_BYTES}"
        )
    if not content.endswith(LINE_SEPARATOR):
        raise HypothesisError("the lines do not end with a line feed")
    raw_lines = content[: -len(LINE_SEPARATOR)].split(LINE_SEPARATOR)
    if len(raw_lines) > MAX_LINE_COUNT:
        raise HypothesisError(f"the run wrote {len(raw_lines)} lines, more than {MAX_LINE_COUNT}")
    return tuple(
        parse_run_line(raw_line, line_number)
        for line_number, raw_line in enumerate(raw_lines, start=1)
    )


def parse_run_line(raw_line: bytes, line_number: int) -> RunLine:
    if len(raw_line) > MAX_LINE_SIZE_BYTES:
        raise HypothesisError(f"line {line_number} holds more than {MAX_LINE_SIZE_BYTES} bytes")
    try:
        line = json.loads(
            refuse_deep_json(raw_line.decode(TEXT_ENCODING), MAX_JSON_DEPTH),
            parse_constant=refuse_constant,
            object_pairs_hook=refuse_duplicate_keys,
        )
    except (UnicodeDecodeError, ValueError) as error:
        raise HypothesisError(f"line {line_number} is not a JSON object: {error}") from error
    if not isinstance(line, dict):
        raise HypothesisError(f"line {line_number} is not a JSON object")
    return line


def judge_hypotheses(
    hypotheses: Sequence[Hypothesis], languages: Collection[str]
) -> tuple[Hypothesis, ...]:
    seen: set[tuple[str, str]] = set()
    for hypothesis in hypotheses:
        judge_hypothesis(hypothesis, languages)
        key = (hypothesis.language, hypothesis.utterance_id)
        if key in seen:
            raise HypothesisError(
                f"the utterance {hypothesis.utterance_id} of {hypothesis.language} has two "
                "hypotheses"
            )
        seen.add(key)
    return tuple(hypotheses)


def judge_hypothesis(hypothesis: Hypothesis, languages: Collection[str]) -> None:
    where = f"the hypothesis of {hypothesis.utterance_id!a}"
    if hypothesis.language not in languages:
        raise HypothesisError(f"{where} is in {hypothesis.language!a}, not a language of its run")
    if not RECORDING_NAME_GRAMMAR.fullmatch(hypothesis.utterance_id):
        raise HypothesisError(
            f"{where}: its utterance id does not match {RECORDING_NAME_GRAMMAR.pattern}"
        )
    judge_token_ids(hypothesis.token_ids, where)
    judge_text(hypothesis.text, where)
    judge_text(hypothesis.text_without_unknown_tokens, f"{where}, without its unknown tokens,")
    if (
        isinstance(hypothesis.unknown_token_count, bool)
        or not isinstance(hypothesis.unknown_token_count, int)
        or hypothesis.unknown_token_count < 0
    ):
        raise HypothesisError(
            f"{where} counts {hypothesis.unknown_token_count!a} unknown tokens, not a count"
        )
    if hypothesis.unknown_token_count > len(hypothesis.token_ids):
        raise HypothesisError(
            f"{where} counts {hypothesis.unknown_token_count} unknown tokens among "
            f"{len(hypothesis.token_ids)} tokens"
        )
    if hypothesis.unknown_token_count == 0 and (
        hypothesis.text != hypothesis.text_without_unknown_tokens
    ):
        raise HypothesisError(f"{where} counts no unknown token, and its two texts differ")
    decoding_time_ms = hypothesis.decoding_time_ms
    if (
        isinstance(decoding_time_ms, bool)
        or not isinstance(decoding_time_ms, int | float)
        or not math.isfinite(decoding_time_ms)
        or not 0 <= decoding_time_ms <= MAX_DECODING_TIME_MS
    ):
        raise HypothesisError(
            f"{where} took {decoding_time_ms!a} ms, outside 0 to {MAX_DECODING_TIME_MS:.0f}"
        )


def judge_token_ids(token_ids: object, where: str) -> None:
    if not isinstance(token_ids, tuple):
        raise HypothesisError(f"{where}: its token ids are not a tuple")
    if len(token_ids) > MAX_TOKEN_COUNT:
        raise HypothesisError(f"{where} holds more than {MAX_TOKEN_COUNT} tokens")
    for token_id in token_ids:
        if (
            isinstance(token_id, bool)
            or not isinstance(token_id, int)
            or not 0 <= token_id <= MAX_TOKEN_ID
        ):
            raise HypothesisError(
                f"{where} holds the token id {token_id!a}, not a whole number from 0 to "
                f"{MAX_TOKEN_ID}"
            )


def judge_text(text: str, where: str) -> None:
    if len(text) > MAX_TEXT_LENGTH:
        raise HypothesisError(f"{where} holds more than {MAX_TEXT_LENGTH} characters")
    if has_invisible_character(text):
        raise HypothesisError(f"{where} holds a character a reader cannot see")


def has_invisible_character(text: str) -> bool:
    return any(unicodedata.category(character) in INVISIBLE_CATEGORIES for character in text)


def refuse_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    keys = [key for key, _ in pairs]
    if len(set(keys)) != len(keys):
        raise ValueError("a key appears twice in an object")
    return dict(pairs)
