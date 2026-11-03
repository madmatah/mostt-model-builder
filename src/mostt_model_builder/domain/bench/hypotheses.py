import json
import math
import unicodedata
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass

from mostt_model_builder.domain.bench.fleurs_index import RECORDING_NAME_GRAMMAR
from mostt_model_builder.domain.json_documents.json_depth import MAX_JSON_DEPTH, refuse_deep_json

RUN_LINES_FILE_NAME = "lines.jsonl"
# A run writes one line per utterance of FLEURS's test sets, under 700 per language, each a few
# hundred bytes; the bounds only refuse lines that no run of the bench could write.
RUN_LINES_MAX_SIZE_BYTES = 16 * 2**20
MAX_LINE_COUNT = 20_000
MAX_LINE_SIZE_BYTES = 64 * 1024
MAX_TEXT_LENGTH = 4096
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
    text: str
    unknown_token_count: int
    decoding_time_ms: float


type RunLine = Mapping[str, object]
# The reader of a run is the recipe's, so that the core never sees a token of the model.
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
    if len(hypothesis.text) > MAX_TEXT_LENGTH:
        raise HypothesisError(f"{where} holds more than {MAX_TEXT_LENGTH} characters")
    if has_invisible_character(hypothesis.text):
        raise HypothesisError(f"{where} holds a character a reader cannot see")
    if (
        isinstance(hypothesis.unknown_token_count, bool)
        or not isinstance(hypothesis.unknown_token_count, int)
        or hypothesis.unknown_token_count < 0
    ):
        raise HypothesisError(
            f"{where} counts {hypothesis.unknown_token_count!a} unknown tokens, not a count"
        )
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


def has_invisible_character(text: str) -> bool:
    return any(unicodedata.category(character) in INVISIBLE_CATEGORIES for character in text)


def refuse_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    keys = [key for key, _ in pairs]
    if len(set(keys)) != len(keys):
        raise ValueError("a key appears twice in an object")
    return dict(pairs)
