import hashlib
import json
import re
from collections.abc import Callable, Mapping, Sequence
from functools import partial

import pytest

from mostt_model_builder.domain.bench.hypotheses import Hypothesis, HypothesisError, RunReader
from mostt_recipe_parakeet_tdt.domain.transcripts.run_readers import (
    RunLineError,
    read_nemo_lines,
    read_nemo_lines_with,
    read_onnx_asr_lines,
    read_onnx_asr_lines_with,
)
from mostt_recipe_parakeet_tdt.domain.transcripts.vocabulary import (
    VOCABULARY_SHA256,
    VocabularyError,
)

PIECES = ("<unk>", "\u2581the", "\u2581cat", "s", "\u2581sleep", "<blk>")
VOCABULARY_CONTENT = "".join(f"{piece} {index}\n" for index, piece in enumerate(PIECES)).encode()
VOCABULARY_CONTENT_SHA256 = hashlib.sha256(VOCABULARY_CONTENT).hexdigest()
CANDIDATE_TEXTS = {"vocab.txt": VOCABULARY_CONTENT, "config.json": b"{}"}
MAX_TOKEN_COUNT = 4096
FRENCH_DECODING_TIME_MS = 12.5
ENGLISH_DECODING_TIME_MS = 3
EXPECTED_HYPOTHESES = (
    Hypothesis("fr_fr", "1.wav", "the cats sleep", 0, FRENCH_DECODING_TIME_MS),
    Hypothesis("en_us", "2.wav", "the<unk> cat", 1, ENGLISH_DECODING_TIME_MS),
)


def render_lines(*records: Mapping[str, object]) -> bytes:
    return b"".join(json.dumps(record).encode() + b"\n" for record in records)


def build_nemo_record(
    token_ids: Sequence[object] = (1, 2), **overrides: object
) -> dict[str, object]:
    return {
        "language": "fr_fr",
        "utterance_id": "1.wav",
        "token_ids": list(token_ids),
        "decoding_time_ms": 1.5,
        **overrides,
    }


def build_onnx_asr_record(
    pieces: Sequence[object] = (" the",), **overrides: object
) -> dict[str, object]:
    return {
        "language": "fr_fr",
        "utterance_id": "1.wav",
        "pieces": list(pieces),
        "decoding_time_ms": 1.5,
        **overrides,
    }


def drop_key(record: Mapping[str, object], key: str) -> dict[str, object]:
    return {name: entry for name, entry in record.items() if name != key}


NEMO_LINES = render_lines(
    build_nemo_record((1, 2, 3, 4), decoding_time_ms=FRENCH_DECODING_TIME_MS),
    build_nemo_record(
        (1, 0, 2),
        language="en_us",
        utterance_id="2.wav",
        decoding_time_ms=ENGLISH_DECODING_TIME_MS,
    ),
)
ONNX_ASR_LINES = render_lines(
    build_onnx_asr_record(
        (" the", " cat", "s", " sleep"), decoding_time_ms=FRENCH_DECODING_TIME_MS
    ),
    build_onnx_asr_record(
        (" the", "<unk>", " cat"),
        language="en_us",
        utterance_id="2.wav",
        decoding_time_ms=ENGLISH_DECODING_TIME_MS,
    ),
)
REFUSED_NEMO_LINES: dict[str, tuple[bytes, type[ValueError], str]] = {
    "a missing key": (
        render_lines(build_nemo_record(), drop_key(build_nemo_record(), "decoding_time_ms")),
        RunLineError,
        "line 2 lacks decoding_time_ms",
    ),
    "the key of the other reader": (
        render_lines(drop_key(build_nemo_record(), "token_ids") | {"pieces": [" the"]}),
        RunLineError,
        "line 1 lacks token_ids",
    ),
    "an extra key": (
        render_lines(build_nemo_record(text="the cat")),
        RunLineError,
        "line 1 holds the unexpected 'text'",
    ),
    "a language that is not a string": (
        render_lines(build_nemo_record(language=1)),
        RunLineError,
        "line 1: language is 1, not a string",
    ),
    "an utterance id that is not a string": (
        render_lines(build_nemo_record(utterance_id=1)),
        RunLineError,
        "line 1: utterance_id is 1, not a string",
    ),
    "a decoding time that is a boolean": (
        render_lines(build_nemo_record(decoding_time_ms=True)),
        RunLineError,
        "line 1: decoding_time_ms is True, not a number",
    ),
    "a decoding time that is a string": (
        render_lines(build_nemo_record(decoding_time_ms="1.5")),
        RunLineError,
        "line 1: decoding_time_ms is '1.5', not a number",
    ),
    "token ids that are not a list": (
        render_lines(build_nemo_record() | {"token_ids": "1 2"}),
        RunLineError,
        "line 1: token_ids is '1 2', not a list",
    ),
    "a token id that is a boolean": (
        render_lines(build_nemo_record((1, True))),
        RunLineError,
        "line 1: token_ids holds True, not an integer",
    ),
    "a token id that is a float": (
        render_lines(build_nemo_record((1, 2.0))),
        RunLineError,
        "line 1: token_ids holds 2.0, not an integer",
    ),
    "a token id that is a string": (
        render_lines(build_nemo_record((1, "2"))),
        RunLineError,
        "line 1: token_ids holds '2', not an integer",
    ),
    "more token ids than the bound": (
        render_lines(build_nemo_record((1,) * (MAX_TOKEN_COUNT + 1))),
        RunLineError,
        f"line 1: token_ids holds {MAX_TOKEN_COUNT + 1} tokens, more than {MAX_TOKEN_COUNT}",
    ),
    "a token id beyond the vocabulary": (
        render_lines(build_nemo_record(), build_nemo_record((1, 6), utterance_id="2.wav")),
        RunLineError,
        "line 2: the token id 6 is outside the vocabulary",
    ),
    "the blank's id": (
        render_lines(build_nemo_record((1, 5))),
        RunLineError,
        "line 1: the token id 5 is the blank",
    ),
    "a line that is not an object": (
        b"[1, 2]\n",
        HypothesisError,
        "line 1 is not a JSON object",
    ),
    "two lines for one utterance": (
        render_lines(build_nemo_record(), build_nemo_record()),
        HypothesisError,
        "the utterance 1.wav of fr_fr has two hypotheses",
    ),
    "a language outside the bench": (
        render_lines(build_nemo_record(language="de_de")),
        HypothesisError,
        "is in 'de_de', not a language of its run",
    ),
}
REFUSED_ONNX_ASR_LINES: dict[str, tuple[bytes, type[ValueError], str]] = {
    "a missing key": (
        render_lines(drop_key(build_onnx_asr_record(), "language")),
        RunLineError,
        "line 1 lacks language",
    ),
    "the key of the other reader": (
        render_lines(drop_key(build_onnx_asr_record(), "pieces") | {"token_ids": [1]}),
        RunLineError,
        "line 1 lacks pieces",
    ),
    "an extra key": (
        render_lines(build_onnx_asr_record(token_ids=[1])),
        RunLineError,
        "line 1 holds the unexpected 'token_ids'",
    ),
    "pieces that are not a list": (
        render_lines(build_onnx_asr_record() | {"pieces": " the cat"}),
        RunLineError,
        "line 1: pieces is ' the cat', not a list",
    ),
    "a piece that is not a string": (
        render_lines(build_onnx_asr_record((" the", 2))),
        RunLineError,
        "line 1: pieces holds 2, not a string",
    ),
    "more pieces than the bound": (
        render_lines(build_onnx_asr_record((" the",) * (MAX_TOKEN_COUNT + 1))),
        RunLineError,
        f"line 1: pieces holds {MAX_TOKEN_COUNT + 1} tokens, more than {MAX_TOKEN_COUNT}",
    ),
    "a piece outside the vocabulary": (
        render_lines(
            build_onnx_asr_record(), build_onnx_asr_record((" dog",), utterance_id="2.wav")
        ),
        RunLineError,
        "line 2: the piece ' dog' is not in the vocabulary",
    ),
    "the blank's piece": (
        render_lines(build_onnx_asr_record((" the", "<blk>"))),
        RunLineError,
        "line 1: the piece '<blk>' is the blank",
    ),
    "two lines for one utterance": (
        render_lines(build_onnx_asr_record(), build_onnx_asr_record()),
        HypothesisError,
        "the utterance 1.wav of fr_fr has two hypotheses",
    ),
}


class TestTheRunReaders:
    class TestTheReadNemoLinesWithFunction:
        class TestWhenEachLineHoldsTheTokenIdsOfAnUtterance:
            @pytest.fixture
            def hypotheses(self) -> tuple[Hypothesis, ...]:
                return read_nemo_lines_with(NEMO_LINES, CANDIDATE_TEXTS, VOCABULARY_CONTENT_SHA256)

            def test_should_render_each_text_and_count_its_unknown_tokens_from_the_ids(
                self, hypotheses: tuple[Hypothesis, ...]
            ) -> None:
                assert hypotheses == EXPECTED_HYPOTHESES

        class TestWhenTheCandidateHoldsNoVocabulary:
            @pytest.fixture
            def read_act(self) -> Callable[[], tuple[Hypothesis, ...]]:
                return partial(
                    read_nemo_lines_with,
                    NEMO_LINES,
                    {"config.json": b"{}"},
                    VOCABULARY_CONTENT_SHA256,
                )

            def test_should_refuse_the_run(
                self, read_act: Callable[[], tuple[Hypothesis, ...]]
            ) -> None:
                with pytest.raises(
                    VocabularyError, match=re.escape("the candidate holds no vocab.txt")
                ):
                    read_act()

        @pytest.mark.parametrize(
            ("lines", "error", "reason"),
            list(REFUSED_NEMO_LINES.values()),
            ids=list(REFUSED_NEMO_LINES.keys()),
        )
        class TestWhenALineIsMalformed:
            @pytest.fixture
            def read_act(self, lines: bytes) -> Callable[[], tuple[Hypothesis, ...]]:
                return partial(
                    read_nemo_lines_with, lines, CANDIDATE_TEXTS, VOCABULARY_CONTENT_SHA256
                )

            def test_should_refuse_it_and_say_why(
                self,
                read_act: Callable[[], tuple[Hypothesis, ...]],
                error: type[ValueError],
                reason: str,
            ) -> None:
                with pytest.raises(error, match=re.escape(reason)):
                    read_act()

    class TestTheReadOnnxAsrLinesWithFunction:
        class TestWhenEachLineHoldsThePiecesOfAnUtterance:
            @pytest.fixture
            def hypotheses(self) -> tuple[Hypothesis, ...]:
                return read_onnx_asr_lines_with(
                    ONNX_ASR_LINES, CANDIDATE_TEXTS, VOCABULARY_CONTENT_SHA256
                )

            def test_should_render_each_text_and_count_its_unknown_tokens_from_the_ids(
                self, hypotheses: tuple[Hypothesis, ...]
            ) -> None:
                assert hypotheses == EXPECTED_HYPOTHESES

        class TestWhenTheCandidateHoldsNoVocabulary:
            @pytest.fixture
            def read_act(self) -> Callable[[], tuple[Hypothesis, ...]]:
                return partial(
                    read_onnx_asr_lines_with,
                    ONNX_ASR_LINES,
                    {"config.json": b"{}"},
                    VOCABULARY_CONTENT_SHA256,
                )

            def test_should_refuse_the_run(
                self, read_act: Callable[[], tuple[Hypothesis, ...]]
            ) -> None:
                with pytest.raises(
                    VocabularyError, match=re.escape("the candidate holds no vocab.txt")
                ):
                    read_act()

        @pytest.mark.parametrize(
            ("lines", "error", "reason"),
            list(REFUSED_ONNX_ASR_LINES.values()),
            ids=list(REFUSED_ONNX_ASR_LINES.keys()),
        )
        class TestWhenALineIsMalformed:
            @pytest.fixture
            def read_act(self, lines: bytes) -> Callable[[], tuple[Hypothesis, ...]]:
                return partial(
                    read_onnx_asr_lines_with, lines, CANDIDATE_TEXTS, VOCABULARY_CONTENT_SHA256
                )

            def test_should_refuse_it_and_say_why(
                self,
                read_act: Callable[[], tuple[Hypothesis, ...]],
                error: type[ValueError],
                reason: str,
            ) -> None:
                with pytest.raises(error, match=re.escape(reason)):
                    read_act()

    @pytest.mark.parametrize(
        ("read_lines", "lines"),
        [(read_nemo_lines, NEMO_LINES), (read_onnx_asr_lines, ONNX_ASR_LINES)],
        ids=["read_nemo_lines", "read_onnx_asr_lines"],
    )
    class TestThePublicReaders:
        class TestWhenTheVocabularyIsNotThePinnedOne:
            @pytest.fixture
            def read_act(
                self,
                read_lines: RunReader,
                lines: bytes,
            ) -> Callable[[], tuple[Hypothesis, ...]]:
                return partial(read_lines, lines, CANDIDATE_TEXTS)

            def test_should_refuse_the_run(
                self, read_act: Callable[[], tuple[Hypothesis, ...]]
            ) -> None:
                with pytest.raises(
                    VocabularyError,
                    match=re.escape(
                        f"the vocabulary's sha256 is {VOCABULARY_CONTENT_SHA256}, not "
                        f"{VOCABULARY_SHA256}"
                    ),
                ):
                    read_act()
