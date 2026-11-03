import json
import random
import re
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from functools import partial

import pytest

from mostt_model_builder.domain.json_documents.json_depth import (
    MAX_JSON_DEPTH,
    JsonDepthError,
    refuse_deep_json,
)

DEPTH = 32
DEPTH_BEYOND = DEPTH + 1
HALF_DEPTH = DEPTH // 2
ACCEPTED_DOCUMENTS = {
    "arrays nested to the depth": "[" * DEPTH + "]" * DEPTH,
    "objects nested to the depth": '{"a": ' * DEPTH + "1" + "}" * DEPTH,
    "both kinds of bracket nested to the depth": '[{"a": ' * HALF_DEPTH + "1" + "}]" * HALF_DEPTH,
    "brackets beyond the depth inside a string": '["' + "[{" * DEPTH_BEYOND + '"]',
    "brackets beyond the depth inside a key": '{"' + "[{" * DEPTH_BEYOND + '": 1}',
    "brackets beyond the depth inside a string after an escaped quote": (
        '["\\"' + "[" * DEPTH_BEYOND + '"]'
    ),
    "brackets beyond the depth inside a string after an escaped backslash": (
        '["\\\\' + "[" * DEPTH_BEYOND + '"]'
    ),
    "brackets beyond the depth inside a string after an escaped backslash and quote": (
        '["\\\\\\"' + "[" * DEPTH_BEYOND + '"]'
    ),
    "an escaped backslash before the closing quote of a string": '["\\\\", 1]',
    "brackets beyond the depth after a string that never closes": '["' + "[" * DEPTH_BEYOND,
    "a lone backslash at the end of the text": '["\\',
    "an empty text": "",
    "closing brackets with no opening one": "]}" * DEPTH_BEYOND,
}
REFUSED_DOCUMENTS = {
    "arrays nested beyond the depth": "[" * DEPTH_BEYOND + "]" * DEPTH_BEYOND,
    "objects nested beyond the depth": '{"a": ' * DEPTH_BEYOND + "1" + "}" * DEPTH_BEYOND,
    "both kinds of bracket nested beyond the depth": '[{"a": ' * HALF_DEPTH
    + "[1]"
    + "}]" * HALF_DEPTH,
    "brackets after a string that ends with an escaped backslash": (
        '["\\\\", ' + "[" * DEPTH_BEYOND
    ),
}
PROPERTY_SEED = 20261104
GENERATED_DOCUMENT_COUNT = 300
REFUSED_GENERATED_DOCUMENT_COUNT = 336
ACCEPTED_GENERATED_DOCUMENT_COUNT = 864
OFTEN_DRAWN_DEPTHS = range(DEPTH - 2, DEPTH + 3)
ANY_DEPTH = range(41)
OFTEN_DRAWN_DEPTH_RATIO = 0.5
LIST_RATIO = 0.5
SMALL_INTEGERS = range(-9, 10)
STRING_ALPHABET = '"\\[]{}a é'
MAX_STRING_LENGTH = 4
MAX_SIBLING_COUNT = 2
MAX_SIBLING_DEPTH = 1
COMPACT_SEPARATORS = (",", ":")
RENDERINGS = (
    partial(json.dumps, ensure_ascii=True, separators=COMPACT_SEPARATORS),
    partial(json.dumps, ensure_ascii=False, separators=COMPACT_SEPARATORS),
    partial(json.dumps, ensure_ascii=True, indent=1),
    partial(json.dumps, ensure_ascii=False, indent=1),
)
PREFIXED_DOCUMENT_COUNT = 12
PREFIXED_DOCUMENT_WITHIN_THE_DEPTH_COUNT = 44
LARGE_UNCLOSED_TEXT = '["' + '\\"' * 50_000

type JsonValue = bool | int | str | list[JsonValue] | dict[str, JsonValue] | None


class CharacterCountingText(str):
    read_count: int = 0

    def __iter__(self) -> Iterator[str]:
        for character in str.__iter__(self):
            self.read_count += 1
            yield character


@dataclass(frozen=True)
class TextScan:
    read_count: int
    given_back_text: str


def draw_string(generator: random.Random) -> str:
    return "".join(
        generator.choice(STRING_ALPHABET) for _ in range(generator.randint(0, MAX_STRING_LENGTH))
    )


def draw_scalar(generator: random.Random) -> JsonValue:
    return generator.choice(
        (None, True, False, generator.choice(SMALL_INTEGERS), draw_string(generator))
    )


def draw_json_value(generator: random.Random, depth: int) -> JsonValue:
    if depth == 0:
        return draw_scalar(generator)
    children = [draw_json_value(generator, depth - 1)] + [
        draw_json_value(generator, generator.randint(0, min(depth - 1, MAX_SIBLING_DEPTH)))
        for _ in range(generator.randint(0, MAX_SIBLING_COUNT))
    ]
    generator.shuffle(children)
    if generator.random() < LIST_RATIO:
        return children
    return {draw_string(generator) + str(index): child for index, child in enumerate(children)}


def draw_depth(generator: random.Random) -> int:
    if generator.random() < OFTEN_DRAWN_DEPTH_RATIO:
        return generator.choice(OFTEN_DRAWN_DEPTHS)
    return generator.choice(ANY_DEPTH)


def measure_depth(json_value: JsonValue) -> int:
    if isinstance(json_value, list):
        return 1 + max((measure_depth(child) for child in json_value), default=0)
    if isinstance(json_value, dict):
        return 1 + max((measure_depth(child) for child in json_value.values()), default=0)
    return 0


def generate_documents(seed: int, count: int) -> list[tuple[JsonValue, str]]:
    generator = random.Random(seed)
    json_values = [draw_json_value(generator, draw_depth(generator)) for _ in range(count)]
    return [(json_value, render(json_value)) for json_value in json_values for render in RENDERINGS]


def scan_depth(text: str) -> str | None:
    try:
        return refuse_deep_json(text, DEPTH)
    except JsonDepthError:
        return None


def describe_mismatch(json_value: JsonValue, text: str) -> str | None:
    given_back_text = scan_depth(text)
    is_refused = given_back_text is None
    if is_refused != (measure_depth(json_value) > DEPTH):
        return f"seed {PROPERTY_SEED}: {'refused' if is_refused else 'accepted'} {text!r}"
    if given_back_text is not None and json.loads(given_back_text) != json_value:
        return f"seed {PROPERTY_SEED}: json.loads did not give the value back from {text!r}"
    return None


class TestTheJsonDepth:
    class TestTheRefuseDeepJsonFunction:
        @pytest.mark.parametrize(
            "text", list(ACCEPTED_DOCUMENTS.values()), ids=list(ACCEPTED_DOCUMENTS.keys())
        )
        class TestWhenTheDocumentNestsNoDeeperThanTheDepth:
            @pytest.fixture
            def accepted_text(self, text: str) -> str:
                return refuse_deep_json(text, DEPTH)

            def test_should_give_the_text_back_unchanged(
                self, accepted_text: str, text: str
            ) -> None:
                assert accepted_text == text

        @pytest.mark.parametrize(
            "text", list(REFUSED_DOCUMENTS.values()), ids=list(REFUSED_DOCUMENTS.keys())
        )
        class TestWhenTheDocumentNestsDeeperThanTheDepth:
            @pytest.fixture
            def refuse_act(self, text: str) -> Callable[[], str]:
                return partial(refuse_deep_json, text, DEPTH)

            def test_should_refuse_it_and_name_the_depth(
                self, refuse_act: Callable[[], str]
            ) -> None:
                with pytest.raises(
                    JsonDepthError,
                    match=f"^{re.escape(f'the document nests deeper than {DEPTH} levels')}$",
                ):
                    refuse_act()

        class TestWhenGivenAnyDocumentThatJsonWrites:
            def test_should_refuse_it_exactly_when_its_value_nests_deeper_than_the_depth(
                self,
            ) -> None:
                documents = generate_documents(PROPERTY_SEED, GENERATED_DOCUMENT_COUNT)
                mismatches = [
                    mismatch
                    for json_value, text in documents
                    if (mismatch := describe_mismatch(json_value, text)) is not None
                ]
                refused_count = sum(scan_depth(text) is None for _, text in documents)
                assert (refused_count, len(documents) - refused_count, mismatches) == (
                    REFUSED_GENERATED_DOCUMENT_COUNT,
                    ACCEPTED_GENERATED_DOCUMENT_COUNT,
                    [],
                )

        class TestWhenGivenAnyPrefixOfSuchADocument:
            def test_should_refuse_no_prefix_of_a_document_within_the_depth_and_raise_nothing_else(
                self,
            ) -> None:
                documents = [
                    (measure_depth(json_value) <= DEPTH, text)
                    for json_value, text in generate_documents(
                        PROPERTY_SEED, PREFIXED_DOCUMENT_COUNT
                    )
                ]
                refused_prefixes = [
                    text[:length]
                    for is_within_the_depth, text in documents
                    for length in range(len(text) + 1)
                    if scan_depth(text[:length]) is None and is_within_the_depth
                ]
                within_the_depth_count = sum(
                    is_within_the_depth for is_within_the_depth, _ in documents
                )
                assert (len(documents), within_the_depth_count, refused_prefixes) == (
                    PREFIXED_DOCUMENT_COUNT * len(RENDERINGS),
                    PREFIXED_DOCUMENT_WITHIN_THE_DEPTH_COUNT,
                    [],
                )

        class TestWhenTheTextIsALargeStringThatNeverCloses:
            @pytest.fixture
            def text_scan(self) -> TextScan:
                text = CharacterCountingText(LARGE_UNCLOSED_TEXT)
                given_back_text = refuse_deep_json(text, DEPTH)
                return TextScan(text.read_count, given_back_text)

            def test_should_read_each_character_once(self, text_scan: TextScan) -> None:
                assert text_scan == TextScan(len(LARGE_UNCLOSED_TEXT), LARGE_UNCLOSED_TEXT)

    class TestTheMaxJsonDepthConstant:
        def test_should_bound_the_depth_at_32_levels(self) -> None:
            assert MAX_JSON_DEPTH == DEPTH
