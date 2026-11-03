import pytest

from mostt_model_builder.domain.bench.normalization import normalize_words

NORMALIZED_TEXTS = {
    "lowercase": ("Le Chat DORT", ("le", "chat", "dort")),
    "a compatibility form folded by nfkc": ("\N{LATIN SMALL LIGATURE FI}n", ("fin",)),
    "a full-width letter folded by nfkc": ("\N{FULLWIDTH LATIN CAPITAL LETTER A}b", ("ab",)),
    "a straight apostrophe kept inside a word": ("l'été", ("l'été",)),
    "a right single quotation mark read as an apostrophe": (
        "aujourd\N{RIGHT SINGLE QUOTATION MARK}hui",
        ("aujourd'hui",),
    ),
    "a left single quotation mark read as an apostrophe, as FLEURS en writes it": (
        "it\N{LEFT SINGLE QUOTATION MARK}s",
        ("it's",),
    ),
    "an apostrophe at the start of a word dropped": ("'tis done", ("tis", "done")),
    "an apostrophe at the end of a word dropped": ("the dogs' bowls", ("the", "dogs", "bowls")),
    "a quotation in apostrophes dropped": ("say 'yes' now", ("say", "yes", "now")),
    "an underscore kept as a character of a word": ("snake_case", ("snake_case",)),
    "the punctuation of french": (
        "Quoi\N{NO-BREAK SPACE}? «\N{NO-BREAK SPACE}Oui\N{NO-BREAK SPACE}» ; non !",
        ("quoi", "oui", "non"),
    ),
    "the punctuation of english": (
        'Well, "yes" -- (maybe): no.',
        ("well", "yes", "maybe", "no"),
    ),
    "a hyphen splitting a word": ("peut-être", ("peut", "être")),
    "digits kept as words": ("in 1999, 3.5 km", ("in", "1999", "3", "5", "km")),
    "an unknown token read as the word unk": ("the <unk> cat", ("the", "unk", "cat")),
    "an unknown token glued to a word": ("the<unk> cat", ("the", "unk", "cat")),
    "no word at all": (" ... ", ()),
    "an empty text": ("", ()),
}


class TestTheNormalization:
    class TestTheNormalizeWordsFunction:
        @pytest.mark.parametrize(
            ("text", "words"),
            list(NORMALIZED_TEXTS.values()),
            ids=list(NORMALIZED_TEXTS.keys()),
        )
        class TestWhenGivenAText:
            @pytest.fixture
            def normalized_words(self, text: str) -> tuple[str, ...]:
                return normalize_words(text)

            def test_should_compare_it_as_these_words(
                self, normalized_words: tuple[str, ...], words: tuple[str, ...]
            ) -> None:
                assert normalized_words == words
