import pytest

from mostt_model_builder.domain.formats.file_names import (
    REFUSED_SUFFIXES,
    find_refused_suffixes,
    is_file_name,
)

STEM = "model"
LAST_SUFFIX = ".txt"
MOSTT_STORE_REFUSED_SUFFIXES = frozenset(
    {
        *(".pt", ".pth", ".ckpt", ".pkl", ".pickle", ".nemo", ".bin", ".joblib", ".npy"),
        *(".npz", ".py", ".pyc", ".pyo", ".so", ".sh", ".exe", ".dll", ".dylib", ".jar"),
        *(".zip", ".whl", ".egg", ".tar", ".gz", ".tgz", ".bz2", ".xz", ".zst", ".7z"),
    }
)
NAMES_WITHOUT_A_REFUSED_SUFFIX = {
    "an onnx file": "encoder-model.onnx",
    "external data": "encoder-model.onnx.data",
    "a vocabulary": "vocab.txt",
    "a json file": "config.json",
    "a name without a suffix": "LICENSE",
    "a refused suffix as the stem": "nemo",
    "a refused suffix inside a longer suffix": "model.binary",
    "a longer suffix that starts like a refused one": "model.gzip",
    "a suffix that ends like a refused one": "model.mypy",
}


def write_in_mixed_case(suffix: str) -> str:
    return "".join(
        character.upper() if character_index % 2 else character.lower()
        for character_index, character in enumerate(suffix)
    )


CASES = {"lowercase": str.lower, "uppercase": str.upper, "mixed case": write_in_mixed_case}
REFUSED_NAMES = [
    pytest.param(
        f"{STEM}{write_in_case(suffix)}{position_suffix}",
        suffix,
        id=f"{suffix} in {case_name} {position_name}",
    )
    for suffix in sorted(REFUSED_SUFFIXES)
    for case_name, write_in_case in CASES.items()
    for position_name, position_suffix in (("as the last suffix", ""), ("inside", LAST_SUFFIX))
]


ALPHANUMERIC_CHARACTERS = frozenset(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
)
LATER_ONLY_CHARACTERS = frozenset("._-")
ASCII_CHARACTERS = [chr(code_point) for code_point in range(128)]
CHARACTERS_BEYOND_ASCII = {
    "an accented letter": "\u00e9",
    "a no-break space": "\u00a0",
    "a right-to-left override": "\u202e",
}
GRAMMAR_STEM = "m"
LONGEST_LENGTH = 128
DOT_NAMES = (".", "..")


def describe_character(character: str) -> str:
    return f"U+{ord(character):04X}"


def build_name_of_length(length: int) -> str:
    return GRAMMAR_STEM * length


class TestTheFileNames:
    class TestTheFindRefusedSuffixesFunction:
        @pytest.mark.parametrize(("name", "suffix"), REFUSED_NAMES)
        class TestWhenANameHoldsARefusedSuffix:
            @pytest.fixture
            def refused_suffixes(self, name: str) -> tuple[str, ...]:
                return find_refused_suffixes(name, REFUSED_SUFFIXES)

            def test_should_find_it_whatever_its_case_and_wherever_it_stands(
                self, refused_suffixes: tuple[str, ...], suffix: str
            ) -> None:
                assert refused_suffixes == (suffix,)

        @pytest.mark.parametrize(
            "name",
            list(NAMES_WITHOUT_A_REFUSED_SUFFIX.values()),
            ids=list(NAMES_WITHOUT_A_REFUSED_SUFFIX.keys()),
        )
        class TestWhenANameHoldsNoRefusedSuffix:
            @pytest.fixture
            def refused_suffixes(self, name: str) -> tuple[str, ...]:
                return find_refused_suffixes(name, REFUSED_SUFFIXES)

            def test_should_find_none(self, refused_suffixes: tuple[str, ...]) -> None:
                assert refused_suffixes == ()

        class TestWhenANameHoldsTwoRefusedSuffixes:
            @pytest.fixture
            def refused_suffixes(self) -> tuple[str, ...]:
                return find_refused_suffixes("model.tar.gz", REFUSED_SUFFIXES)

            def test_should_find_both_in_their_order(
                self, refused_suffixes: tuple[str, ...]
            ) -> None:
                assert refused_suffixes == (".tar", ".gz")

    class TestTheRefusedSuffixes:
        def test_should_hold_the_suffixes_of_mostts_model_store_so_every_published_file_passes_it(
            self,
        ) -> None:
            assert MOSTT_STORE_REFUSED_SUFFIXES == REFUSED_SUFFIXES

    class TestTheIsFileNameFunction:
        @pytest.mark.parametrize("character", ASCII_CHARACTERS, ids=describe_character)
        class TestWhenAnAsciiCharacterComesFirst:
            @pytest.fixture
            def is_accepted(self, character: str) -> bool:
                return is_file_name(character + GRAMMAR_STEM)

            def test_should_accept_only_a_letter_or_a_digit_so_no_name_hides_or_reads_as_an_option(
                self, is_accepted: bool, character: str
            ) -> None:
                assert is_accepted is (character in ALPHANUMERIC_CHARACTERS)

        @pytest.mark.parametrize("character", ASCII_CHARACTERS, ids=describe_character)
        class TestWhenAnAsciiCharacterComesLater:
            @pytest.fixture
            def is_accepted(self, character: str) -> bool:
                return is_file_name(GRAMMAR_STEM + character)

            def test_should_accept_only_a_letter_a_digit_a_dot_an_underscore_or_a_dash(
                self, is_accepted: bool, character: str
            ) -> None:
                assert is_accepted is (character in ALPHANUMERIC_CHARACTERS | LATER_ONLY_CHARACTERS)

        @pytest.mark.parametrize(
            "character",
            list(CHARACTERS_BEYOND_ASCII.values()),
            ids=list(CHARACTERS_BEYOND_ASCII.keys()),
        )
        class TestWhenACharacterBeyondAsciiAppears:
            @pytest.fixture
            def acceptances(self, character: str) -> tuple[bool, bool]:
                return (
                    is_file_name(character + GRAMMAR_STEM),
                    is_file_name(GRAMMAR_STEM + character),
                )

            def test_should_refuse_it_first_and_later(self, acceptances: tuple[bool, bool]) -> None:
                assert acceptances == (False, False)

        @pytest.mark.parametrize(
            ("length", "is_expected"),
            [
                pytest.param(0, False, id="no character"),
                pytest.param(1, True, id="one character"),
                pytest.param(LONGEST_LENGTH, True, id="128 characters"),
                pytest.param(LONGEST_LENGTH + 1, False, id="129 characters"),
            ],
        )
        class TestWhenANameHasALength:
            @pytest.fixture
            def is_accepted(self, length: int) -> bool:
                return is_file_name(build_name_of_length(length))

            def test_should_accept_one_to_128_characters(
                self, is_accepted: bool, is_expected: bool
            ) -> None:
                assert is_accepted is is_expected

        @pytest.mark.parametrize("name", DOT_NAMES)
        class TestWhenTheNameIsADotSegment:
            @pytest.fixture
            def is_accepted(self, name: str) -> bool:
                return is_file_name(name)

            def test_should_refuse_it_so_that_no_name_points_at_a_directory(
                self, is_accepted: bool
            ) -> None:
                assert is_accepted is False
