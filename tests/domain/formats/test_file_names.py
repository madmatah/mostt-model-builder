import pytest

from mostt_model_builder.domain.formats.file_names import REFUSED_SUFFIXES, find_refused_suffixes

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
