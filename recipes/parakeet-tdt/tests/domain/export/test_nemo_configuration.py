import re
from collections.abc import Callable
from functools import partial
from pathlib import PurePosixPath

import pytest

from mostt_recipe_parakeet_tdt.domain.export.nemo_configuration import (
    NemoConfigurationError,
    TokenizerPreparation,
    list_dataset_keys,
    prepare_tokenizer,
)

TOKENIZER_DIRECTORY = PurePosixPath("/tmp/nemo-tokenizer")
MODEL_FILE = "902810505b78491d95378fa3000d1bc6_tokenizer.model"
VOCABULARY_FILE = "ddbad8f2a94d4186963950e53809d27f_vocab.txt"
PIECES_FILE = "0ee587b5d4b94f48993dcccf9868ea77_tokenizer.vocab"
V3_TOKENIZER_SETTINGS: dict[str, object] = {
    "dir": None,
    "type": "bpe",
    "model_path": f"nemo:{MODEL_FILE}",
    "vocab_path": f"nemo:{VOCABULARY_FILE}",
    "spe_tokenizer_vocab": f"nemo:{PIECES_FILE}",
}
V3_ARCHIVE_MEMBERS = (
    ".",
    f"./{PIECES_FILE}",
    f"./{MODEL_FILE}",
    f"./{VOCABULARY_FILE}",
    "./model_config.yaml",
    "./model_weights.ckpt",
)
V3_CONFIGURATION_KEYS = (
    "sample_rate",
    "train_ds",
    "validation_ds",
    "test_ds",
    "tokenizer",
    "preprocessor",
    "encoder",
)
REFUSED_PREPARATIONS = {
    "a tokenizer that names no file of the archive": (
        {"dir": None, "type": "bpe", "model_path": "tokenizer.model"},
        V3_ARCHIVE_MEMBERS,
        "names no file of the archive",
    ),
    "a reference that climbs out of the tokenizer's directory": (
        {"model_path": "nemo:../model_weights.ckpt"},
        V3_ARCHIVE_MEMBERS,
        "the tokenizer's model_path names '../model_weights.ckpt', which is not a file name",
    ),
    "a reference to a file the archive lacks": (
        {"model_path": "nemo:missing.model"},
        V3_ARCHIVE_MEMBERS,
        "the archive holds 0 members named missing.model at its root",
    ),
    "a reference to a file the archive holds only below its root": (
        {"model_path": f"nemo:{MODEL_FILE}"},
        (f"./nested/{MODEL_FILE}",),
        f"the archive holds 0 members named {MODEL_FILE} at its root",
    ),
    "a reference to a file the archive holds twice": (
        {"model_path": f"nemo:{MODEL_FILE}"},
        (MODEL_FILE, f"./{MODEL_FILE}"),
        f"the archive holds 2 members named {MODEL_FILE} at its root",
    ),
}


class TestTheNemoConfiguration:
    class TestThePrepareTokenizerFunction:
        class TestWhenGivenTheTokenizerOfTheV3Archive:
            @pytest.fixture
            def tokenizer_preparation(self) -> TokenizerPreparation:
                return prepare_tokenizer(
                    V3_TOKENIZER_SETTINGS, V3_ARCHIVE_MEMBERS, TOKENIZER_DIRECTORY
                )

            def test_should_extract_its_three_members_and_never_the_checkpoint(
                self, tokenizer_preparation: TokenizerPreparation
            ) -> None:
                assert sorted(tokenizer_preparation.archive_members) == [
                    f"./{PIECES_FILE}",
                    f"./{MODEL_FILE}",
                    f"./{VOCABULARY_FILE}",
                ]

            def test_should_point_each_reference_and_the_directory_at_the_extracted_files(
                self, tokenizer_preparation: TokenizerPreparation
            ) -> None:
                assert dict(tokenizer_preparation.tokenizer_settings) == {
                    "model_path": f"/tmp/nemo-tokenizer/{MODEL_FILE}",
                    "vocab_path": f"/tmp/nemo-tokenizer/{VOCABULARY_FILE}",
                    "spe_tokenizer_vocab": f"/tmp/nemo-tokenizer/{PIECES_FILE}",
                    "dir": "/tmp/nemo-tokenizer",
                }

        class TestWhenTwoSettingsNameTheSameFile:
            @pytest.fixture
            def tokenizer_preparation(self) -> TokenizerPreparation:
                return prepare_tokenizer(
                    {"model_path": f"nemo:{MODEL_FILE}", "spe_model": f"nemo:{MODEL_FILE}"},
                    V3_ARCHIVE_MEMBERS,
                    TOKENIZER_DIRECTORY,
                )

            def test_should_extract_the_file_once(
                self, tokenizer_preparation: TokenizerPreparation
            ) -> None:
                assert tokenizer_preparation.archive_members == (f"./{MODEL_FILE}",)

        @pytest.mark.parametrize(
            ("tokenizer_settings", "archive_members", "reason"),
            list(REFUSED_PREPARATIONS.values()),
            ids=list(REFUSED_PREPARATIONS.keys()),
        )
        class TestWhenTheTokenizerCannotBeReadFromTheArchiveAlone:
            @pytest.fixture
            def prepare_act(
                self, tokenizer_settings: dict[str, object], archive_members: tuple[str, ...]
            ) -> Callable[[], TokenizerPreparation]:
                return partial(
                    prepare_tokenizer, tokenizer_settings, archive_members, TOKENIZER_DIRECTORY
                )

            def test_should_refuse_it_and_say_why(
                self, prepare_act: Callable[[], TokenizerPreparation], reason: str
            ) -> None:
                with pytest.raises(NemoConfigurationError, match=re.escape(reason)):
                    prepare_act()

    class TestTheListDatasetKeysFunction:
        class TestWhenGivenTheSectionsOfTheV3Configuration:
            @pytest.fixture
            def dataset_keys(self) -> tuple[str, ...]:
                return list_dataset_keys(V3_CONFIGURATION_KEYS)

            def test_should_name_its_three_datasets_which_the_build_sets_to_null(
                self, dataset_keys: tuple[str, ...]
            ) -> None:
                assert dataset_keys == ("train_ds", "validation_ds", "test_ds")

        class TestWhenAConfigurationHoldsNoTestSet:
            @pytest.fixture
            def dataset_keys(self) -> tuple[str, ...]:
                return list_dataset_keys(("train_ds", "tokenizer"))

            def test_should_name_only_the_datasets_it_holds(
                self, dataset_keys: tuple[str, ...]
            ) -> None:
                assert dataset_keys == ("train_ds",)
