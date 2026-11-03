from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import PurePosixPath

from mostt_model_builder.domain.formats.file_names import is_file_name

# NeMo names a file of its archive in its configuration as `nemo:<file name>`, and resolves it
# only when it restores the whole archive, which would unpickle its checkpoint.
ARCHIVE_REFERENCE_PREFIX = "nemo:"
ARCHIVE_ROOT_PREFIX = "./"
# NeMo would build a data loader for each of these sections, which name the datasets its authors
# trained on, absent here.
DATASET_KEYS = ("train_ds", "validation_ds", "test_ds")
TOKENIZER_DIRECTORY_KEY = "dir"


class NemoConfigurationError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class TokenizerPreparation:
    archive_members: tuple[str, ...]
    tokenizer_settings: Mapping[str, str]


def prepare_tokenizer(
    tokenizer_settings: Mapping[str, object],
    archive_members: Sequence[str],
    tokenizer_directory: PurePosixPath,
) -> TokenizerPreparation:
    referenced_files = find_archive_references(tokenizer_settings)
    if not referenced_files:
        raise NemoConfigurationError(
            "the tokenizer's configuration names no file of the archive, "
            f"as {ARCHIVE_REFERENCE_PREFIX}<file name>"
        )
    members = {
        file_name: find_archive_member(file_name, archive_members)
        for file_name in sorted(set(referenced_files.values()))
    }
    settings = {
        key: str(tokenizer_directory / file_name) for key, file_name in referenced_files.items()
    }
    settings[TOKENIZER_DIRECTORY_KEY] = str(tokenizer_directory)
    return TokenizerPreparation(tuple(members.values()), settings)


def find_archive_references(tokenizer_settings: Mapping[str, object]) -> dict[str, str]:
    references: dict[str, str] = {}
    for key, setting in tokenizer_settings.items():
        if not isinstance(setting, str) or not setting.startswith(ARCHIVE_REFERENCE_PREFIX):
            continue
        file_name = setting.removeprefix(ARCHIVE_REFERENCE_PREFIX)
        if not is_file_name(file_name):
            raise NemoConfigurationError(
                f"the tokenizer's {key} names {file_name!a}, which is not a file name"
            )
        references[key] = file_name
    return references


def find_archive_member(file_name: str, archive_members: Sequence[str]) -> str:
    matches = [
        member
        for member in archive_members
        if member in (file_name, ARCHIVE_ROOT_PREFIX + file_name)
    ]
    if len(matches) != 1:
        raise NemoConfigurationError(
            f"the archive holds {len(matches)} members named {file_name} at its root, "
            "where the tokenizer reads one"
        )
    return matches[0]


def list_dataset_keys(configuration_keys: Iterable[str]) -> tuple[str, ...]:
    present_keys = set(configuration_keys)
    return tuple(key for key in DATASET_KEYS if key in present_keys)
