import re
import tomllib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum

from mostt_model_builder.domain.formats.file_names import REFUSED_SUFFIXES, find_refused_suffixes

SOURCES_KEY = "sources"
TOP_LEVEL_KEYS = frozenset({SOURCES_KEY})
PIN_KEYS = frozenset({"repository_type", "repository", "revision", "path", "size_bytes", "sha256"})
REPOSITORY_GRAMMAR = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,95}/[A-Za-z0-9][A-Za-z0-9._-]{0,95}")
REVISION_GRAMMAR = re.compile(r"[0-9a-f]{40}")
PATH_SEGMENT_GRAMMAR = re.compile(r"[A-Za-z0-9._-]+")
SHA256_GRAMMAR = re.compile(r"[0-9a-f]{64}")
PATH_SEPARATOR = "/"
DOT_SEGMENTS = frozenset({".", ".."})
MIN_SIZE_BYTES = 1
# A bound that only refuses a size no source could mean: the largest checkpoint a build reads
# is under 3 GiB.
MAX_SIZE_BYTES = 2**36
# FLEURS ships its audio as `.tar.gz` archives, which a container reads member by member.
ARCHIVE_SUFFIXES_OF_THE_BENCH = frozenset({".tar", ".gz"})
REFUSED_SOURCE_SUFFIXES = REFUSED_SUFFIXES - ARCHIVE_SUFFIXES_OF_THE_BENCH


class SourcePinError(ValueError):
    pass


class RepositoryType(StrEnum):
    MODEL = "model"
    DATASET = "dataset"


@dataclass(frozen=True, slots=True)
class SourcePin:
    repository_type: RepositoryType
    repository: str
    revision: str
    path: str
    size_bytes: int
    sha256: str

    @property
    def layout_segments(self) -> tuple[str, ...]:
        return (
            *self.repository.split(PATH_SEPARATOR),
            self.revision,
            *self.path.split(PATH_SEPARATOR),
        )


def parse_source_pins(text: str) -> tuple[SourcePin, ...]:
    try:
        document = tomllib.loads(text)
    except (tomllib.TOMLDecodeError, RecursionError) as error:
        raise SourcePinError(f"the sources are not valid TOML: {error}") from error
    require_exact_keys(document, TOP_LEVEL_KEYS, "the sources")
    entries = document[SOURCES_KEY]
    if not isinstance(entries, list) or not entries:
        raise SourcePinError(f"{SOURCES_KEY} must be a non-empty array of tables")
    pins = tuple(
        parse_source_pin(entry, f"{SOURCES_KEY}[{entry_index}]")
        for entry_index, entry in enumerate(entries)
    )
    refuse_overlapping_pins(pins)
    return pins


def parse_source_pin(entry: object, where: str) -> SourcePin:
    if not isinstance(entry, dict):
        raise SourcePinError(f"{where} must be a table")
    require_exact_keys(entry, PIN_KEYS, where)
    return SourcePin(
        repository_type=parse_repository_type(entry["repository_type"], f"{where}.repository_type"),
        repository=parse_grammar(entry["repository"], REPOSITORY_GRAMMAR, f"{where}.repository"),
        revision=parse_grammar(entry["revision"], REVISION_GRAMMAR, f"{where}.revision"),
        path=parse_path(entry["path"], f"{where}.path"),
        size_bytes=parse_size(entry["size_bytes"], f"{where}.size_bytes"),
        sha256=parse_grammar(entry["sha256"], SHA256_GRAMMAR, f"{where}.sha256"),
    )


def require_exact_keys(table: Mapping[str, object], keys: frozenset[str], where: str) -> None:
    unknown_keys = sorted(set(table) - keys)
    if unknown_keys:
        raise SourcePinError(f"{where}: unknown key {unknown_keys[0]!a}")
    missing_keys = sorted(keys - set(table))
    if missing_keys:
        raise SourcePinError(f"{where}: missing key {missing_keys[0]}")


def parse_repository_type(field: object, where: str) -> RepositoryType:
    if not isinstance(field, str) or field not in set(RepositoryType):
        allowed = ", ".join(repository_type.value for repository_type in RepositoryType)
        raise SourcePinError(f"{where}: {field!a} is not one of {allowed}")
    return RepositoryType(field)


def parse_grammar(field: object, grammar: re.Pattern[str], where: str) -> str:
    if not isinstance(field, str) or not grammar.fullmatch(field):
        raise SourcePinError(f"{where}: {field!a} does not match {grammar.pattern}")
    return field


def parse_path(field: object, where: str) -> str:
    if not isinstance(field, str):
        raise SourcePinError(f"{where}: {field!a} is not a string")
    for segment in field.split(PATH_SEPARATOR):
        if segment in DOT_SEGMENTS or not PATH_SEGMENT_GRAMMAR.fullmatch(segment):
            raise SourcePinError(
                f"{where}: the segment {segment!a} of {field!a} is not a name "
                f"that matches {PATH_SEGMENT_GRAMMAR.pattern}"
            )
    file_name = field.rsplit(PATH_SEPARATOR, 1)[-1]
    refused_suffixes = find_refused_suffixes(file_name, REFUSED_SOURCE_SUFFIXES)
    if refused_suffixes:
        raise SourcePinError(
            f"{where}: {field!a} has the refused suffix {refused_suffixes[0]}, which marks "
            "code, an archive or a format that runs code when it loads"
        )
    return field


def parse_size(field: object, where: str) -> int:
    if (
        not isinstance(field, int)
        or isinstance(field, bool)
        or not MIN_SIZE_BYTES <= field <= MAX_SIZE_BYTES
    ):
        raise SourcePinError(
            f"{where}: {field!a} is not a whole number of bytes "
            f"from {MIN_SIZE_BYTES} to {MAX_SIZE_BYTES}"
        )
    return field


def refuse_overlapping_pins(pins: Sequence[SourcePin]) -> None:
    layouts: set[tuple[str, ...]] = set()
    for pin in pins:
        if pin.layout_segments in layouts:
            raise SourcePinError(f"{pin.path} of {pin.repository} is pinned twice")
        layouts.add(pin.layout_segments)
    for pin in pins:
        if any(
            pin.layout_segments[:length] in layouts for length in range(1, len(pin.layout_segments))
        ):
            raise SourcePinError(
                f"{pin.path} of {pin.repository} would lie under the file of another pin"
            )


def is_pinned_content(pin: SourcePin, size_bytes: int, sha256: str) -> bool:
    return size_bytes == pin.size_bytes and sha256 == pin.sha256
