import json
import re
from collections.abc import Sequence
from dataclasses import dataclass

from mostt_model_builder.domain.json_documents.json_depth import (
    JsonDepthError,
    refuse_deep_json,
)
from mostt_model_builder.domain.sources.source_pins import (
    REPOSITORY_GRAMMAR,
    REVISION_GRAMMAR,
    RepositoryType,
    SourcePin,
    SourcePinError,
    parse_source_pin,
)

PUBLISHED_LISTING_MAX_SIZE_BYTES = 2**20
MAX_LISTING_ENTRY_COUNT = 1024
# A listing nests an entry's object in the top-level array, and the Hub adds objects of its own
# to an entry, such as `lfs`, whose `oid` alone the parser reads.
MAX_LISTING_DEPTH = 8
MAX_PATH_CHARACTERS = 512
MAX_LISTED_SIZE_BYTES = 2**40
LISTING_ENCODING = "utf-8"
FILE_ENTRY_TYPE = "file"
DIRECTORY_ENTRY_TYPE = "directory"
ENTRY_TYPES = frozenset({FILE_ENTRY_TYPE, DIRECTORY_ENTRY_TYPE})
# The Hub writes this file into every repository it creates, and it is none of the build's.
HUB_ATTRIBUTES_FILE_NAME = ".gitattributes"
BLOB_ID_GRAMMAR = re.compile(r"[0-9a-f]{40}")
LARGE_FILE_STORAGE_SHA256_GRAMMAR = re.compile(r"[0-9a-f]{64}")
# The Hub's own names: `oid` is a file's git blob id, and `lfs` holds the SHA-256 of a file in its
# large file storage as its own `oid`.
BLOB_ID_FIELD = "oid"
LARGE_FILE_STORAGE_FIELD = "lfs"


class PublishedTreeError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class ListedEntry:
    entry_type: str
    path: str
    size_bytes: int
    blob_id: str | None = None
    large_file_storage_sha256: str | None = None


@dataclass(frozen=True, slots=True)
class ExpectedFile:
    name: str
    size_bytes: int
    sha256: str


def judge_published_revision(repository: str, revision: str) -> None:
    if not REPOSITORY_GRAMMAR.fullmatch(repository):
        raise PublishedTreeError(
            f"the repository {repository!a} is not <namespace>/<name> under "
            f"{REPOSITORY_GRAMMAR.pattern}"
        )
    if not REVISION_GRAMMAR.fullmatch(revision):
        raise PublishedTreeError(
            f"the revision {revision!a} is not a commit of 40 hexadecimal digits"
        )


def parse_published_listing(content: bytes) -> tuple[ListedEntry, ...]:
    if len(content) > PUBLISHED_LISTING_MAX_SIZE_BYTES:
        raise PublishedTreeError(
            f"the listing holds more than {PUBLISHED_LISTING_MAX_SIZE_BYTES} bytes"
        )
    try:
        text = refuse_deep_json(content.decode(LISTING_ENCODING), MAX_LISTING_DEPTH)
        document = json.loads(text)
    except (UnicodeDecodeError, JsonDepthError, ValueError) as error:
        raise PublishedTreeError(f"the listing is not a JSON document: {error}") from error
    if not isinstance(document, list):
        raise PublishedTreeError("the listing is not an array")
    if len(document) > MAX_LISTING_ENTRY_COUNT:
        raise PublishedTreeError(f"the listing holds more than {MAX_LISTING_ENTRY_COUNT} entries")
    return tuple(parse_listed_entry(entry) for entry in document)


def parse_listed_entry(entry: object) -> ListedEntry:
    if not isinstance(entry, dict):
        raise PublishedTreeError("an entry of the listing is not an object")
    entry_type = entry.get("type")
    if entry_type not in ENTRY_TYPES:
        raise PublishedTreeError(f"an entry of the listing has the type {entry_type!a}")
    path = entry.get("path")
    if (
        not isinstance(path, str)
        or not path
        or len(path) > MAX_PATH_CHARACTERS
        or not path.isascii()
        or not path.isprintable()
    ):
        raise PublishedTreeError(f"an entry of the listing has the path {path!a}")
    size_bytes = entry.get("size")
    if (
        not isinstance(size_bytes, int)
        or isinstance(size_bytes, bool)
        or not 0 <= size_bytes <= MAX_LISTED_SIZE_BYTES
    ):
        raise PublishedTreeError(f"the entry {path} of the listing has the size {size_bytes!a}")
    if entry_type == DIRECTORY_ENTRY_TYPE:
        return ListedEntry(str(entry_type), path, size_bytes)
    blob_id = entry.get(BLOB_ID_FIELD)
    if not isinstance(blob_id, str) or not BLOB_ID_GRAMMAR.fullmatch(blob_id):
        raise PublishedTreeError(
            f"the entry {path} of the listing has the blob id {blob_id!a}, not 40 hexadecimal "
            "digits"
        )
    return ListedEntry(
        str(entry_type), path, size_bytes, blob_id, parse_large_file_storage(entry, path)
    )


def parse_large_file_storage(entry: dict[str, object], path: str) -> str | None:
    if LARGE_FILE_STORAGE_FIELD not in entry:
        return None
    large_file_storage = entry[LARGE_FILE_STORAGE_FIELD]
    if not isinstance(large_file_storage, dict):
        raise PublishedTreeError(
            f"the entry {path} of the listing has a large file storage that is not an object"
        )
    sha256 = large_file_storage.get(BLOB_ID_FIELD)
    if not isinstance(sha256, str) or not LARGE_FILE_STORAGE_SHA256_GRAMMAR.fullmatch(sha256):
        raise PublishedTreeError(
            f"the entry {path} of the listing has the large-file-storage SHA-256 {sha256!a}, "
            "not 64 hexadecimal digits"
        )
    return sha256


def refuse_listed_directories(entries: Sequence[ListedEntry]) -> None:
    for entry in entries:
        if entry.entry_type == DIRECTORY_ENTRY_TYPE:
            raise PublishedTreeError(
                f"the revision holds the directory {entry.path}, which the build never publishes"
            )


def judge_published_listing(
    entries: Sequence[ListedEntry], expected_files: Sequence[ExpectedFile]
) -> None:
    expected_sizes = {expected.name: expected.size_bytes for expected in expected_files}
    listed_paths = [entry.path for entry in entries]
    if len(set(listed_paths)) != len(listed_paths):
        raise PublishedTreeError("the listing names one path twice")
    refuse_listed_directories(entries)
    for entry in entries:
        if entry.path == HUB_ATTRIBUTES_FILE_NAME:
            continue
        if entry.path not in expected_sizes:
            raise PublishedTreeError(
                f"the revision holds {entry.path}, which is not a file of the build"
            )
        if entry.size_bytes != expected_sizes[entry.path]:
            raise PublishedTreeError(
                f"the revision holds {entry.size_bytes} bytes of {entry.path}, where the build "
                f"holds {expected_sizes[entry.path]}"
            )
    missing = sorted(set(expected_sizes) - set(listed_paths))
    if missing:
        raise PublishedTreeError(f"the revision lacks {missing[0]}, a file of the build")


def pin_published_file(repository: str, revision: str, expected: ExpectedFile) -> SourcePin:
    try:
        return parse_source_pin(
            {
                "repository_type": RepositoryType.MODEL.value,
                "repository": repository,
                "revision": revision,
                "path": expected.name,
                "size_bytes": expected.size_bytes,
                "sha256": expected.sha256,
            },
            f"the published file {expected.name}",
        )
    except SourcePinError as error:
        raise PublishedTreeError(str(error)) from error
