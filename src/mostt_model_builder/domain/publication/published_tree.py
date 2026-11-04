import json
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
# to an entry, such as `lfs`, which the parser skips.
MAX_LISTING_DEPTH = 8
MAX_PATH_CHARACTERS = 512
MAX_LISTED_SIZE_BYTES = 2**40
LISTING_ENCODING = "utf-8"
FILE_ENTRY_TYPE = "file"
DIRECTORY_ENTRY_TYPE = "directory"
ENTRY_TYPES = frozenset({FILE_ENTRY_TYPE, DIRECTORY_ENTRY_TYPE})
# The Hub writes this file into every repository it creates, and it is none of the build's.
HUB_ATTRIBUTES_FILE_NAME = ".gitattributes"


class PublishedTreeError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class ListedEntry:
    entry_type: str
    path: str
    size_bytes: int


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
    return ListedEntry(str(entry_type), path, size_bytes)


def judge_published_listing(
    entries: Sequence[ListedEntry], expected_files: Sequence[ExpectedFile]
) -> None:
    expected_sizes = {expected.name: expected.size_bytes for expected in expected_files}
    listed_paths = [entry.path for entry in entries]
    if len(set(listed_paths)) != len(listed_paths):
        raise PublishedTreeError("the listing names one path twice")
    for entry in entries:
        if entry.entry_type == DIRECTORY_ENTRY_TYPE:
            raise PublishedTreeError(
                f"the revision holds the directory {entry.path}, which the build never publishes"
            )
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
