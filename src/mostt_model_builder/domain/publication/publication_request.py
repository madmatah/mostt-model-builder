import json
from collections.abc import Mapping
from dataclasses import dataclass

from mostt_model_builder.domain.formats.file_names import is_file_name
from mostt_model_builder.domain.json_documents.json_depth import MAX_JSON_DEPTH, refuse_deep_json
from mostt_model_builder.domain.sources.source_pins import REPOSITORY_GRAMMAR, SHA256_GRAMMAR

MODEL_CARD_FILE_NAME = "README.md"
PUBLICATION_REQUEST_FILE_NAME = "publication-request.json"
PUBLICATION_REQUEST_MAX_SIZE_BYTES = 64 * 1024
REQUEST_ENCODING = "ascii"
MAX_COMMIT_MESSAGE_CHARACTERS = 200
MAX_PUBLISHED_FILE_COUNT = 256
MAX_PUBLISHED_SIZE_BYTES = 2**36
REQUEST_KEYS = frozenset({"repository", "commit_message", "candidate", "card"})
FILE_KEYS = frozenset({"name", "size_bytes", "sha256"})


class PublicationRequestError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class PublishedFile:
    name: str
    size_bytes: int
    sha256: str


@dataclass(frozen=True, slots=True)
class PublicationRequest:
    repository: str
    commit_message: str
    candidate: tuple[PublishedFile, ...]
    card: PublishedFile

    @property
    def files(self) -> tuple[PublishedFile, ...]:
        return (*self.candidate, self.card)


def describe_commit_message(card_title: str) -> str:
    return f"Publish {card_title}"


def render_published_file(published_file: PublishedFile) -> dict[str, object]:
    return {
        "name": published_file.name,
        "size_bytes": published_file.size_bytes,
        "sha256": published_file.sha256,
    }


def render_publication_request(request: PublicationRequest) -> bytes:
    document = {
        "repository": request.repository,
        "commit_message": request.commit_message,
        "candidate": [render_published_file(published) for published in request.candidate],
        "card": render_published_file(request.card),
    }
    return (json.dumps(document, indent=2, ensure_ascii=True) + "\n").encode(REQUEST_ENCODING)


def parse_publication_request(content: bytes) -> PublicationRequest:
    document = decode_document(content, PUBLICATION_REQUEST_MAX_SIZE_BYTES, "the request")
    fields = require_object(document, REQUEST_KEYS, "the request")
    entries = fields["candidate"]
    if not isinstance(entries, list) or not 1 <= len(entries) <= MAX_PUBLISHED_FILE_COUNT:
        raise PublicationRequestError(
            f"the request's candidate is not a list of 1 to {MAX_PUBLISHED_FILE_COUNT} files"
        )
    request = PublicationRequest(
        repository=parse_repository(fields["repository"], "the request"),
        commit_message=parse_commit_message(fields["commit_message"]),
        candidate=tuple(
            parse_published_file(entry, "the request's candidate") for entry in entries
        ),
        card=parse_published_file(fields["card"], "the request's card"),
    )
    if request.card.name != MODEL_CARD_FILE_NAME:
        raise PublicationRequestError(f"the request's card is not named {MODEL_CARD_FILE_NAME}")
    names = [published.name for published in request.files]
    if len(set(names)) != len(names):
        raise PublicationRequestError("the request names one file twice")
    return request


def decode_document(content: bytes, max_size_bytes: int, what: str) -> object:
    if len(content) > max_size_bytes:
        raise PublicationRequestError(f"{what} holds more than {max_size_bytes} bytes")
    try:
        return json.loads(
            refuse_deep_json(content.decode(REQUEST_ENCODING), MAX_JSON_DEPTH),
            object_pairs_hook=refuse_duplicate_keys,
        )
    except (UnicodeDecodeError, ValueError) as error:
        raise PublicationRequestError(f"{what} is not ASCII JSON: {error}") from error


def refuse_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    keys = [key for key, _ in pairs]
    if len(set(keys)) != len(keys):
        raise ValueError("a key appears twice in an object")
    return dict(pairs)


def require_object(document: object, keys: frozenset[str], what: str) -> Mapping[str, object]:
    if not isinstance(document, dict) or set(document) != keys:
        raise PublicationRequestError(f"{what} is not an object of exactly {sorted(keys)}")
    return document


def parse_repository(field: object, what: str) -> str:
    if not isinstance(field, str) or not REPOSITORY_GRAMMAR.fullmatch(field):
        raise PublicationRequestError(f"{what} names the repository {field!a}")
    return field


def parse_commit_message(field: object) -> str:
    if (
        not isinstance(field, str)
        or not field
        or len(field) > MAX_COMMIT_MESSAGE_CHARACTERS
        or not field.isprintable()
    ):
        raise PublicationRequestError(f"the request's commit message is {field!a}")
    return field


def parse_published_file(entry: object, what: str) -> PublishedFile:
    fields = require_object(entry, FILE_KEYS, f"a file of {what}")
    name = fields["name"]
    if not isinstance(name, str) or not is_file_name(name):
        raise PublicationRequestError(f"a file of {what} is named {name!a}")
    size_bytes = fields["size_bytes"]
    if (
        not isinstance(size_bytes, int)
        or isinstance(size_bytes, bool)
        or not 1 <= size_bytes <= MAX_PUBLISHED_SIZE_BYTES
    ):
        raise PublicationRequestError(f"the file {name} of {what} holds {size_bytes!a} bytes")
    sha256 = fields["sha256"]
    if not isinstance(sha256, str) or not SHA256_GRAMMAR.fullmatch(sha256):
        raise PublicationRequestError(f"the file {name} of {what} has the SHA-256 {sha256!a}")
    return PublishedFile(name, size_bytes, sha256)
