import json
import re
from dataclasses import dataclass
from pathlib import PurePosixPath

from mostt_model_builder.domain.bench.fleurs_index import LANGUAGE_GRAMMAR
from mostt_model_builder.domain.json_documents.json_depth import MAX_JSON_DEPTH, refuse_deep_json

FLEURS_CHECK_REQUEST_FILE_NAME = "request.json"
FLEURS_CHECK_REQUEST_MAX_SIZE_BYTES = 64 * 1024
REQUEST_KEYS = frozenset({"languages"})
LANGUAGE_KEYS = frozenset({"language", "index", "archive"})
CONTAINER_PATH_GRAMMAR = re.compile(r"/sources(/[A-Za-z0-9._-]+)+")
DOT_SEGMENTS = frozenset({".", ".."})


class FleursCheckRequestError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class BenchFiles:
    language: str
    index: PurePosixPath
    archive: PurePosixPath


@dataclass(frozen=True, slots=True)
class FleursCheckRequest:
    languages: tuple[BenchFiles, ...]


def render_fleurs_check_request(request: FleursCheckRequest) -> bytes:
    document = {
        "languages": [
            {"language": files.language, "index": str(files.index), "archive": str(files.archive)}
            for files in request.languages
        ]
    }
    return json.dumps(document, indent=1).encode("ascii")


def parse_fleurs_check_request(content: bytes) -> FleursCheckRequest:
    if len(content) > FLEURS_CHECK_REQUEST_MAX_SIZE_BYTES:
        raise FleursCheckRequestError(f"the request holds {len(content)} bytes, too many")
    try:
        document = json.loads(refuse_deep_json(content.decode("utf-8"), MAX_JSON_DEPTH))
    except (UnicodeDecodeError, ValueError) as error:
        raise FleursCheckRequestError(f"the request is not JSON: {error}") from error
    if not isinstance(document, dict) or set(document) != REQUEST_KEYS:
        raise FleursCheckRequestError(f"the request is not an object of {sorted(REQUEST_KEYS)}")
    entries = document["languages"]
    if not isinstance(entries, list) or not entries:
        raise FleursCheckRequestError("the request names no language")
    languages = tuple(parse_bench_files(entry) for entry in entries)
    names = [files.language for files in languages]
    if len(set(names)) != len(names):
        raise FleursCheckRequestError("the request names a language twice")
    return FleursCheckRequest(languages)


def parse_bench_files(entry: object) -> BenchFiles:
    if not isinstance(entry, dict) or set(entry) != LANGUAGE_KEYS:
        raise FleursCheckRequestError(
            f"a language of the request is not an object of {sorted(LANGUAGE_KEYS)}"
        )
    language = entry["language"]
    if not isinstance(language, str) or not LANGUAGE_GRAMMAR.fullmatch(language):
        raise FleursCheckRequestError(
            f"{language!a} does not name a language as FLEURS does, {LANGUAGE_GRAMMAR.pattern}"
        )
    return BenchFiles(
        language=language,
        index=parse_container_path(entry["index"]),
        archive=parse_container_path(entry["archive"]),
    )


def parse_container_path(field: object) -> PurePosixPath:
    if (
        not isinstance(field, str)
        or not CONTAINER_PATH_GRAMMAR.fullmatch(field)
        or any(segment in DOT_SEGMENTS for segment in field.split("/"))
    ):
        raise FleursCheckRequestError(f"{field!a} is not a path under /sources")
    return PurePosixPath(field)
