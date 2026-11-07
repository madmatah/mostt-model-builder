import json
import re
from collections.abc import Sequence
from dataclasses import dataclass

from mostt_model_builder.domain.bench.fleurs_index import BENCH_LANGUAGES
from mostt_model_builder.domain.containers.devices import Device
from mostt_model_builder.domain.json_documents.json_depth import MAX_JSON_DEPTH, refuse_deep_json

SHARD_REQUEST_FILE_NAME = "shard.json"
SHARD_REQUEST_MAX_SIZE_BYTES = 4096
RUN_NAME_GRAMMAR = re.compile(r"[a-z0-9][a-z0-9-]{0,31}")
SHARD_REQUEST_KEYS = frozenset({"run", "variant", "device", "languages", "shard", "shard_count"})
MIN_SHARD_COUNT = 1
MAX_SHARD_COUNT = 16


class ShardRequestError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class ShardRequest:
    run: str
    variant: str
    device: Device
    languages: tuple[str, ...]
    shard: int
    shard_count: int


def render_shard_request(request: ShardRequest) -> bytes:
    document = {
        "run": request.run,
        "variant": request.variant,
        "device": request.device.value,
        "languages": list(request.languages),
        "shard": request.shard,
        "shard_count": request.shard_count,
    }
    return json.dumps(document, indent=1).encode("ascii")


def parse_shard_request(content: bytes) -> ShardRequest:
    if len(content) > SHARD_REQUEST_MAX_SIZE_BYTES:
        raise ShardRequestError(f"the shard's request holds {len(content)} bytes, too many")
    try:
        document = json.loads(
            refuse_deep_json(content.decode("ascii"), MAX_JSON_DEPTH),
            object_pairs_hook=refuse_duplicate_keys,
        )
    except (UnicodeDecodeError, ValueError) as error:
        raise ShardRequestError(f"the shard's request is not ASCII JSON: {error}") from error
    if not isinstance(document, dict) or set(document) != SHARD_REQUEST_KEYS:
        raise ShardRequestError(
            f"the shard's request is not an object of {sorted(SHARD_REQUEST_KEYS)}"
        )
    shard_count = require_integer(
        document["shard_count"], MIN_SHARD_COUNT, MAX_SHARD_COUNT, "its count of shards"
    )
    return ShardRequest(
        run=require_name(document["run"], "its run"),
        variant=require_name(document["variant"], "its variant"),
        device=require_device(document["device"]),
        languages=require_languages(document["languages"]),
        shard=require_integer(document["shard"], 0, shard_count - 1, "its shard"),
        shard_count=shard_count,
    )


def require_name(field: object, what: str) -> str:
    if not isinstance(field, str) or not RUN_NAME_GRAMMAR.fullmatch(field):
        raise ShardRequestError(f"{what}, {field!a}, does not match {RUN_NAME_GRAMMAR.pattern}")
    return field


def require_device(field: object) -> Device:
    if not isinstance(field, str) or field not in {device.value for device in Device}:
        raise ShardRequestError(f"its device, {field!a}, is neither cpu nor gpu")
    return Device(field)


def require_languages(field: object) -> tuple[str, ...]:
    if (
        not isinstance(field, list)
        or not field
        or not all(language in BENCH_LANGUAGES for language in field)
        or len(set(field)) != len(field)
    ):
        raise ShardRequestError(
            f"its languages, {field!a}, are not distinct languages of the bench"
        )
    return tuple(str(language) for language in field)


def require_integer(field: object, minimum: int, maximum: int, what: str) -> int:
    if isinstance(field, bool) or not isinstance(field, int) or not minimum <= field <= maximum:
        raise ShardRequestError(f"{what}, {field!a}, lies outside {minimum} to {maximum}")
    return field


def select_shard[Entry](entries: Sequence[Entry], shard: int, shard_count: int) -> list[Entry]:
    return list(entries[shard::shard_count])


def refuse_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    keys = [key for key, _ in pairs]
    if len(set(keys)) != len(keys):
        raise ValueError("a key appears twice in an object")
    return dict(pairs)
