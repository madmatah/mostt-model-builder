import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from mostt_model_builder.domain.cache.output_keys import PlainData, compute_key
from mostt_model_builder.domain.containers.container_files import (
    SINGLE_LINK_COUNT,
    FileFacts,
    FileKind,
)
from mostt_model_builder.domain.formats.file_names import is_file_name
from mostt_model_builder.domain.json_documents.json_depth import MAX_JSON_DEPTH, refuse_deep_json
from mostt_model_builder.domain.sources.source_pins import SHA256_GRAMMAR

RECORD_FILE_NAME = "record.json"
RECORD_FILE_MODE = 0o444
RECORD_MAX_SIZE_BYTES = 4 * 2**20
RECORD_ENCODING = "ascii"
KEY_GRAMMAR = re.compile(r"[0-9a-f]{64}")
ENTRY_NAME_GRAMMAR = re.compile(r"[a-z0-9][a-z0-9_-]{0,81}")
RECORD_KEYS = frozenset({"key", "row", "document", "outputs"})
OUTPUT_KEYS = frozenset({"name", "size_bytes", "sha256"})
MAX_RECORDED_OUTPUT_COUNT = 256
# A key shows by its first digits in the lines a person reads; the record and the directory keep
# the whole of it.
SHOWN_KEY_CHARACTERS = 12


class OutputRecordError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class RecordedOutput:
    name: str
    size_bytes: int
    sha256: str


@dataclass(frozen=True, slots=True)
class OutputRecord:
    key: str
    row: str
    document: Mapping[str, PlainData]
    outputs: tuple[RecordedOutput, ...]

    @property
    def digests(self) -> dict[str, str]:
        return {output.name: output.sha256 for output in self.outputs}


def show_key(key: str) -> str:
    return key[:SHOWN_KEY_CHARACTERS]


def is_key(name: str) -> bool:
    return KEY_GRAMMAR.fullmatch(name) is not None


def render_output_record(record: OutputRecord) -> bytes:
    document = {
        "key": record.key,
        "row": record.row,
        "document": dict(record.document),
        "outputs": [
            {"name": output.name, "size_bytes": output.size_bytes, "sha256": output.sha256}
            for output in record.outputs
        ],
    }
    return (json.dumps(document, sort_keys=True, ensure_ascii=True, indent=1) + "\n").encode(
        RECORD_ENCODING
    )


# The host writes every record itself, so a malformed one is a bug or a hand's edit, and fails
# loudly rather than being read past.
def parse_output_record(content: bytes) -> OutputRecord:
    if len(content) > RECORD_MAX_SIZE_BYTES:
        raise OutputRecordError(f"the record holds {len(content)} bytes, beyond its bound")
    try:
        document = json.loads(refuse_deep_json(content.decode(RECORD_ENCODING), MAX_JSON_DEPTH))
    except (UnicodeDecodeError, ValueError) as error:
        raise OutputRecordError(f"the record is not ASCII JSON: {error}") from error
    if not isinstance(document, dict) or set(document) != RECORD_KEYS:
        raise OutputRecordError(f"the record does not hold exactly {sorted(RECORD_KEYS)}")
    key = document["key"]
    if not isinstance(key, str) or not is_key(key):
        raise OutputRecordError(f"the record's key is not under {KEY_GRAMMAR.pattern}")
    row = document["row"]
    if not isinstance(row, str) or not ENTRY_NAME_GRAMMAR.fullmatch(row):
        raise OutputRecordError(f"the record's row is not under {ENTRY_NAME_GRAMMAR.pattern}")
    key_document = document["document"]
    if not isinstance(key_document, dict):
        raise OutputRecordError("the record's key document is not an object")
    return OutputRecord(key, row, key_document, parse_recorded_outputs(document["outputs"]))


def parse_recorded_outputs(field: object) -> tuple[RecordedOutput, ...]:
    if not isinstance(field, list) or len(field) > MAX_RECORDED_OUTPUT_COUNT:
        raise OutputRecordError(
            f"the record's outputs are not a list of at most {MAX_RECORDED_OUTPUT_COUNT}"
        )
    outputs = tuple(parse_recorded_output(entry) for entry in field)
    names = [output.name for output in outputs]
    if len(set(names)) != len(names):
        raise OutputRecordError("the record names an output twice")
    return outputs


def parse_recorded_output(entry: object) -> RecordedOutput:
    if not isinstance(entry, dict) or set(entry) != OUTPUT_KEYS:
        raise OutputRecordError(
            f"an output of the record does not hold exactly {sorted(OUTPUT_KEYS)}"
        )
    name = entry["name"]
    if not isinstance(name, str) or not is_file_name(name):
        raise OutputRecordError("an output of the record is not named by a file name")
    size_bytes = entry["size_bytes"]
    if not isinstance(size_bytes, int) or isinstance(size_bytes, bool) or size_bytes < 0:
        raise OutputRecordError(f"the output {name} of the record has no size in bytes")
    sha256 = entry["sha256"]
    if not isinstance(sha256, str) or not SHA256_GRAMMAR.fullmatch(sha256):
        raise OutputRecordError(f"the output {name} of the record has no SHA-256")
    return RecordedOutput(name, size_bytes, sha256)


def judge_record_file(facts: FileFacts, user_id: int) -> None:
    if (
        facts.kind is not FileKind.REGULAR
        or facts.owner_id != user_id
        or facts.permission_bits != RECORD_FILE_MODE
        or facts.link_count != SINGLE_LINK_COUNT
        or facts.size_bytes > RECORD_MAX_SIZE_BYTES
    ):
        raise OutputRecordError(
            f"the record is not the host's own: a regular file of the user's, in mode "
            f"{RECORD_FILE_MODE:04o}, with a single link, of at most {RECORD_MAX_SIZE_BYTES} bytes"
        )


def judge_record_identity(record: OutputRecord, key: str, row: str) -> None:
    if record.row != row:
        raise OutputRecordError(f"the record names the row {record.row}, not {row}")
    if record.key != key:
        raise OutputRecordError(f"the record holds the key {record.key}, not {key}")
    if compute_key(record.document) != key:
        raise OutputRecordError("the record's key document does not give its key")


def judge_recorded_outputs(record: OutputRecord, outputs: Sequence[RecordedOutput]) -> None:
    recorded = {output.name: output for output in record.outputs}
    found = {output.name: output for output in outputs}
    if set(recorded) != set(found):
        raise OutputRecordError(
            f"the record names the outputs {sorted(recorded)}, and the entry holds {sorted(found)}"
        )
    for name, output in found.items():
        if output != recorded[name]:
            raise OutputRecordError(
                f"the output {name} holds {output.size_bytes} bytes of SHA-256 {output.sha256}, "
                f"and its record says {recorded[name].size_bytes} bytes of SHA-256 "
                f"{recorded[name].sha256}"
            )
