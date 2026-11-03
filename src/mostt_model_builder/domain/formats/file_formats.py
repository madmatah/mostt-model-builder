import codecs
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum, IntEnum, StrEnum, auto
from typing import Protocol, assert_never


class FileFormat(StrEnum):
    ONNX = "onnx"
    TEXT = "text"
    ONNX_EXTERNAL_DATA = "onnx-external-data"


class FormatWithoutStructuralCheckError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class FormatMismatch:
    reason: str


class FormatCheck(Protocol):
    def feed(self, chunk: bytes) -> None: ...

    def finish(self) -> FormatMismatch | None: ...


def has_structural_check(file_format: FileFormat) -> bool:
    match file_format:
        case FileFormat.ONNX | FileFormat.TEXT:
            return True
        case FileFormat.ONNX_EXTERNAL_DATA:
            return False
        case _:
            assert_never(file_format)


def start_format_check(file_format: FileFormat) -> FormatCheck:
    match file_format:
        case FileFormat.ONNX:
            return OnnxModelCheck()
        case FileFormat.TEXT:
            return TextCheck()
        case FileFormat.ONNX_EXTERNAL_DATA:
            raise FormatWithoutStructuralCheckError(
                f"the format {file_format} has no structural check of its own: it is checked "
                "against the inspection's report of its onnx file"
            )
        case _:
            assert_never(file_format)


class WireType(IntEnum):
    VARINT = 0
    FIXED64 = 1
    LENGTH_DELIMITED = 2
    START_GROUP = 3
    END_GROUP = 4
    FIXED32 = 5


@dataclass(frozen=True, slots=True)
class OnnxField:
    name: str
    wire_type: WireType
    is_repeatable: bool


IR_VERSION_FIELD = 1
GRAPH_FIELD = 7
# The fields of the message ModelProto in onnx.proto
# (https://github.com/onnx/onnx/blob/main/onnx/onnx.proto), each with the one wire type
# that its declared type takes in the Protocol Buffers encoding
# (https://protobuf.dev/programming-guides/encoding/): int64 as a varint, a string or a
# message as a length-delimited record. A field that ONNX adds later is refused until it
# joins this table.
ONNX_MODEL_FIELDS: Mapping[int, OnnxField] = {
    IR_VERSION_FIELD: OnnxField("ir_version", WireType.VARINT, is_repeatable=False),
    2: OnnxField("producer_name", WireType.LENGTH_DELIMITED, is_repeatable=False),
    3: OnnxField("producer_version", WireType.LENGTH_DELIMITED, is_repeatable=False),
    4: OnnxField("domain", WireType.LENGTH_DELIMITED, is_repeatable=False),
    5: OnnxField("model_version", WireType.VARINT, is_repeatable=False),
    6: OnnxField("doc_string", WireType.LENGTH_DELIMITED, is_repeatable=False),
    GRAPH_FIELD: OnnxField("graph", WireType.LENGTH_DELIMITED, is_repeatable=False),
    8: OnnxField("opset_import", WireType.LENGTH_DELIMITED, is_repeatable=True),
    14: OnnxField("metadata_props", WireType.LENGTH_DELIMITED, is_repeatable=True),
    20: OnnxField("training_info", WireType.LENGTH_DELIMITED, is_repeatable=True),
    25: OnnxField("functions", WireType.LENGTH_DELIMITED, is_repeatable=True),
    26: OnnxField("configuration", WireType.LENGTH_DELIMITED, is_repeatable=True),
}
LOWEST_IR_VERSION = 1
HIGHEST_IR_VERSION = 64
MAX_KEY_BYTES = 5
MAX_VARINT_BYTES = 10
# The tenth byte of a varint carries the 64th bit only, so any larger value overflows.
MAX_LAST_VARINT_BYTE = 1
CONTINUATION_BIT = 0x80
PAYLOAD_BITS = 0x7F
PAYLOAD_BIT_COUNT = 7
WIRE_TYPE_BIT_COUNT = 3
WIRE_TYPE_BITS = 0b111


class OnnxReading(Enum):
    KEY = auto()
    VARINT = auto()
    LENGTH = auto()
    CONTENT = auto()


# A model streams by in gigabytes, more than the host may hold in its memory.
class OnnxModelCheck:
    def __init__(self) -> None:
        self._mismatch: FormatMismatch | None = None
        self._reading = OnnxReading.KEY
        self._number = 0
        self._number_bytes = 0
        self._field_number = 0
        self._content_left_bytes = 0
        self._seen_fields: set[int] = set()

    def feed(self, chunk: bytes) -> None:
        position = 0
        while position < len(chunk) and self._mismatch is None:
            if self._reading is OnnxReading.CONTENT:
                skipped_bytes = min(self._content_left_bytes, len(chunk) - position)
                position += skipped_bytes
                self._content_left_bytes -= skipped_bytes
                if self._content_left_bytes == 0:
                    self._reading = OnnxReading.KEY
                continue
            self._take_byte(chunk[position])
            position += 1

    def finish(self) -> FormatMismatch | None:
        if self._mismatch is not None:
            return self._mismatch
        if self._reading is not OnnxReading.KEY:
            return FormatMismatch(f"the file ends inside {self._describe_reading_place()}")
        if self._number_bytes > 0:
            return FormatMismatch("the file ends inside a key")
        if IR_VERSION_FIELD not in self._seen_fields:
            return FormatMismatch("the model has no ir_version (field 1)")
        if GRAPH_FIELD not in self._seen_fields:
            return FormatMismatch("the model has no graph (field 7)")
        return None

    def _describe_reading_place(self) -> str:
        field = f"field {self._field_number} ({ONNX_MODEL_FIELDS[self._field_number].name})"
        match self._reading:
            case OnnxReading.VARINT:
                return f"the varint of {field}"
            case OnnxReading.LENGTH:
                return f"the length of {field}"
            case OnnxReading.CONTENT:
                return f"the content of {field}"
            case OnnxReading.KEY:
                return "a key"
            case _:
                assert_never(self._reading)

    def _take_byte(self, byte: int) -> None:
        self._number |= (byte & PAYLOAD_BITS) << (PAYLOAD_BIT_COUNT * self._number_bytes)
        self._number_bytes += 1
        has_more_bytes = byte & CONTINUATION_BIT != 0
        if self._reading is OnnxReading.KEY:
            self._take_key_byte(byte, has_more_bytes)
        elif self._number_bytes == MAX_VARINT_BYTES and byte > MAX_LAST_VARINT_BYTE:
            self._mismatch = FormatMismatch("a varint is longer than 64 bits")
        elif not has_more_bytes:
            self._end_number()

    def _take_key_byte(self, byte: int, has_more_bytes: bool) -> None:
        if has_more_bytes:
            if self._number_bytes == MAX_KEY_BYTES:
                self._mismatch = FormatMismatch(f"a key is longer than {MAX_KEY_BYTES} bytes")
            return
        if byte == 0 and self._number_bytes > 1:
            self._mismatch = FormatMismatch("a key is not in its shortest encoding")
            return
        self._end_number()

    def _end_number(self) -> None:
        number = self._number
        self._number = 0
        self._number_bytes = 0
        match self._reading:
            case OnnxReading.KEY:
                self._start_field(number >> WIRE_TYPE_BIT_COUNT, number & WIRE_TYPE_BITS)
            case OnnxReading.VARINT:
                self._end_varint(number)
            case OnnxReading.LENGTH:
                self._content_left_bytes = number
                self._reading = OnnxReading.CONTENT if number > 0 else OnnxReading.KEY
            case OnnxReading.CONTENT:
                raise AssertionError("the content of a field holds no number")
            case _:
                assert_never(self._reading)

    def _start_field(self, field_number: int, wire_type: int) -> None:
        field = ONNX_MODEL_FIELDS.get(field_number)
        if field is None:
            self._mismatch = FormatMismatch(f"field {field_number} is not a field of an ONNX model")
            return
        if wire_type != field.wire_type:
            self._mismatch = FormatMismatch(
                f"field {field_number} ({field.name}) is encoded with wire type {wire_type}, "
                f"not {field.wire_type.value}"
            )
            return
        if field_number in self._seen_fields and not field.is_repeatable:
            self._mismatch = FormatMismatch(
                f"field {field_number} ({field.name}) appears more than once"
            )
            return
        self._seen_fields.add(field_number)
        self._field_number = field_number
        self._reading = (
            OnnxReading.VARINT if field.wire_type is WireType.VARINT else OnnxReading.LENGTH
        )

    def _end_varint(self, number: int) -> None:
        self._reading = OnnxReading.KEY
        if self._field_number == IR_VERSION_FIELD and not (
            LOWEST_IR_VERSION <= number <= HIGHEST_IR_VERSION
        ):
            self._mismatch = FormatMismatch(
                f"ir_version {number} is outside {LOWEST_IR_VERSION} to {HIGHEST_IR_VERSION}"
            )


NULL_CHARACTER = "\x00"
NULL_BYTE = b"\x00"


class TextCheck:
    def __init__(self) -> None:
        self._mismatch: FormatMismatch | None = None
        self._decoder = codecs.getincrementaldecoder("utf-8")("strict")

    def feed(self, chunk: bytes) -> None:
        if self._mismatch is not None:
            return
        try:
            decoded = self._decoder.decode(chunk)
        except UnicodeDecodeError as error:
            valid_prefix = error.object[: error.start]
            self._mismatch = (
                FormatMismatch("the text holds a NUL character")
                if NULL_BYTE in valid_prefix
                else FormatMismatch("the text is not valid UTF-8")
            )
            return
        if NULL_CHARACTER in decoded:
            self._mismatch = FormatMismatch("the text holds a NUL character")

    def finish(self) -> FormatMismatch | None:
        if self._mismatch is not None:
            return self._mismatch
        try:
            self._decoder.decode(b"", final=True)
        except UnicodeDecodeError:
            return FormatMismatch("the text ends inside a UTF-8 sequence")
        return None
