import random
import re
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass

import pytest

from mostt_model_builder.domain.formats.file_formats import (
    GRAPH_FIELD,
    IR_VERSION_FIELD,
    ONNX_MODEL_FIELDS,
    FileFormat,
    FormatCheck,
    FormatMismatch,
    FormatWithoutStructuralCheckError,
    WireType,
    has_structural_check,
    start_format_check,
)

BYTE_COUNT = 256
VARINT_CONTINUATION_BIT = 0x80
VARINT_PAYLOAD_BITS = 0x7F
VARINT_PAYLOAD_BIT_COUNT = 7
KEY_WIRE_TYPE_BIT_COUNT = 3
FIXED64_BYTES = 8
FIXED32_BYTES = 4
HIGHEST_FIELD_ENUMERATED = 63
HIGHEST_WIRE_TYPE_ENUMERATED = 7
ENUMERATED_KEY_COUNT = (HIGHEST_FIELD_ENUMERATED + 1) * (HIGHEST_WIRE_TYPE_ENUMERATED + 1)
MODEL_VERSION_FIELD = 5
DOC_STRING_FIELD = 6
OPSET_IMPORT_FIELD = 8
METADATA_PROPS_FIELD = 14
TRAINING_INFO_FIELD = 20
FUNCTIONS_FIELD = 25
CONFIGURATION_FIELD = 26
FIELDS_OUTSIDE_THE_TABLE = (0, 9, 13, 15, 27)
LOWEST_ASCII_BYTE_AFTER_NULL = 0x01
HIGHEST_ASCII_BYTE = 0x7F
FIRST_BYTE_NEVER_IN_UTF8 = 0xF5
STRUCTURALLY_CHECKED_FORMATS = {
    FileFormat.ONNX: True,
    FileFormat.TEXT: True,
    FileFormat.ONNX_EXTERNAL_DATA: False,
}
EXTERNAL_DATA_REFUSAL = (
    "the format onnx-external-data has no structural check of its own: it is checked "
    "against the inspection's report of its onnx file"
)
PROPERTY_SEED = 20261103
SAMPLED_CHANGE_COUNT = 4096


@dataclass(frozen=True, slots=True)
class AcceptedFile:
    name: str
    content: bytes


@dataclass(frozen=True, slots=True)
class RefusedFile:
    name: str
    content: bytes
    reason: str


@dataclass(frozen=True, slots=True)
class Judgment[Case]:
    misjudged: list[Case]
    judged_count: int


def encode_varint(number: int) -> bytes:
    encoded = bytearray()
    while True:
        low_bits = number & VARINT_PAYLOAD_BITS
        number >>= VARINT_PAYLOAD_BIT_COUNT
        if number == 0:
            encoded.append(low_bits)
            return bytes(encoded)
        encoded.append(low_bits | VARINT_CONTINUATION_BIT)


def encode_key(field_number: int, wire_type: int) -> bytes:
    return encode_varint(field_number << KEY_WIRE_TYPE_BIT_COUNT | wire_type)


def build_smallest_content(wire_type: int) -> bytes:
    match wire_type:
        case WireType.VARINT | WireType.LENGTH_DELIMITED:
            return b"\x00"
        case WireType.FIXED64:
            return bytes(FIXED64_BYTES)
        case WireType.FIXED32:
            return bytes(FIXED32_BYTES)
        case _:
            return b""


def encode_field_record(field_number: int, wire_type: int) -> bytes:
    return encode_key(field_number, wire_type) + build_smallest_content(wire_type)


def encode_string_record(field_number: int, text: bytes) -> bytes:
    return encode_key(field_number, WireType.LENGTH_DELIMITED) + encode_varint(len(text)) + text


def judge_chunks(file_format: FileFormat, chunks: Iterable[bytes]) -> FormatMismatch | None:
    format_check = start_format_check(file_format)
    for chunk in chunks:
        format_check.feed(chunk)
    return format_check.finish()


def judge_whole_file(file_format: FileFormat, content: bytes) -> FormatMismatch | None:
    return judge_chunks(file_format, [content])


def split_anywhere(content: bytes) -> list[list[bytes]]:
    two_pieces = [[content[:cut], content[cut:]] for cut in range(len(content) + 1)]
    byte_by_byte = [[content[index : index + 1] for index in range(len(content))]]
    return two_pieces + byte_by_byte


def is_admitted_after_the_smallest_model(field_number: int, wire_type: int) -> bool:
    field = ONNX_MODEL_FIELDS.get(field_number)
    if field is None or field.wire_type != wire_type:
        return False
    return field.is_repeatable or field_number not in (IR_VERSION_FIELD, GRAPH_FIELD)


def starts_a_check(file_format: FileFormat) -> bool:
    try:
        start_format_check(file_format)
    except FormatWithoutStructuralCheckError:
        return False
    return True


def change_one_byte(model: bytes, generator: random.Random) -> bytes:
    position = generator.randrange(len(model))
    changed_byte = (model[position] + generator.randrange(1, BYTE_COUNT)) % BYTE_COUNT
    return model[:position] + bytes([changed_byte]) + model[position + 1 :]


def build_damaged_models(model: bytes, generator: random.Random) -> list[bytes]:
    truncations = [model[:length] for length in range(len(model))]
    changes = [change_one_byte(model, generator) for _ in range(SAMPLED_CHANGE_COUNT)]
    return truncations + changes


def cut_at_random(content: bytes, generator: random.Random) -> list[bytes]:
    cut = generator.randrange(len(content) + 1)
    return [content[:cut], content[cut:]]


def describe_errors_other_than_a_mismatch(content: bytes) -> list[str]:
    try:
        judge_whole_file(FileFormat.ONNX, content)
    except Exception as error:
        return [f"{content.hex()}: {error!r}"]
    return []


def build_accepted_parameters(files: Sequence[AcceptedFile]) -> list[object]:
    return [pytest.param(file.content, id=file.name) for file in files]


def build_refused_parameters(files: Sequence[RefusedFile]) -> list[object]:
    return [pytest.param(file.content, file.reason, id=file.name) for file in files]


IR_VERSION_7 = encode_key(IR_VERSION_FIELD, WireType.VARINT) + encode_varint(7)
EMPTY_GRAPH = encode_key(GRAPH_FIELD, WireType.LENGTH_DELIMITED) + encode_varint(0)
SMALLEST_MODEL = IR_VERSION_7 + EMPTY_GRAPH
MINIMAL_MODEL = (
    b"\x08\x0a"
    b"\x12\x07pytorch"
    b"\x1a\x052.5.0"
    b"\x3a\x28"
    b"\x0a\x10\x0a\x01x\x12\x01y\x22\x08Identity"
    b"\x12\x0amain_graph"
    b"\x5a\x03\x0a\x01x"
    b"\x62\x03\x0a\x01y"
    b"\x42\x04\x0a\x00\x10\x11"
)
TEN_BYTE_VARINT_ENDING_IN_1 = b"\x80" * 9 + b"\x01"
TEN_BYTE_VARINT_ENDING_IN_2 = b"\x80" * 9 + b"\x02"
ELEVEN_BYTE_VARINT = b"\x80" * 10 + b"\x01"
NON_MINIMAL_KEY_OF_IR_VERSION = b"\x88\x00"
SIX_BYTE_KEY = b"\x80\x80\x80\x80\x80\x01"
TWO_BYTE_KEYS = b"\xa2\x01\x00\xca\x01\x00\xd2\x01\x00"
MODEL_WITH_EVERY_FIELD = (
    IR_VERSION_7
    + encode_string_record(2, b"producer")
    + encode_string_record(3, b"1.0")
    + encode_string_record(4, b"ai.onnx")
    + encode_key(MODEL_VERSION_FIELD, WireType.VARINT)
    + encode_varint(300)
    + encode_string_record(DOC_STRING_FIELD, b"doc")
    + encode_string_record(OPSET_IMPORT_FIELD, b"\x08\x11")
    + encode_string_record(METADATA_PROPS_FIELD, b"\x0a\x01k")
    + encode_string_record(TRAINING_INFO_FIELD, b"")
    + encode_string_record(FUNCTIONS_FIELD, b"")
    + encode_string_record(CONFIGURATION_FIELD, b"")
    + encode_string_record(GRAPH_FIELD, b"\x0a\x00")
)

ACCEPTED_MODELS = [
    AcceptedFile("the smallest model, an ir_version and an empty graph", SMALLEST_MODEL),
    AcceptedFile("a model of one Identity node, written byte by byte", MINIMAL_MODEL),
    AcceptedFile("a model with every field of the table", MODEL_WITH_EVERY_FIELD),
    *(
        AcceptedFile(
            f"field {field_number} ({field.name}) in its own wire type",
            SMALLEST_MODEL + encode_field_record(field_number, field.wire_type),
        )
        for field_number, field in ONNX_MODEL_FIELDS.items()
        if field_number not in (IR_VERSION_FIELD, GRAPH_FIELD)
    ),
    *(
        AcceptedFile(
            f"the repeatable field {field_number} ({field.name}) twice",
            SMALLEST_MODEL
            + encode_field_record(field_number, field.wire_type)
            + encode_field_record(field_number, field.wire_type),
        )
        for field_number, field in ONNX_MODEL_FIELDS.items()
        if field.is_repeatable
    ),
    AcceptedFile(
        "ir_version 1",
        encode_key(IR_VERSION_FIELD, WireType.VARINT) + encode_varint(1) + EMPTY_GRAPH,
    ),
    AcceptedFile(
        "ir_version 64",
        encode_key(IR_VERSION_FIELD, WireType.VARINT) + encode_varint(64) + EMPTY_GRAPH,
    ),
    AcceptedFile(
        "a model_version in 10 bytes whose last is 1",
        SMALLEST_MODEL
        + encode_key(MODEL_VERSION_FIELD, WireType.VARINT)
        + TEN_BYTE_VARINT_ENDING_IN_1,
    ),
    AcceptedFile("the two-byte keys of fields 20, 25 and 26", SMALLEST_MODEL + TWO_BYTE_KEYS),
    AcceptedFile("the graph before the ir_version", EMPTY_GRAPH + IR_VERSION_7),
]

REFUSED_MODELS = [
    *(
        RefusedFile(
            f"field {field_number} ({field.name}) in wire type {wire_type}",
            SMALLEST_MODEL + encode_field_record(field_number, wire_type),
            f"field {field_number} ({field.name}) is encoded with wire type {wire_type}, "
            f"not {field.wire_type.value}",
        )
        for field_number, field in ONNX_MODEL_FIELDS.items()
        for wire_type in range(HIGHEST_WIRE_TYPE_ENUMERATED + 1)
        if wire_type != field.wire_type
    ),
    *(
        RefusedFile(
            f"field {field_number}, outside the table",
            SMALLEST_MODEL + encode_field_record(field_number, WireType.LENGTH_DELIMITED),
            f"field {field_number} is not a field of an ONNX model",
        )
        for field_number in FIELDS_OUTSIDE_THE_TABLE
    ),
    RefusedFile(
        "a key padded with a zero byte",
        NON_MINIMAL_KEY_OF_IR_VERSION + encode_varint(7) + EMPTY_GRAPH,
        "a key is not in its shortest encoding",
    ),
    RefusedFile("a key of 6 bytes", SMALLEST_MODEL + SIX_BYTE_KEY, "a key is longer than 5 bytes"),
    RefusedFile(
        "a varint in 10 bytes whose last is 2",
        SMALLEST_MODEL
        + encode_key(MODEL_VERSION_FIELD, WireType.VARINT)
        + TEN_BYTE_VARINT_ENDING_IN_2,
        "a varint is longer than 64 bits",
    ),
    RefusedFile(
        "a varint of 11 bytes",
        SMALLEST_MODEL + encode_key(MODEL_VERSION_FIELD, WireType.VARINT) + ELEVEN_BYTE_VARINT,
        "a varint is longer than 64 bits",
    ),
    RefusedFile(
        "a length of 11 bytes",
        SMALLEST_MODEL
        + encode_key(DOC_STRING_FIELD, WireType.LENGTH_DELIMITED)
        + ELEVEN_BYTE_VARINT,
        "a varint is longer than 64 bits",
    ),
    RefusedFile(
        "a length beyond the end of the file",
        IR_VERSION_7
        + encode_key(GRAPH_FIELD, WireType.LENGTH_DELIMITED)
        + encode_varint(5)
        + b"\x00",
        "the file ends inside the content of field 7 (graph)",
    ),
    RefusedFile(
        "the ir_version twice",
        IR_VERSION_7 + SMALLEST_MODEL,
        "field 1 (ir_version) appears more than once",
    ),
    RefusedFile(
        "the graph twice", SMALLEST_MODEL + EMPTY_GRAPH, "field 7 (graph) appears more than once"
    ),
    RefusedFile(
        "ir_version 0",
        encode_key(IR_VERSION_FIELD, WireType.VARINT) + encode_varint(0) + EMPTY_GRAPH,
        "ir_version 0 is outside 1 to 64",
    ),
    RefusedFile(
        "ir_version 65",
        encode_key(IR_VERSION_FIELD, WireType.VARINT) + encode_varint(65) + EMPTY_GRAPH,
        "ir_version 65 is outside 1 to 64",
    ),
    RefusedFile("no ir_version", EMPTY_GRAPH, "the model has no ir_version (field 1)"),
    RefusedFile("no graph", IR_VERSION_7, "the model has no graph (field 7)"),
    RefusedFile("an empty file", b"", "the model has no ir_version (field 1)"),
    RefusedFile("a file cut inside a key", SMALLEST_MODEL + b"\xa2", "the file ends inside a key"),
    RefusedFile(
        "a file cut inside a varint",
        encode_key(IR_VERSION_FIELD, WireType.VARINT) + b"\x87",
        "the file ends inside the varint of field 1 (ir_version)",
    ),
    RefusedFile(
        "a file cut inside a length",
        SMALLEST_MODEL + encode_key(DOC_STRING_FIELD, WireType.LENGTH_DELIMITED) + b"\x81",
        "the file ends inside the length of field 6 (doc_string)",
    ),
]

TWO_BYTE_CHARACTER = "\u00e9".encode()
THREE_BYTE_CHARACTER = "\u20ac".encode()
FOUR_BYTE_CHARACTER = "\U0001f600".encode()
MULTIBYTE_TEXT = TWO_BYTE_CHARACTER + THREE_BYTE_CHARACTER + FOUR_BYTE_CHARACTER
NOT_UTF8 = "the text is not valid UTF-8"
ENDS_INSIDE_A_SEQUENCE = "the text ends inside a UTF-8 sequence"
HOLDS_A_NULL_CHARACTER = "the text holds a NUL character"
ACCEPTED_TEXTS = [
    AcceptedFile("ASCII", b"<unk> 0\n"),
    AcceptedFile("a sequence of 2 bytes", TWO_BYTE_CHARACTER),
    AcceptedFile("a sequence of 3 bytes", THREE_BYTE_CHARACTER),
    AcceptedFile("a sequence of 4 bytes", FOUR_BYTE_CHARACTER),
    AcceptedFile("a byte order mark, U+FEFF", "\ufeff".encode()),
    AcceptedFile("the last code point before the surrogates, U+D7FF", "\ud7ff".encode()),
    AcceptedFile("the first code point after the surrogates, U+E000", "\ue000".encode()),
    AcceptedFile("the last code point, U+10FFFF", "\U0010ffff".encode()),
    AcceptedFile("an empty text", b""),
]
REFUSED_TEXTS = [
    RefusedFile("a NUL first", b"\x00abc", HOLDS_A_NULL_CHARACTER),
    RefusedFile("a NUL in the middle", b"ab\x00c", HOLDS_A_NULL_CHARACTER),
    RefusedFile("a NUL last", b"abc\x00", HOLDS_A_NULL_CHARACTER),
    RefusedFile("the first surrogate, U+D800", b"\xed\xa0\x80", NOT_UTF8),
    RefusedFile("the last surrogate, U+DFFF", b"\xed\xbf\xbf", NOT_UTF8),
    RefusedFile("an overlong 2-byte form of U+0000", b"\xc0\x80", NOT_UTF8),
    RefusedFile("an overlong 2-byte form of U+007F", b"\xc1\xbf", NOT_UTF8),
    RefusedFile("an overlong 3-byte form of U+0000", b"\xe0\x80\x80", NOT_UTF8),
    RefusedFile("an overlong 3-byte form of U+07FF", b"\xe0\x9f\xbf", NOT_UTF8),
    RefusedFile("an overlong 4-byte form of U+0000", b"\xf0\x80\x80\x80", NOT_UTF8),
    RefusedFile("an overlong 4-byte form of U+FFFF", b"\xf0\x8f\xbf\xbf", NOT_UTF8),
    RefusedFile("beyond U+10FFFF", b"\xf4\x90\x80\x80", NOT_UTF8),
    *(
        RefusedFile(f"byte {byte:02X}", bytes([byte]) + b"a", NOT_UTF8)
        for byte in range(FIRST_BYTE_NEVER_IN_UTF8, BYTE_COUNT)
    ),
    RefusedFile("a lone continuation byte", b"a\x80b", NOT_UTF8),
    RefusedFile(
        "a continuation byte after a whole sequence", TWO_BYTE_CHARACTER + b"\x80", NOT_UTF8
    ),
    RefusedFile(
        "a 2-byte sequence cut at the end", b"a" + TWO_BYTE_CHARACTER[:1], ENDS_INSIDE_A_SEQUENCE
    ),
    RefusedFile(
        "a 3-byte sequence cut at the end", THREE_BYTE_CHARACTER[:2], ENDS_INSIDE_A_SEQUENCE
    ),
    RefusedFile(
        "a 4-byte sequence cut after its first byte",
        FOUR_BYTE_CHARACTER[:1],
        ENDS_INSIDE_A_SEQUENCE,
    ),
    RefusedFile("a truncated end", FOUR_BYTE_CHARACTER[:3], ENDS_INSIDE_A_SEQUENCE),
    RefusedFile("a 3-byte sequence cut before ASCII", THREE_BYTE_CHARACTER[:2] + b"a", NOT_UTF8),
    RefusedFile(
        "a 2-byte sequence cut before another sequence",
        TWO_BYTE_CHARACTER[:1] + TWO_BYTE_CHARACTER,
        NOT_UTF8,
    ),
    RefusedFile("a NUL before an invalid byte", b"\x00\xff", HOLDS_A_NULL_CHARACTER),
    RefusedFile("an invalid byte before a NUL", b"\xff\x00", NOT_UTF8),
]
FILES_CUT_ANYWHERE = [
    (FileFormat.ONNX, "a model with every field", MODEL_WITH_EVERY_FIELD),
    (FileFormat.ONNX, "a model of one Identity node", MINIMAL_MODEL),
    *((FileFormat.ONNX, file.name, file.content) for file in REFUSED_MODELS),
    (FileFormat.TEXT, "multibyte sequences", MULTIBYTE_TEXT),
    *((FileFormat.TEXT, file.name, file.content) for file in REFUSED_TEXTS),
]
# A file of n bytes has n + 1 cuts in two pieces, and one more feeding it byte by byte.
CUTS_BEYOND_THE_FILE_LENGTH = 2


class TestTheFileFormats:
    class TestTheHasStructuralCheckFunction:
        @pytest.mark.parametrize(
            ("file_format", "is_expected"),
            list(STRUCTURALLY_CHECKED_FORMATS.items()),
            ids=[file_format.value for file_format in STRUCTURALLY_CHECKED_FORMATS],
        )
        class TestWhenGivenAFormat:
            @pytest.fixture
            def has_check(self, file_format: FileFormat) -> bool:
                return has_structural_check(file_format)

            def test_should_tell_whether_its_structure_is_checked_as_the_file_streams_by(
                self, has_check: bool, is_expected: bool
            ) -> None:
                assert has_check is is_expected

        @pytest.mark.parametrize("file_format", list(FileFormat), ids=str)
        class TestWhenGivenEachFormatThatTheBuilderKnows:
            @pytest.fixture
            def answers(self, file_format: FileFormat) -> tuple[bool, bool]:
                return has_structural_check(file_format), starts_a_check(file_format)

            def test_should_answer_as_start_format_check_does(
                self, answers: tuple[bool, bool]
            ) -> None:
                has_check, starts = answers
                assert has_check is starts

    class TestTheStartFormatCheckFunction:
        class TestWhenTheFormatIsOnnxExternalData:
            @pytest.fixture
            def start(self) -> Callable[[], FormatCheck]:
                return lambda: start_format_check(FileFormat.ONNX_EXTERNAL_DATA)

            def test_should_refuse_it_since_its_onnx_file_checks_it(
                self, start: Callable[[], FormatCheck]
            ) -> None:
                with pytest.raises(
                    FormatWithoutStructuralCheckError, match=re.escape(EXTERNAL_DATA_REFUSAL)
                ):
                    start()

    class TestTheFinishMethod:
        @pytest.mark.parametrize("content", build_accepted_parameters(ACCEPTED_MODELS))
        class TestWhenAnOnnxModelFollowsTheTable:
            @pytest.fixture
            def mismatch(self, content: bytes) -> FormatMismatch | None:
                return judge_whole_file(FileFormat.ONNX, content)

            def test_should_accept_it(self, mismatch: FormatMismatch | None) -> None:
                assert mismatch is None

        @pytest.mark.parametrize(("content", "reason"), build_refused_parameters(REFUSED_MODELS))
        class TestWhenAnOnnxModelBreaksTheTable:
            @pytest.fixture
            def mismatch(self, content: bytes) -> FormatMismatch | None:
                return judge_whole_file(FileFormat.ONNX, content)

            def test_should_refuse_it_naming_the_reason(
                self, mismatch: FormatMismatch | None, reason: str
            ) -> None:
                assert mismatch == FormatMismatch(reason)

        class TestWhenAnyKeyFollowsTheSmallestModel:
            @pytest.fixture(scope="class")
            @classmethod
            def judgment(cls) -> Judgment[tuple[int, int]]:
                keys = [
                    (field_number, wire_type)
                    for field_number in range(HIGHEST_FIELD_ENUMERATED + 1)
                    for wire_type in range(HIGHEST_WIRE_TYPE_ENUMERATED + 1)
                ]
                misjudged = [
                    (field_number, wire_type)
                    for field_number, wire_type in keys
                    if (
                        judge_whole_file(
                            FileFormat.ONNX,
                            SMALLEST_MODEL + encode_field_record(field_number, wire_type),
                        )
                        is None
                    )
                    != is_admitted_after_the_smallest_model(field_number, wire_type)
                ]
                return Judgment(misjudged, len(keys))

            def test_should_accept_only_a_table_field_in_its_wire_type_and_a_singular_one_once(
                self, judgment: Judgment[tuple[int, int]]
            ) -> None:
                assert judgment.misjudged == []

            def test_should_judge_every_field_up_to_63_in_every_wire_type(
                self, judgment: Judgment[tuple[int, int]]
            ) -> None:
                assert judgment.judged_count == ENUMERATED_KEY_COUNT

        class TestWhenAModelIsCutShort:
            @pytest.fixture(scope="class")
            @classmethod
            def judgment(cls) -> Judgment[int]:
                prefix_lengths = range(len(MODEL_WITH_EVERY_FIELD))
                accepted = [
                    length
                    for length in prefix_lengths
                    if judge_whole_file(FileFormat.ONNX, MODEL_WITH_EVERY_FIELD[:length]) is None
                ]
                return Judgment(accepted, len(prefix_lengths))

            def test_should_refuse_every_strict_prefix(self, judgment: Judgment[int]) -> None:
                assert judgment.misjudged == []

            def test_should_cut_the_model_after_each_of_its_bytes_but_the_last(
                self, judgment: Judgment[int]
            ) -> None:
                assert judgment.judged_count == len(MODEL_WITH_EVERY_FIELD)

        class TestWhenGivenAnyTruncationOrSingleByteChangeOfAMinimalModel:
            def test_should_pass_it_or_refuse_it_with_a_mismatch_never_with_another_error(
                self,
            ) -> None:
                generator = random.Random(PROPERTY_SEED)
                damaged_models = build_damaged_models(MINIMAL_MODEL, generator)
                errors = [
                    error
                    for model in damaged_models
                    for error in describe_errors_other_than_a_mismatch(model)
                ]
                assert errors == [], f"seed {PROPERTY_SEED}"
                assert len(damaged_models) == len(MINIMAL_MODEL) + SAMPLED_CHANGE_COUNT, (
                    f"seed {PROPERTY_SEED}"
                )

            def test_should_reach_the_verdict_of_the_whole_file_wherever_it_is_cut(
                self,
            ) -> None:
                generator = random.Random(PROPERTY_SEED)
                damaged_models = build_damaged_models(MINIMAL_MODEL, generator)
                misjudged = [
                    model.hex()
                    for model in damaged_models
                    if judge_chunks(FileFormat.ONNX, cut_at_random(model, generator))
                    != judge_whole_file(FileFormat.ONNX, model)
                ]
                assert misjudged == [], f"seed {PROPERTY_SEED}"
                assert len(damaged_models) == len(MINIMAL_MODEL) + SAMPLED_CHANGE_COUNT, (
                    f"seed {PROPERTY_SEED}"
                )

        @pytest.mark.parametrize("content", build_accepted_parameters(ACCEPTED_TEXTS))
        class TestWhenATextIsStrictUtf8WithoutANullCharacter:
            @pytest.fixture
            def mismatch(self, content: bytes) -> FormatMismatch | None:
                return judge_whole_file(FileFormat.TEXT, content)

            def test_should_accept_it(self, mismatch: FormatMismatch | None) -> None:
                assert mismatch is None

        @pytest.mark.parametrize(("content", "reason"), build_refused_parameters(REFUSED_TEXTS))
        class TestWhenATextBreaksStrictUtf8OrHoldsANullCharacter:
            @pytest.fixture
            def mismatch(self, content: bytes) -> FormatMismatch | None:
                return judge_whole_file(FileFormat.TEXT, content)

            def test_should_refuse_it_naming_its_first_fault(
                self, mismatch: FormatMismatch | None, reason: str
            ) -> None:
                assert mismatch == FormatMismatch(reason)

        class TestWhenATextIsASingleByte:
            @pytest.fixture(scope="class")
            @classmethod
            def judgment(cls) -> Judgment[int]:
                every_byte = range(BYTE_COUNT)
                misjudged = [
                    byte
                    for byte in every_byte
                    if (judge_whole_file(FileFormat.TEXT, bytes([byte])) is None)
                    != (LOWEST_ASCII_BYTE_AFTER_NULL <= byte <= HIGHEST_ASCII_BYTE)
                ]
                return Judgment(misjudged, len(every_byte))

            def test_should_accept_only_the_ascii_bytes_after_the_null_byte(
                self, judgment: Judgment[int]
            ) -> None:
                assert judgment.misjudged == []

            def test_should_judge_every_byte(self, judgment: Judgment[int]) -> None:
                assert judgment.judged_count == BYTE_COUNT

        class TestWhenAFileIsFedInChunksCutAnywhere:
            @pytest.fixture(scope="class")
            @classmethod
            def judgment(cls) -> Judgment[tuple[str, list[bytes]]]:
                cut_files = [
                    (file_format, name, content, chunks)
                    for file_format, name, content in FILES_CUT_ANYWHERE
                    for chunks in split_anywhere(content)
                ]
                misjudged = [
                    (name, chunks)
                    for file_format, name, content, chunks in cut_files
                    if judge_chunks(file_format, chunks) != judge_whole_file(file_format, content)
                ]
                return Judgment(misjudged, len(cut_files))

            def test_should_reach_the_verdict_of_the_whole_file(
                self, judgment: Judgment[tuple[str, list[bytes]]]
            ) -> None:
                assert judgment.misjudged == []

            def test_should_cut_every_file_at_each_position_and_byte_by_byte(
                self, judgment: Judgment[tuple[str, list[bytes]]]
            ) -> None:
                assert judgment.judged_count == sum(
                    len(content) + CUTS_BEYOND_THE_FILE_LENGTH
                    for _, _, content in FILES_CUT_ANYWHERE
                )
