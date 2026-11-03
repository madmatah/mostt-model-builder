import pickletools
import struct
from dataclasses import dataclass

import pytest

from mostt_model_builder.domain.formats.file_signatures import (
    PICKLE_OPCODE_CODES,
    PICKLE_PROTOCOL_OPCODE,
    SIGNATURE_HEAD_BYTES,
    SIGNATURES,
    TAR_MAGIC_OFFSET_BYTES,
    Signature,
    SignatureRefusal,
    find_signature_refusal,
)

BYTE_COUNT = 256
PAIR_COUNT = BYTE_COUNT * BYTE_COUNT
SAFETENSORS_HEADER_LENGTHS = (128, 384, 640)
CONSTRAINED_BYTE_COUNT = sum(
    1 for signature in SIGNATURES for choices in signature.byte_choices if len(choices) < BYTE_COUNT
)
SCRIPT_HEAD = b"#!/bin/sh\n"


@dataclass(frozen=True, slots=True)
class Judgment[Case]:
    misjudged: list[Case]
    judged_count: int


def pad_with_zeros(start: bytes) -> bytes:
    return start + bytes(SIGNATURE_HEAD_BYTES - len(start))


def place_at_offset(offset_bytes: int, signature_bytes: bytes) -> bytes:
    return pad_with_zeros(bytes(offset_bytes) + signature_bytes)


def build_safetensors_head(header_length: int) -> bytes:
    return struct.pack("<Q", header_length) + b'{"weight":{"dtype":"F32"'


def find_bytes_outside(choices: frozenset[int]) -> list[int]:
    return sorted(set(range(BYTE_COUNT)) - choices)


def build_matching_bytes(signature: Signature) -> bytes:
    return bytes(min(choices) for choices in signature.byte_choices)


def build_file_holding(signature: Signature) -> bytes:
    return bytes(signature.offset_bytes) + build_matching_bytes(signature)


def build_file_with_its_last_byte_changed(signature: Signature) -> bytes:
    last_byte_off_the_signature = find_bytes_outside(signature.byte_choices[-1])[0]
    return build_file_holding(signature)[:-1] + bytes([last_byte_off_the_signature])


def build_file_one_byte_short_of(signature: Signature) -> bytes:
    return build_file_holding(signature)[:-1]


def build_proper_prefixes_completed_off(signature: Signature) -> list[bytes]:
    return [
        place_at_offset(
            signature.offset_bytes,
            build_matching_bytes(signature)[:prefix_length] + bytes([next_byte]),
        )
        for prefix_length in range(len(signature.byte_choices))
        for next_byte in find_bytes_outside(signature.byte_choices[prefix_length])[:1]
    ]


def describe_signature(signature: Signature) -> str:
    return f"{signature.marks} at offset {signature.offset_bytes}"


REFUSED_HEADS = [
    pytest.param(b"\x80\x04\x95\x10\x00", "a binary pickle", id="a pickle of protocol 4"),
    pytest.param(b"\x80\x02}q\x00", "a binary pickle", id="a pickle of protocol 2"),
    pytest.param(b"PK\x03\x04\x14\x00", "a zip archive", id="a zip archive"),
    pytest.param(b"PK\x05\x06\x00\x00", "an empty zip archive", id="an empty zip archive"),
    pytest.param(b"PK\x07\x08\x00\x00", "a spanned zip archive", id="a spanned zip archive"),
    pytest.param(b"\x93NUMPY\x01\x00", "a numpy array", id="a numpy array"),
    pytest.param(b"\x7fELF\x02\x01\x01", "a native executable", id="an ELF executable"),
    pytest.param(SCRIPT_HEAD, "a script", id="a script of 10 bytes, shorter than the head"),
    pytest.param(
        SCRIPT_HEAD + bytes(SIGNATURE_HEAD_BYTES),
        "a script",
        id="a script longer than the head",
    ),
    pytest.param(
        place_at_offset(TAR_MAGIC_OFFSET_BYTES, b"ustar\x0000"),
        "a tar archive",
        id="a POSIX tar archive",
    ),
    pytest.param(
        place_at_offset(TAR_MAGIC_OFFSET_BYTES, b"ustar  \x00"),
        "a tar archive",
        id="a GNU tar archive",
    ),
    pytest.param(b"\x1f\x8b\x08\x00", "a gzip stream", id="a gzip stream"),
    pytest.param(b"BZh1", "a bzip2 stream", id="a bzip2 stream of 100 kB blocks"),
    pytest.param(b"BZh9", "a bzip2 stream", id="a bzip2 stream of 900 kB blocks"),
    pytest.param(b"\xfd7zXZ\x00\x00\x04", "an xz stream", id="an xz stream"),
    pytest.param(b"\x28\xb5\x2f\xfd\x04", "a Zstandard stream", id="a Zstandard stream"),
    pytest.param(b"7z\xbc\xaf\x27\x1c\x00\x04", "a 7-Zip archive", id="a 7-Zip archive"),
]
ACCEPTED_HEADS = [
    *(
        pytest.param(
            build_safetensors_head(header_length),
            id=f"a safetensors header of {header_length} bytes",
        )
        for header_length in SAFETENSORS_HEADER_LENGTHS
    ),
    pytest.param(b"\x08\x07\x12\x07pytorch", id="the first key of an ONNX model"),
    pytest.param(b"<unk> 0\n<|nospeech|> 1\n", id="the first tokens of Parakeet's tokens.txt"),
    pytest.param(b"<unk> 0\n<bg-BG> 1\n", id="the first tokens of Nemotron's tokens.txt"),
    pytest.param(b"RIFF\x24\x08\x00\x00WAVEfmt ", id="a WAV recording"),
    pytest.param(b'{"architectures": ["x"]}', id="a JSON object"),
    pytest.param(
        b"cos\nsystem\n(S'echo'\ntR.", id="a text pickle of protocol 0, which no signature marks"
    ),
    pytest.param(b"P", id="a head shorter than any signature"),
    pytest.param(b"", id="an empty file"),
]


class TestTheFileSignatures:
    class TestTheFindSignatureRefusalFunction:
        @pytest.mark.parametrize(("head", "marks"), REFUSED_HEADS)
        class TestWhenTheHeadStartsWithASignature:
            @pytest.fixture
            def refusal(self, head: bytes) -> SignatureRefusal | None:
                return find_signature_refusal(head)

            def test_should_refuse_it_naming_what_the_signature_marks(
                self, refusal: SignatureRefusal | None, marks: str
            ) -> None:
                assert refusal == SignatureRefusal(marks)

        @pytest.mark.parametrize("head", ACCEPTED_HEADS)
        class TestWhenTheHeadStartsWithNoSignature:
            @pytest.fixture
            def refusal(self, head: bytes) -> SignatureRefusal | None:
                return find_signature_refusal(head)

            def test_should_accept_it(self, refusal: SignatureRefusal | None) -> None:
                assert refusal is None

        @pytest.mark.parametrize("signature", SIGNATURES, ids=describe_signature)
        class TestWhenGivenEachSignatureOfTheTable:
            class TestWhenAFileHoldsItsBytesAtItsOffset:
                @pytest.fixture
                def refusal(self, signature: Signature) -> SignatureRefusal | None:
                    return find_signature_refusal(build_file_holding(signature))

                def test_should_refuse_it_naming_what_the_signature_marks(
                    self, refusal: SignatureRefusal | None, signature: Signature
                ) -> None:
                    assert refusal == SignatureRefusal(signature.marks)

            class TestWhenTheLastOfItsBytesIsChanged:
                @pytest.fixture
                def refusal(self, signature: Signature) -> SignatureRefusal | None:
                    return find_signature_refusal(build_file_with_its_last_byte_changed(signature))

                def test_should_accept_the_file(self, refusal: SignatureRefusal | None) -> None:
                    assert refusal is None

            class TestWhenTheFileEndsOneByteShortOfIt:
                @pytest.fixture
                def refusal(self, signature: Signature) -> SignatureRefusal | None:
                    return find_signature_refusal(build_file_one_byte_short_of(signature))

                def test_should_accept_the_file(self, refusal: SignatureRefusal | None) -> None:
                    assert refusal is None

        class TestWhenASignatureLiesBeyondTheHead:
            @pytest.fixture
            def refusal(self) -> SignatureRefusal | None:
                return find_signature_refusal(bytes(SIGNATURE_HEAD_BYTES - 1) + SCRIPT_HEAD)

            def test_should_read_only_the_first_512_bytes(
                self, refusal: SignatureRefusal | None
            ) -> None:
                assert refusal is None

        class TestWhenAnyTwoBytesFollowTheProtocolOpcode:
            @pytest.fixture(scope="class")
            @classmethod
            def judgment(cls) -> Judgment[tuple[int, int]]:
                pairs = [
                    (second_byte, third_byte)
                    for second_byte in range(BYTE_COUNT)
                    for third_byte in range(BYTE_COUNT)
                ]
                misjudged = [
                    (second_byte, third_byte)
                    for second_byte, third_byte in pairs
                    if (
                        find_signature_refusal(
                            bytes([PICKLE_PROTOCOL_OPCODE, second_byte, third_byte])
                        )
                        is None
                    )
                    == (third_byte in PICKLE_OPCODE_CODES)
                ]
                return Judgment(misjudged, len(pairs))

            def test_should_refuse_exactly_the_heads_whose_third_byte_is_an_opcode(
                self, judgment: Judgment[tuple[int, int]]
            ) -> None:
                assert judgment.misjudged == []

            def test_should_judge_every_pair(self, judgment: Judgment[tuple[int, int]]) -> None:
                assert judgment.judged_count == PAIR_COUNT

        class TestWhenGivenAnyFirstByteFollowedByZeros:
            @pytest.fixture
            def refused_first_bytes(self) -> list[int]:
                return [
                    first_byte
                    for first_byte in range(BYTE_COUNT)
                    if find_signature_refusal(pad_with_zeros(bytes([first_byte]))) is not None
                ]

            def test_should_refuse_no_head_of_one_byte_and_zeros(
                self, refused_first_bytes: list[int]
            ) -> None:
                assert refused_first_bytes == []

        class TestWhenGivenAProperPrefixOfASignatureFollowedOffIt:
            @pytest.fixture
            def judgment(self) -> Judgment[bytes]:
                heads = [
                    head
                    for signature in SIGNATURES
                    for head in build_proper_prefixes_completed_off(signature)
                ]
                refused_heads = [head for head in heads if find_signature_refusal(head) is not None]
                return Judgment(refused_heads, len(heads))

            def test_should_refuse_no_such_head(self, judgment: Judgment[bytes]) -> None:
                assert judgment.misjudged == []

            def test_should_judge_one_head_per_byte_that_a_signature_constrains(
                self, judgment: Judgment[bytes]
            ) -> None:
                assert CONSTRAINED_BYTE_COUNT > 0
                assert judgment.judged_count == CONSTRAINED_BYTE_COUNT

    class TestTheSignatures:
        class TestWhenComparedWithTheHead:
            def test_should_each_end_within_the_first_512_bytes_so_that_each_can_be_found(
                self,
            ) -> None:
                signatures_beyond_the_head = [
                    signature.marks
                    for signature in SIGNATURES
                    if signature.offset_bytes + len(signature.byte_choices) > SIGNATURE_HEAD_BYTES
                ]
                assert signatures_beyond_the_head == []
                assert len(SIGNATURES) > 0

    class TestThePickleOpcodeCodes:
        class TestWhenComparedWithTheOpcodesOfPickletools:
            def test_should_hold_the_code_of_every_opcode_and_nothing_else(self) -> None:
                assert {ord(opcode.code) for opcode in pickletools.opcodes} == PICKLE_OPCODE_CODES
