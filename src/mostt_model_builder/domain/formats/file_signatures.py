from collections.abc import Iterable
from dataclasses import dataclass

SIGNATURE_HEAD_BYTES = 512
EVERY_BYTE = frozenset(range(256))
PICKLE_PROTOCOL_OPCODE = 0x80
# The one-byte codes of every opcode that pickletools.opcodes lists, written out so that
# the builder never imports anything of pickle; a test keeps the two equal.
PICKLE_OPCODE_CODES = frozenset(
    b"().012BCFGIJKLMNPQRSTUVX]abcdeghijlopqrstu}"
    b"\x80\x81\x82\x83\x84\x85\x86\x87\x88\x89\x8a\x8b\x8c\x8d\x8e\x8f"
    b"\x90\x91\x92\x93\x94\x95\x96\x97\x98"
)
TAR_MAGIC_OFFSET_BYTES = 257


@dataclass(frozen=True, slots=True)
class Signature:
    marks: str
    offset_bytes: int
    byte_choices: tuple[frozenset[int], ...]

    def is_found_in(self, head: bytes) -> bool:
        end = self.offset_bytes + len(self.byte_choices)
        if len(head) < end:
            return False
        return all(
            byte in choices
            for byte, choices in zip(head[self.offset_bytes : end], self.byte_choices, strict=True)
        )


def build_exact_signature(marks: str, signature_bytes: bytes, offset_bytes: int = 0) -> Signature:
    return Signature(marks, offset_bytes, tuple(frozenset({byte}) for byte in signature_bytes))


def build_signature_with_last_choices(
    marks: str, signature_bytes: bytes, last_choices: Iterable[int]
) -> Signature:
    exact = build_exact_signature(marks, signature_bytes)
    return Signature(marks, 0, (*exact.byte_choices, frozenset(last_choices)))


SIGNATURES = (
    Signature(
        "a binary pickle",
        0,
        (frozenset({PICKLE_PROTOCOL_OPCODE}), EVERY_BYTE, PICKLE_OPCODE_CODES),
    ),
    build_exact_signature("a zip archive", b"PK\x03\x04"),
    build_exact_signature("an empty zip archive", b"PK\x05\x06"),
    build_exact_signature("a spanned zip archive", b"PK\x07\x08"),
    build_exact_signature("a numpy array", b"\x93NUMPY"),
    build_exact_signature("a native executable", b"\x7fELF"),
    build_exact_signature("a script", b"#!"),
    build_exact_signature("a tar archive", b"ustar", TAR_MAGIC_OFFSET_BYTES),
    build_exact_signature("a gzip stream", b"\x1f\x8b\x08"),
    build_signature_with_last_choices("a bzip2 stream", b"BZh", b"123456789"),
    build_exact_signature("an xz stream", b"\xfd7zXZ\x00"),
    build_exact_signature("a Zstandard stream", b"\x28\xb5\x2f\xfd"),
    build_exact_signature("a 7-Zip archive", b"7z\xbc\xaf\x27\x1c"),
)


@dataclass(frozen=True, slots=True)
class SignatureRefusal:
    marks: str


def find_signature_refusal(head: bytes) -> SignatureRefusal | None:
    first_bytes = head[:SIGNATURE_HEAD_BYTES]
    for signature in SIGNATURES:
        if signature.is_found_in(first_bytes):
            return SignatureRefusal(signature.marks)
    return None
