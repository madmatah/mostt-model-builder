LOG_CAP_BYTES = 16 * 2**20
TAIL_BYTES = 8 * 2**10
REPLACEMENT = "?"
TAB = 0x09
LINE_FEED = 0x0A
FIRST_PRINTABLE = 0x20
LAST_PRINTABLE = 0x7E
PRINTABLE_TRANSLATION = bytes(
    byte
    if byte in (TAB, LINE_FEED) or FIRST_PRINTABLE <= byte <= LAST_PRINTABLE
    else ord(REPLACEMENT)
    for byte in range(256)
)


def filter_printable(raw: bytes) -> bytes:
    return raw.translate(PRINTABLE_TRANSLATION)


def show_printable(raw: bytes) -> str:
    return filter_printable(raw).decode("ascii")


def take_within_cap(written_size_bytes: int, chunk: bytes) -> bytes:
    return chunk[: max(0, LOG_CAP_BYTES - written_size_bytes)]


def keep_tail(tail: bytes, chunk: bytes) -> bytes:
    return (tail + chunk)[-TAIL_BYTES:]
