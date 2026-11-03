import pytest

from mostt_model_builder.domain.containers.printable_output import (
    LOG_CAP_BYTES,
    TAIL_BYTES,
    filter_printable,
    keep_tail,
    take_within_cap,
)

KEPT_CONTROL_BYTES = frozenset({0x09, 0x0A})
FIRST_PRINTABLE = 0x20
LAST_PRINTABLE = 0x7E
REPLACEMENT = ord("?")
CHUNK = b"x" * 100


def is_kept(byte: int) -> bool:
    return byte in KEPT_CONTROL_BYTES or FIRST_PRINTABLE <= byte <= LAST_PRINTABLE


class TestThePrintableOutput:
    class TestTheFilterPrintableFunction:
        class TestWhenGivenEachOfThe256ByteValues:
            @pytest.fixture
            def filtered(self) -> bytes:
                return filter_printable(bytes(range(256)))

            def test_should_keep_printable_ascii_the_tab_and_the_line_feed_and_mark_the_rest(
                self, filtered: bytes
            ) -> None:
                assert filtered == bytes(
                    byte if is_kept(byte) else REPLACEMENT for byte in range(256)
                )

    class TestTheTakeWithinCapFunction:
        class TestWhenTheLogIsBelowItsCap:
            @pytest.fixture
            def kept(self) -> bytes:
                return take_within_cap(0, CHUNK)

            def test_should_keep_the_whole_chunk(self, kept: bytes) -> None:
                assert kept == CHUNK

        class TestWhenTheChunkCrossesTheCap:
            @pytest.fixture
            def kept(self) -> bytes:
                return take_within_cap(LOG_CAP_BYTES - 10, CHUNK)

            def test_should_keep_only_what_fits_under_16_mib(self, kept: bytes) -> None:
                assert kept == CHUNK[:10]

        class TestWhenTheLogHasReachedItsCap:
            @pytest.fixture
            def kept(self) -> bytes:
                return take_within_cap(LOG_CAP_BYTES, CHUNK)

            def test_should_drain_the_chunk_and_keep_nothing(self, kept: bytes) -> None:
                assert kept == b""

    class TestTheKeepTailFunction:
        class TestWhenTheOutputExceedsTheTail:
            @pytest.fixture
            def tail(self) -> bytes:
                return keep_tail(b"a" * TAIL_BYTES, b"b" * 10)

            def test_should_keep_its_last_8_kib(self, tail: bytes) -> None:
                assert tail == b"a" * (TAIL_BYTES - 10) + b"b" * 10
