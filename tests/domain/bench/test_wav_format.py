import re
import struct
from collections.abc import Callable
from functools import partial

import pytest

from mostt_model_builder.domain.bench.wav_format import (
    SampleFormat,
    WavFormatError,
    WavLayout,
    parse_wav_layout,
)

SAMPLE_COUNT = 160
FLOAT_FORMAT_CODE = 3
INTEGER_FORMAT_CODE = 1


def build_format_chunk(
    format_code: int, channel_count: int, sample_rate_hz: int, bits_per_sample: int
) -> bytes:
    block_size_bytes = channel_count * bits_per_sample // 8
    body = struct.pack(
        "<HHIIHH",
        format_code,
        channel_count,
        sample_rate_hz,
        sample_rate_hz * block_size_bytes,
        block_size_bytes,
        bits_per_sample,
    )
    return b"fmt " + struct.pack("<I", len(body)) + body


def build_chunk(chunk_id: bytes, body: bytes) -> bytes:
    return chunk_id + struct.pack("<I", len(body)) + body + b"\0" * (len(body) % 2)


def build_wav(*chunks: bytes) -> bytes:
    body = b"WAVE" + b"".join(chunks)
    return b"RIFF" + struct.pack("<I", len(body)) + body


FLOAT_FORMAT = build_format_chunk(FLOAT_FORMAT_CODE, 1, 16_000, 32)
INTEGER_FORMAT = build_format_chunk(INTEGER_FORMAT_CODE, 1, 16_000, 16)
FLOAT_DATA = build_chunk(b"data", b"\0" * (SAMPLE_COUNT * 4))
ACCEPTED_FILES = {
    "32-bit floats": (build_wav(FLOAT_FORMAT, FLOAT_DATA), SampleFormat.FLOAT32, 44),
    "16-bit integers": (
        build_wav(INTEGER_FORMAT, build_chunk(b"data", b"\0" * (SAMPLE_COUNT * 2))),
        SampleFormat.INTEGER16,
        44,
    ),
    "a list chunk before the data": (
        build_wav(FLOAT_FORMAT, build_chunk(b"LIST", b"INFOx"), FLOAT_DATA),
        SampleFormat.FLOAT32,
        58,
    ),
}
REFUSED_FILES = {
    "too few bytes for a header": (b"RIFF", "4 bytes are too few for a WAV header"),
    "not riff": (b"RIFX" + build_wav(FLOAT_FORMAT, FLOAT_DATA)[4:], "not a RIFF WAVE file"),
    "not wave": (build_wav(FLOAT_FORMAT, FLOAT_DATA)[:8] + b"AVI " + b"\0" * 8, "not a RIFF WAVE"),
    "no data chunk": (build_wav(FLOAT_FORMAT), "the file has no data chunk"),
    "data before its format": (
        build_wav(FLOAT_DATA, FLOAT_FORMAT),
        "the data chunk comes before any format chunk",
    ),
    "a format chunk too short": (
        build_wav(build_chunk(b"fmt ", b"\x03\x00"), FLOAT_DATA),
        "the format chunk holds 2 bytes, too few",
    ),
    "a chunk beyond the end of the file": (
        build_wav(FLOAT_FORMAT, b"data" + struct.pack("<I", 1000) + b"\0" * 10),
        "declares 1000 bytes, beyond the file's end",
    ),
    "two channels": (
        build_wav(build_format_chunk(FLOAT_FORMAT_CODE, 2, 16_000, 32), FLOAT_DATA),
        "has 2 channels, not one",
    ),
    "another sample rate": (
        build_wav(build_format_chunk(FLOAT_FORMAT_CODE, 1, 44_100, 32), FLOAT_DATA),
        "sampled at 44100 Hz, not 16000 Hz",
    ),
    "64-bit floats": (
        build_wav(build_format_chunk(FLOAT_FORMAT_CODE, 1, 16_000, 64), FLOAT_DATA),
        "the format 3 with 64 bits per sample is neither",
    ),
    "a compressed format": (
        build_wav(build_format_chunk(6, 1, 16_000, 8), FLOAT_DATA),
        "the format 6 with 8 bits per sample is neither",
    ),
    "a part of a sample": (
        build_wav(FLOAT_FORMAT, build_chunk(b"data", b"\0" * 6)),
        "the data chunk holds 6 bytes, not a whole count of samples",
    ),
}


class TestTheWavFormat:
    class TestTheParseWavLayoutFunction:
        @pytest.mark.parametrize(
            ("content", "sample_format", "data_offset"),
            list(ACCEPTED_FILES.values()),
            ids=list(ACCEPTED_FILES.keys()),
        )
        class TestWhenTheFileIsMonoAt16Khz:
            @pytest.fixture
            def layout(self, content: bytes) -> WavLayout:
                return parse_wav_layout(content)

            def test_should_find_its_samples_and_count_them(
                self, layout: WavLayout, sample_format: SampleFormat, data_offset: int
            ) -> None:
                assert (layout.sample_format, layout.data_offset, layout.sample_count) == (
                    sample_format,
                    data_offset,
                    SAMPLE_COUNT,
                )

        @pytest.mark.parametrize(
            ("content", "reason"), list(REFUSED_FILES.values()), ids=list(REFUSED_FILES.keys())
        )
        class TestWhenTheFileIsNotOneTheBenchReads:
            @pytest.fixture
            def parse_act(self, content: bytes) -> Callable[[], WavLayout]:
                return partial(parse_wav_layout, content)

            def test_should_refuse_it_and_say_why(
                self, parse_act: Callable[[], WavLayout], reason: str
            ) -> None:
                with pytest.raises(WavFormatError, match=re.escape(reason)):
                    parse_act()
