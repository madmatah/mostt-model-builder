from dataclasses import dataclass
from enum import Enum

SAMPLE_RATE_HZ = 16_000
CHANNEL_COUNT = 1
RIFF_HEADER_SIZE_BYTES = 12
CHUNK_HEADER_SIZE_BYTES = 8
FORMAT_CHUNK_MIN_SIZE_BYTES = 16
RIFF_MAGIC = b"RIFF"
WAVE_MAGIC = b"WAVE"
FORMAT_CHUNK_ID = b"fmt "
DATA_CHUNK_ID = b"data"
INTEGER_PCM_FORMAT_CODE = 1
FLOAT_FORMAT_CODE = 3


class WavFormatError(ValueError):
    pass


class SampleFormat(Enum):
    INTEGER16 = (INTEGER_PCM_FORMAT_CODE, 16)
    FLOAT32 = (FLOAT_FORMAT_CODE, 32)

    @property
    def sample_size_bytes(self) -> int:
        return self.value[1] // 8


@dataclass(frozen=True, slots=True)
class WavLayout:
    sample_format: SampleFormat
    data_offset: int
    data_size_bytes: int

    @property
    def sample_count(self) -> int:
        return self.data_size_bytes // self.sample_format.sample_size_bytes


@dataclass(frozen=True, slots=True)
class FormatChunk:
    format_code: int
    channel_count: int
    sample_rate_hz: int
    block_size_bytes: int
    bits_per_sample: int


def read_little_endian(content: bytes, offset: int, size_bytes: int) -> int:
    return int.from_bytes(content[offset : offset + size_bytes], "little")


def parse_wav_layout(content: bytes) -> WavLayout:
    if len(content) < RIFF_HEADER_SIZE_BYTES:
        raise WavFormatError(f"{len(content)} bytes are too few for a WAV header")
    if content[:4] != RIFF_MAGIC or content[8:12] != WAVE_MAGIC:
        raise WavFormatError("the file is not a RIFF WAVE file")
    format_chunk: FormatChunk | None = None
    offset = RIFF_HEADER_SIZE_BYTES
    while offset + CHUNK_HEADER_SIZE_BYTES <= len(content):
        chunk_id = content[offset : offset + 4]
        chunk_size_bytes = read_little_endian(content, offset + 4, 4)
        body_offset = offset + CHUNK_HEADER_SIZE_BYTES
        if body_offset + chunk_size_bytes > len(content):
            raise WavFormatError(
                f"the chunk {chunk_id!a} declares {chunk_size_bytes} bytes, beyond the file's end"
            )
        if chunk_id == FORMAT_CHUNK_ID:
            format_chunk = parse_format_chunk(content[body_offset : body_offset + chunk_size_bytes])
        elif chunk_id == DATA_CHUNK_ID:
            return judge_data_chunk(format_chunk, body_offset, chunk_size_bytes)
        offset = body_offset + chunk_size_bytes + chunk_size_bytes % 2
    raise WavFormatError("the file has no data chunk")


def parse_format_chunk(body: bytes) -> FormatChunk:
    if len(body) < FORMAT_CHUNK_MIN_SIZE_BYTES:
        raise WavFormatError(f"the format chunk holds {len(body)} bytes, too few")
    return FormatChunk(
        format_code=read_little_endian(body, 0, 2),
        channel_count=read_little_endian(body, 2, 2),
        sample_rate_hz=read_little_endian(body, 4, 4),
        block_size_bytes=read_little_endian(body, 12, 2),
        bits_per_sample=read_little_endian(body, 14, 2),
    )


def judge_data_chunk(
    format_chunk: FormatChunk | None, data_offset: int, data_size_bytes: int
) -> WavLayout:
    if format_chunk is None:
        raise WavFormatError("the data chunk comes before any format chunk")
    sample_format = find_sample_format(format_chunk)
    if format_chunk.channel_count != CHANNEL_COUNT:
        raise WavFormatError(f"the file has {format_chunk.channel_count} channels, not one")
    if format_chunk.sample_rate_hz != SAMPLE_RATE_HZ:
        raise WavFormatError(
            f"the file is sampled at {format_chunk.sample_rate_hz} Hz, not {SAMPLE_RATE_HZ} Hz"
        )
    if format_chunk.block_size_bytes != sample_format.sample_size_bytes:
        raise WavFormatError(
            f"a block holds {format_chunk.block_size_bytes} bytes, "
            f"not the {sample_format.sample_size_bytes} of one sample"
        )
    if data_size_bytes % sample_format.sample_size_bytes:
        raise WavFormatError(
            f"the data chunk holds {data_size_bytes} bytes, not a whole count of samples"
        )
    return WavLayout(sample_format, data_offset, data_size_bytes)


def find_sample_format(format_chunk: FormatChunk) -> SampleFormat:
    for sample_format in SampleFormat:
        if sample_format.value == (format_chunk.format_code, format_chunk.bits_per_sample):
            return sample_format
    raise WavFormatError(
        f"the format {format_chunk.format_code} with {format_chunk.bits_per_sample} bits "
        "per sample is neither 16-bit integers nor 32-bit floats"
    )
