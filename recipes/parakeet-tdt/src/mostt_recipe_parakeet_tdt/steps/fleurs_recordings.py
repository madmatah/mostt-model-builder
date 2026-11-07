from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy

from mostt_model_builder.domain.bench.fleurs_index import (
    FLEURS_REPOSITORY,
    locate_bench_files,
    parse_fleurs_index,
)
from mostt_model_builder.domain.bench.wav_format import SampleFormat, parse_wav_layout
from mostt_model_builder.domain.contract.shard_request import select_shard
from mostt_model_builder.steps.bound_sources import find_bound_source
from mostt_model_builder.steps.fleurs_audio import read_recordings

# NumPy is installed in the images of the recipe alone, so mypy on the host knows nothing of its
# types.
type NumpyArray = Any

INTEGER16_FULL_SCALE = 32768.0


@dataclass(frozen=True, slots=True)
class Recording:
    language: str
    utterance_id: str
    samples: NumpyArray


def decode_samples(content: bytes) -> NumpyArray:
    layout = parse_wav_layout(content)
    samples = content[layout.data_offset : layout.data_offset + layout.data_size_bytes]
    if layout.sample_format is SampleFormat.FLOAT32:
        return numpy.frombuffer(samples, dtype="<f4").astype(numpy.float32)
    return numpy.frombuffer(samples, dtype="<i2").astype(numpy.float32) / INTEGER16_FULL_SCALE


def list_utterance_ids(language: str) -> list[str]:
    index_path, _ = (
        find_bound_source(FLEURS_REPOSITORY, path) for path in locate_bench_files(language)
    )
    utterances = parse_fleurs_index(index_path.read_text(encoding="utf-8"))
    return [utterance.recording_name for utterance in utterances]


def read_language_recordings(language: str, utterance_ids: Sequence[str]) -> list[Recording]:
    _, archive_path = (
        find_bound_source(FLEURS_REPOSITORY, path) for path in locate_bench_files(language)
    )
    samples_by_utterance = {
        utterance_id: decode_samples(content)
        for utterance_id, content in read_recordings(archive_path, utterance_ids)
    }
    return [
        Recording(language, utterance_id, samples_by_utterance[utterance_id])
        for utterance_id in utterance_ids
    ]


# Dealing the utterances of every language as one list gives each shard its share of each language.
def read_shard_recordings(
    languages: Sequence[str], shard: int, shard_count: int
) -> list[Recording]:
    utterances = [
        (language, utterance_id)
        for language in languages
        for utterance_id in list_utterance_ids(language)
    ]
    selected = select_shard(utterances, shard, shard_count)
    recordings: list[Recording] = []
    for language in languages:
        recordings.extend(
            read_language_recordings(
                language,
                [
                    utterance_id
                    for selected_language, utterance_id in selected
                    if selected_language == language
                ],
            )
        )
    return recordings
