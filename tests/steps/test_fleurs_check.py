import io
import json
import re
import struct
import tarfile
from collections.abc import Callable
from functools import partial
from pathlib import Path, PurePosixPath

import pytest

from mostt_model_builder.domain.bench.fleurs_check_request import (
    BenchFiles,
    FleursCheckRequest,
    render_fleurs_check_request,
)
from mostt_model_builder.steps.fleurs_check import FleursCheckError, check_fleurs

FIRST_RECORDING = "7105.wav"
SECOND_RECORDING = "8412.wav"
CONTAINER_INDEX = PurePosixPath("/sources/google/fleurs/test.tsv")
CONTAINER_ARCHIVE = PurePosixPath("/sources/google/fleurs/test.tar.gz")
FIRST_SAMPLES = 160
SECOND_SAMPLES = 320


def build_wav(sample_count: int) -> bytes:
    format_body = struct.pack("<HHIIHH", 3, 1, 16_000, 64_000, 4, 32)
    samples = b"\0" * (sample_count * 4)
    body = (
        b"WAVE"
        + b"fmt "
        + struct.pack("<I", len(format_body))
        + format_body
        + b"data"
        + struct.pack("<I", len(samples))
        + samples
    )
    return b"RIFF" + struct.pack("<I", len(body)) + body


def build_index_line(recording_name: str, sample_count: int) -> str:
    return f"1829\t{recording_name}\tRaw.\traw\tr a w\t{sample_count}\tFEMALE\n"


def write_bench(directory: Path, indexed_first_samples: int) -> Path:
    sources = directory / "sources" / "google" / "fleurs"
    sources.mkdir(parents=True)
    (sources / "test.tsv").write_text(
        build_index_line(FIRST_RECORDING, indexed_first_samples)
        + build_index_line(SECOND_RECORDING, SECOND_SAMPLES)
    )
    with tarfile.open(sources / "test.tar.gz", "w:gz") as archive:
        for name, sample_count in (
            (FIRST_RECORDING, FIRST_SAMPLES),
            (SECOND_RECORDING, SECOND_SAMPLES),
        ):
            content = build_wav(sample_count)
            member = tarfile.TarInfo(f"test/{name}")
            member.size = len(content)
            archive.addfile(member, io.BytesIO(content))
    request_directory = directory / "request"
    request_directory.mkdir()
    (request_directory / "request.json").write_bytes(
        render_fleurs_check_request(
            FleursCheckRequest((BenchFiles("fr_fr", CONTAINER_INDEX, CONTAINER_ARCHIVE),))
        )
    )
    (directory / "output").mkdir()
    return directory


class TestTheFleursCheck:
    class TestTheCheckFleursFunction:
        class TestWhenEveryIndexedRecordingIsInTheArchive:
            @pytest.fixture
            def report(self, tmp_path: Path) -> object:
                bench = write_bench(tmp_path, FIRST_SAMPLES)
                check_fleurs(bench / "request", bench / "output", bench)
                return json.loads((bench / "output" / "fleurs-check.json").read_text())

            def test_should_report_each_recordings_duration_in_the_order_of_the_index(
                self, report: object
            ) -> None:
                assert report == {
                    "languages": [
                        {
                            "language": "fr_fr",
                            "utterances": [
                                {
                                    "recording_name": FIRST_RECORDING,
                                    "duration_samples": FIRST_SAMPLES,
                                },
                                {
                                    "recording_name": SECOND_RECORDING,
                                    "duration_samples": SECOND_SAMPLES,
                                },
                            ],
                        }
                    ]
                }

        class TestWhenARecordingHoldsAnotherCountOfSamplesThanItsIndex:
            @pytest.fixture
            def check_act(self, tmp_path: Path) -> Callable[[], object]:
                bench = write_bench(tmp_path, FIRST_SAMPLES + 1)
                return partial(check_fleurs, bench / "request", bench / "output", bench)

            def test_should_refuse_the_bench(self, check_act: Callable[[], object]) -> None:
                with pytest.raises(
                    FleursCheckError,
                    match=re.escape(
                        "the recording 7105.wav holds 160 samples, where its index gives 161"
                    ),
                ):
                    check_act()
