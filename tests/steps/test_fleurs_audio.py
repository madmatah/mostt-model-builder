import io
import re
import tarfile
from collections.abc import Callable, Sequence
from functools import partial
from pathlib import Path

import pytest

from mostt_model_builder.steps.fleurs_audio import FleursAudioError, read_recordings

FIRST_RECORDING = "7105.wav"
SECOND_RECORDING = "8412.wav"
OUTSIDE_THE_INDEX = "9999.wav"
INDEXED_RECORDINGS = (FIRST_RECORDING, SECOND_RECORDING)


def build_archive(path: Path, members: Sequence[tuple[str, bytes | None]]) -> Path:
    with tarfile.open(path, "w:gz") as archive:
        for name, content in members:
            member = tarfile.TarInfo(name)
            if content is None:
                member.type = tarfile.DIRTYPE
                archive.addfile(member)
                continue
            member.size = len(content)
            archive.addfile(member, io.BytesIO(content))
    return path


REFUSED_ARCHIVES = {
    "a recording missing": (
        [(f"test/{FIRST_RECORDING}", b"first")],
        "1 recordings of the index are missing from test.tar.gz, such as 8412.wav",
    ),
    "a recording twice": (
        [
            (f"test/{FIRST_RECORDING}", b"first"),
            (f"test/{FIRST_RECORDING}", b"again"),
            (f"test/{SECOND_RECORDING}", b"second"),
        ],
        "the archive holds test/7105.wav twice",
    ),
    "a directory under a recording's name": (
        [(f"test/{FIRST_RECORDING}", None), (f"test/{SECOND_RECORDING}", b"second")],
        "the member test/7105.wav is not a regular file",
    ),
}


class TestTheFleursAudio:
    class TestTheReadRecordingsFunction:
        class TestWhenTheArchiveHoldsTheIndexedRecordingsAndOthers:
            @pytest.fixture
            def recordings(self, tmp_path: Path) -> list[tuple[str, bytes]]:
                archive = build_archive(
                    tmp_path / "test.tar.gz",
                    [
                        (f"test/{FIRST_RECORDING}", b"first"),
                        (f"test/{OUTSIDE_THE_INDEX}", b"outside"),
                        (f"other/{SECOND_RECORDING}", b"elsewhere"),
                        (f"test/{SECOND_RECORDING}", b"second"),
                    ],
                )
                return list(read_recordings(archive, INDEXED_RECORDINGS))

            def test_should_read_only_the_indexed_recordings_of_the_test_directory(
                self, recordings: list[tuple[str, bytes]]
            ) -> None:
                assert recordings == [(FIRST_RECORDING, b"first"), (SECOND_RECORDING, b"second")]

        @pytest.mark.parametrize(
            ("members", "reason"),
            list(REFUSED_ARCHIVES.values()),
            ids=list(REFUSED_ARCHIVES.keys()),
        )
        class TestWhenTheArchiveDoesNotHoldEachRecordingOnce:
            @pytest.fixture
            def read_act(
                self, tmp_path: Path, members: Sequence[tuple[str, bytes | None]]
            ) -> Callable[[], list[tuple[str, bytes]]]:
                archive = build_archive(tmp_path / "test.tar.gz", members)
                return partial(list, read_recordings(archive, INDEXED_RECORDINGS))

            def test_should_refuse_it_and_say_why(
                self, read_act: Callable[[], list[tuple[str, bytes]]], reason: str
            ) -> None:
                with pytest.raises(FleursAudioError, match=re.escape(reason)):
                    read_act()
