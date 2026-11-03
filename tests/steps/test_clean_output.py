import os
import stat
from pathlib import Path

import pytest

from mostt_model_builder.steps.clean_output import empty_directory

KEPT_DIRECTORY_MODE = 0o550


class TestTheCleanOutput:
    class TestTheEmptyDirectoryFunction:
        class TestWhenAStepLeftFilesLockedDirectoriesAndLinksToADirectory:
            @pytest.fixture
            def entries_left(self, tmp_path: Path) -> tuple[list[str], list[str], int]:
                output = tmp_path / "output"
                (output / "nested" / "deeper").mkdir(parents=True)
                (output / "nested" / "deeper" / "file").write_bytes(b"x")
                (output / "peaks.json").write_bytes(b"{}")
                kept = tmp_path / "kept"
                kept.mkdir()
                (kept / "file").write_bytes(b"kept")
                kept.chmod(KEPT_DIRECTORY_MODE)
                (output / "link").symlink_to(kept)
                (output / "nested" / "link").symlink_to(kept)
                (output / "nested" / "deeper").chmod(0o500)
                (output / "nested").chmod(0o500)
                empty_directory(output)
                return (
                    sorted(os.listdir(output)) + sorted(os.listdir(tmp_path)),
                    sorted(os.listdir(kept)),
                    stat.S_IMODE(kept.stat().st_mode),
                )

            def test_should_leave_the_directory_empty(
                self, entries_left: tuple[list[str], list[str], int]
            ) -> None:
                assert entries_left[0] == ["kept", "output"]

            def test_should_follow_no_link_so_the_directory_it_targets_keeps_its_files_and_mode(
                self, entries_left: tuple[list[str], list[str], int]
            ) -> None:
                assert entries_left[1:] == (["file"], KEPT_DIRECTORY_MODE)
