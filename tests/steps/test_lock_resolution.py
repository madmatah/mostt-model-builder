import sys
from pathlib import Path

import pytest

from mostt_model_builder.domain.lock.lock_check import LOCK_FILE_NAME, MANIFEST_FILE_NAME
from mostt_model_builder.steps.lock_resolution import build_lock_arguments, resolve_lock

MANIFEST = b'[project]\nname = "mostt-model-builder-images"\n'
COMMITTED_LOCK = b"version = 1\n"
RESOLVED_LINE = "revision = 3\n"
FAKE_UV = (
    "from pathlib import Path\n"
    "lock = Path('uv.lock')\n"
    f"lock.write_text(lock.read_text() + {RESOLVED_LINE!r})\n"
)


def write_request(directory: Path, has_lock: bool) -> Path:
    request = directory / "request"
    request.mkdir()
    (request / MANIFEST_FILE_NAME).write_bytes(MANIFEST)
    if has_lock:
        (request / LOCK_FILE_NAME).write_bytes(COMMITTED_LOCK)
    return request


class TestTheLockResolution:
    class TestTheBuildLockArgumentsFunction:
        class TestWhenGivenTheUvOfTheImage:
            @pytest.fixture
            def lock_arguments(self) -> list[str]:
                return build_lock_arguments("/usr/local/bin/uv", Path("/tmp/uv-cache"))

            def test_should_lock_with_the_cache_of_uv_under_tmp(
                self, lock_arguments: list[str]
            ) -> None:
                assert lock_arguments == [
                    "/usr/local/bin/uv",
                    "lock",
                    "--cache-dir",
                    "/tmp/uv-cache",
                ]

    class TestTheResolveLockFunction:
        class TestWhenUvWritesANewLock:
            @pytest.fixture
            def resolved(self, tmp_path: Path) -> tuple[str, bytes, list[str], list[str]]:
                request = write_request(tmp_path, has_lock=True)
                (tmp_path / "output").mkdir()
                resolve_lock(
                    request,
                    tmp_path / "project",
                    tmp_path / "output",
                    [sys.executable, "-I", "-c", FAKE_UV],
                )
                return (
                    (tmp_path / "output" / LOCK_FILE_NAME).read_text(),
                    (tmp_path / "project" / "pyproject.toml").read_bytes(),
                    sorted(path.name for path in (tmp_path / "project").iterdir()),
                    sorted(path.name for path in (tmp_path / "output").iterdir()),
                )

            def test_should_let_uv_update_the_committed_lock_and_copy_the_new_one_into_the_output(
                self, resolved: tuple[str, bytes, list[str], list[str]]
            ) -> None:
                assert resolved[0] == COMMITTED_LOCK.decode() + RESOLVED_LINE

            def test_should_give_uv_a_copy_of_the_manifest_under_the_name_it_finds_a_project_by(
                self, resolved: tuple[str, bytes, list[str], list[str]]
            ) -> None:
                assert resolved[1] == MANIFEST

            def test_should_write_the_new_lock_under_a_name_that_no_uv_on_the_host_reads(
                self, resolved: tuple[str, bytes, list[str], list[str]]
            ) -> None:
                assert resolved[3] == ["environment.lock"]

            def test_should_leave_the_project_only_the_files_uv_reads(
                self, resolved: tuple[str, bytes, list[str], list[str]]
            ) -> None:
                assert resolved[2] == ["pyproject.toml", "uv.lock"]
