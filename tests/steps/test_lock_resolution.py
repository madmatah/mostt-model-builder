import sys
from pathlib import Path

import pytest

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
    (request / "pyproject.toml").write_bytes(MANIFEST)
    if has_lock:
        (request / "uv.lock").write_bytes(COMMITTED_LOCK)
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
            def resolved(self, tmp_path: Path) -> tuple[str, bytes]:
                request = write_request(tmp_path, has_lock=True)
                (tmp_path / "output").mkdir()
                resolve_lock(
                    request,
                    tmp_path / "project",
                    tmp_path / "output",
                    [sys.executable, "-I", "-c", FAKE_UV],
                )
                return (
                    (tmp_path / "output" / "uv.lock").read_text(),
                    (tmp_path / "project" / "pyproject.toml").read_bytes(),
                )

            def test_should_let_uv_update_the_committed_lock_and_copy_the_new_one_into_the_output(
                self, resolved: tuple[str, bytes]
            ) -> None:
                assert resolved[0] == COMMITTED_LOCK.decode() + RESOLVED_LINE

            def test_should_work_on_a_copy_of_the_manifest(
                self, resolved: tuple[str, bytes]
            ) -> None:
                assert resolved[1] == MANIFEST
