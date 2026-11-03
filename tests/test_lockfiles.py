import tomllib
from pathlib import Path

import pytest

from mostt_model_builder.domain.lock.lock_check import (
    HOST_ROOT_NAME,
    IMAGES_ROOT_NAME,
    CheckedLock,
    check_lock,
)
from mostt_model_builder.domain.lock.lock_freshness import check_lock_freshness

REPO_ROOT = Path(__file__).resolve().parent.parent
LOCKS = {
    "the host's lock": (REPO_ROOT / "uv.lock", HOST_ROOT_NAME),
    "the images' lock": (REPO_ROOT / "images" / "uv.lock", IMAGES_ROOT_NAME),
}
IMAGES_MANIFEST_PATH = REPO_ROOT / "images" / "pyproject.toml"
IMAGES_LOCK_PATH = REPO_ROOT / "images" / "uv.lock"


def count_packages_besides_the_root(lock_path: Path) -> int:
    return len(tomllib.loads(lock_path.read_text(encoding="utf-8"))["package"]) - 1


def read_manifest_extras() -> list[str]:
    manifest = tomllib.loads(IMAGES_MANIFEST_PATH.read_text(encoding="utf-8"))
    return list(manifest["project"].get("optional-dependencies", {}))


class TestTheLockfiles:
    @pytest.mark.parametrize(
        ("lock_path", "root_name"), list(LOCKS.values()), ids=list(LOCKS.keys())
    )
    class TestWhenACommittedLockIsChecked:
        @pytest.fixture
        def checked_lock(self, lock_path: Path, root_name: str) -> CheckedLock:
            return check_lock(lock_path.read_text(encoding="utf-8"), root_name)

        def test_should_take_every_package_from_pypi_by_hash_with_a_wheel_that_installs_here(
            self, checked_lock: CheckedLock, lock_path: Path
        ) -> None:
            assert len(checked_lock.packages) == count_packages_besides_the_root(lock_path)

    class TestWhenTheImagesLockIsComparedWithItsManifest:
        @pytest.fixture
        def fresh_extras(self) -> tuple[str, ...]:
            return check_lock_freshness(
                IMAGES_MANIFEST_PATH.read_text(encoding="utf-8"),
                IMAGES_LOCK_PATH.read_text(encoding="utf-8"),
                IMAGES_ROOT_NAME,
            )

        def test_should_find_it_fresh_with_every_extra_since_no_host_command_may_resolve_it_again(
            self, fresh_extras: tuple[str, ...]
        ) -> None:
            assert fresh_extras == tuple(sorted(read_manifest_extras()))
