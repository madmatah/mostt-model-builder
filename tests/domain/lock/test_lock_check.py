import re
from collections.abc import Callable
from functools import partial

import pytest

from mostt_model_builder.domain.lock.lock_check import (
    CheckedLock,
    LockedPackage,
    LockError,
    check_lock,
    is_installable_wheel,
)

ROOT = "mostt-model-builder-images"
HASH = "sha256:" + "0" * 64
FILES = "https://files.pythonhosted.org/packages/00/00"
ROOT_ENTRY = f'[[package]]\nname = "{ROOT}"\nversion = "0.1.0"\nsource = {{ virtual = "." }}\n'


def build_package(
    name: str,
    wheels: list[str],
    source: str = '{ registry = "https://pypi.org/simple" }',
    sdist: str | None = None,
    url_prefix: str = FILES,
    file_hash: str = HASH,
) -> str:
    wheel_entries = ", ".join(
        f'{{ url = "{url_prefix}/{wheel}", hash = "{file_hash}" }}' for wheel in wheels
    )
    sdist_line = (
        "" if sdist is None else f'sdist = {{ url = "{FILES}/{sdist}", hash = "{HASH}" }}\n'
    )
    return (
        f'[[package]]\nname = "{name}"\nversion = "1.0"\nsource = {source}\n'
        f"{sdist_line}wheels = [{wheel_entries}]\n"
    )


def build_lock(*packages: str, version: int = 1, manifest: str = "") -> str:
    return f"version = {version}\nrevision = 3\n{manifest}\n" + "\n".join((ROOT_ENTRY, *packages))


ANTLR4_METADATA = (
    '[[manifest.dependency-metadata]]\nname = "antlr4-python3-runtime"\nversion = "4.9.3"\n'
)
INSTALLABLE_WHEELS = {
    "pure python for 2 and 3": "six-1.0-py2.py3-none-any.whl",
    "the stable abi from 3.11 on manylinux": (
        "pkg-1.0-cp311-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl"
    ),
    "cpython 3.12 on manylinux 2014": "pkg-1.0-cp312-cp312-manylinux2014_x86_64.whl",
    "a build tag": "pkg-1.0-1-py3-none-any.whl",
    "pure python for 3.12": "multiprocess-0.70.19-py312-none-any.whl",
    "pure python for 3.10": "multiprocess-0.70.19-py310-none-any.whl",
    "manylinux at glibc 2.36, the glibc of the images' base": (
        "pkg-1.0-cp312-cp312-manylinux_2_36_x86_64.whl"
    ),
    "manylinux 1, read as glibc 2.5": "pkg-1.0-cp312-cp312-manylinux1_x86_64.whl",
    "manylinux 2010, read as glibc 2.12": "pkg-1.0-cp312-cp312-manylinux2010_x86_64.whl",
}
UNINSTALLABLE_WHEELS = {
    "musllinux only": "pkg-1.0-cp312-cp312-musllinux_1_2_x86_64.whl",
    "the stable abi from 3.13 on": "pkg-1.0-cp313-abi3-manylinux_2_17_x86_64.whl",
    "pure python for 3.13": "multiprocess-0.70.19-py313-none-any.whl",
    "a python 3.12 tag with an abi": "pkg-1.0-py312-cp312-any.whl",
    "cpython 3.11 without the stable abi": "pkg-1.0-cp311-cp311-manylinux_2_17_x86_64.whl",
    "arm": "pkg-1.0-cp312-cp312-manylinux_2_17_aarch64.whl",
    "windows": "pkg-1.0-py3-none-win_amd64.whl",
    "not a wheel": "pkg-1.0.tar.gz",
    "too few parts": "pkg-py3-none-any.whl",
    "manylinux beyond glibc 2.36": "pkg-1.0-cp312-cp312-manylinux_2_37_x86_64.whl",
    "manylinux at a glibc 3": "pkg-1.0-cp312-cp312-manylinux_3_0_x86_64.whl",
}
REFUSED_LOCKS = {
    "an array nested 3 000 deep": ("value = " + "[" * 3000, "the lock is not valid TOML"),
    "a lock version other than 1": (build_lock(version=2), "the lock's version is 2, not 1"),
    "another registry": (
        build_lock(
            build_package(
                "torch",
                ["torch-1.0-py3-none-any.whl"],
                source='{ registry = "https://download.pytorch.org/whl" }',
            )
        ),
        "torch 1.0 comes from {'registry': 'https://download.pytorch.org/whl'}",
    ),
    "a git source": (
        build_lock(build_package("nemo", [], source='{ git = "https://github.com/x/y" }')),
        "nemo 1.0 comes from {'git':",
    ),
    "a path source": (
        build_lock(build_package("local", [], source='{ path = "wheels/local.whl" }')),
        "local 1.0 comes from {'path':",
    ),
    "a directory source": (
        build_lock(build_package("local", [], source='{ directory = "local" }')),
        "local 1.0 comes from {'directory':",
    ),
    "an editable source": (
        build_lock(build_package("local", [], source='{ editable = "local" }')),
        "local 1.0 comes from {'editable':",
    ),
    "a virtual source on another package": (
        build_lock(build_package("other", [], source='{ virtual = "." }')),
        "other 1.0 comes from {'virtual': '.'}",
    ),
    "a distribution without its hash": (
        build_lock(build_package("six", ["six-1.0-py3-none-any.whl"], file_hash="md5:00")),
        "six 1.0 has a distribution without its SHA-256",
    ),
    "a distribution outside files.pythonhosted.org": (
        build_lock(
            build_package("six", ["six-1.0-py3-none-any.whl"], url_prefix="https://example.org")
        ),
        "six 1.0 has a distribution at 'https://example.org/six-1.0-py3-none-any.whl'",
    ),
    "a package without a wheel that installs here": (
        build_lock(build_package("antlr4-python3-runtime", [], sdist="antlr4.tar.gz")),
        "antlr4-python3-runtime 1.0 has no wheel that installs on Python 3.12",
    ),
    "a package whose only wheels are musllinux": (
        build_lock(build_package("pkg", [UNINSTALLABLE_WHEELS["musllinux only"]])),
        "pkg 1.0 has no wheel that installs",
    ),
    "a package whose only wheels need python 3.13": (
        build_lock(build_package("pkg", [UNINSTALLABLE_WHEELS["the stable abi from 3.13 on"]])),
        "pkg 1.0 has no wheel that installs",
    ),
    "a lock without its root": (
        "version = 1\n" + build_package("six", ["six-1.0-py3-none-any.whl"]),
        "the lock holds the root mostt-model-builder-images 0 times, not once",
    ),
}


class TestTheLockCheck:
    class TestTheCheckLockFunction:
        class TestWhenEveryPackageHasAWheelThatInstallsHere:
            @pytest.fixture
            def checked_lock(self) -> CheckedLock:
                return check_lock(
                    build_lock(
                        *(
                            build_package(f"package-{wheel_index}", [wheel])
                            for wheel_index, wheel in enumerate(INSTALLABLE_WHEELS.values())
                        )
                    ),
                    ROOT,
                )

            def test_should_list_every_package_but_the_root(
                self, checked_lock: CheckedLock
            ) -> None:
                assert checked_lock.packages == tuple(
                    LockedPackage(f"package-{wheel_index}", "1.0")
                    for wheel_index in range(len(INSTALLABLE_WHEELS))
                )

        class TestWhenTheLockHoldsOnlyItsRootWithoutAWheel:
            @pytest.fixture
            def checked_lock(self) -> CheckedLock:
                return check_lock(build_lock(), ROOT)

            def test_should_accept_the_virtual_root(self, checked_lock: CheckedLock) -> None:
                assert checked_lock == CheckedLock(ROOT, ())

        class TestWhenTheDependencyMetadataNamesAPackageWithoutAWheel:
            @pytest.fixture
            def checked_lock(self) -> CheckedLock:
                return check_lock(
                    build_lock(
                        build_package("antlr4-python3-runtime", [], sdist="antlr4.tar.gz"),
                        manifest=ANTLR4_METADATA,
                    ),
                    ROOT,
                )

            def test_should_accept_it_since_a_build_stage_builds_it(
                self, checked_lock: CheckedLock
            ) -> None:
                assert checked_lock.packages == (LockedPackage("antlr4-python3-runtime", "1.0"),)

        @pytest.mark.parametrize(
            ("text", "reason"), list(REFUSED_LOCKS.values()), ids=list(REFUSED_LOCKS.keys())
        )
        class TestWhenTheLockBreaksARule:
            @pytest.fixture
            def check_act(self, text: str) -> Callable[[], CheckedLock]:
                return partial(check_lock, text, ROOT)

            def test_should_refuse_it_and_say_why(
                self, check_act: Callable[[], CheckedLock], reason: str
            ) -> None:
                with pytest.raises(LockError, match=re.escape(reason)):
                    check_act()

    class TestTheIsInstallableWheelFunction:
        @pytest.mark.parametrize(
            "file_name", list(INSTALLABLE_WHEELS.values()), ids=list(INSTALLABLE_WHEELS.keys())
        )
        class TestWhenATagFitsPython312OnLinuxX8664:
            @pytest.fixture
            def is_installable(self, file_name: str) -> bool:
                return is_installable_wheel(file_name)

            def test_should_accept_it(self, is_installable: bool) -> None:
                assert is_installable is True

        @pytest.mark.parametrize(
            "file_name", list(UNINSTALLABLE_WHEELS.values()), ids=list(UNINSTALLABLE_WHEELS.keys())
        )
        class TestWhenNoTagFitsPython312OnLinuxX8664:
            @pytest.fixture
            def is_installable(self, file_name: str) -> bool:
                return is_installable_wheel(file_name)

            def test_should_refuse_it(self, is_installable: bool) -> None:
                assert is_installable is False
