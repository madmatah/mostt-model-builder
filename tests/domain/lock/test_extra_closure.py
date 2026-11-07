import re
from collections.abc import Callable
from functools import partial
from pathlib import Path

import pytest

from mostt_model_builder.domain.lock.extra_closure import (
    LockedDistribution,
    LockedEntry,
    close_extra,
)
from mostt_model_builder.domain.lock.lock_check import LockError

ROOT = "mostt-model-builder-images"
HASH = "sha256:" + "0" * 64
FILES = "https://files.pythonhosted.org/packages/00/00"
IMAGES_LOCK_PATH = Path(__file__).resolve().parents[3] / "images" / "environment.lock"
EXTRA_REQUIREMENTS = {
    "core-hub": {"huggingface-hub"},
    "core-onnx-convert": {"ml-dtypes", "numpy", "onnx", "onnx-ir", "onnxruntime"},
    "core-onnx-inspect": {"onnx"},
    "parakeet-tdt-export": {"nemo-toolkit", "numpy", "onnx", "safetensors", "torch"},
    "parakeet-tdt-transcribe-cpu": {"numpy", "onnx-asr", "onnxruntime"},
    "parakeet-tdt-transcribe-gpu": {"numpy", "onnx-asr", "onnxruntime-gpu"},
    "parakeet-tdt-transformers-reference": {"librosa", "numpy", "safetensors", "torch"},
}


def build_package(
    name: str, version: str = "1.0", extra_lines: str = "", optional_lines: str = ""
) -> str:
    return (
        f'[[package]]\nname = "{name}"\nversion = "{version}"\n'
        'source = { registry = "https://pypi.org/simple" }\n'
        f"{extra_lines}"
        f'wheels = [{{ url = "{FILES}/{name}-{version}-py3-none-any.whl", hash = "{HASH}" }}]\n'
        f"{optional_lines}"
    )


def build_lock(root_lines: str, *packages: str) -> str:
    root = f'[[package]]\nname = "{ROOT}"\nversion = "0.1.0"\nsource = {{ virtual = "." }}\n'
    return "version = 1\n" + "\n".join((root + root_lines, *packages))


def build_entry(name: str, version: str = "1.0") -> LockedEntry:
    return LockedEntry(
        name,
        version,
        (LockedDistribution(f"{FILES}/{name}-{version}-py3-none-any.whl", HASH),),
    )


ROOT_OF_EXTRAS = (
    "[package.optional-dependencies]\n"
    'image = [{ name = "runtime" }]\n'
    'other = [{ name = "unrelated" }]\n'
)
LOCK_FORMS = {
    "a dependency of a dependency": (
        build_lock(
            ROOT_OF_EXTRAS,
            build_package("runtime", extra_lines='dependencies = [{ name = "numpy" }]\n'),
            build_package("numpy"),
            build_package("unrelated"),
        ),
        (build_entry("numpy"), build_entry("runtime")),
    ),
    "a dependency with an extra, which brings that extra's dependencies": (
        build_lock(
            ROOT_OF_EXTRAS,
            build_package(
                "runtime",
                extra_lines='dependencies = [{ name = "fsspec", extra = ["http"] }]\n',
            ),
            build_package(
                "fsspec",
                optional_lines='[package.optional-dependencies]\nhttp = [{ name = "aiohttp" }]\n'
                'ssh = [{ name = "paramiko" }]\n',
            ),
            build_package("aiohttp"),
            build_package("paramiko"),
            build_package("unrelated"),
        ),
        (build_entry("aiohttp"), build_entry("fsspec"), build_entry("runtime")),
    ),
    "a dependency under a marker, which counts whatever the marker says": (
        build_lock(
            ROOT_OF_EXTRAS,
            build_package(
                "runtime",
                extra_lines='dependencies = [{ name = "pywin", marker = "sys_platform == '
                "'win32'\" }]\n",
            ),
            build_package("pywin"),
            build_package("unrelated"),
        ),
        (build_entry("pywin"), build_entry("runtime")),
    ),
    "two versions of one name, both taken": (
        build_lock(
            ROOT_OF_EXTRAS,
            build_package("runtime", "1.0"),
            build_package("runtime", "2.0"),
            build_package("unrelated"),
        ),
        (build_entry("runtime", "1.0"), build_entry("runtime", "2.0")),
    ),
}


class TestTheExtraClosure:
    class TestTheCloseExtraFunction:
        @pytest.mark.parametrize(
            ("lock_text", "expected_closure"),
            list(LOCK_FORMS.values()),
            ids=list(LOCK_FORMS.keys()),
        )
        class TestWhenTheLockHoldsOneForm:
            @pytest.fixture
            def closure(self, lock_text: str) -> tuple[LockedEntry, ...]:
                return close_extra(lock_text, "image")

            def test_should_hold_every_package_the_extra_can_reach_and_no_other(
                self,
                closure: tuple[LockedEntry, ...],
                expected_closure: tuple[LockedEntry, ...],
            ) -> None:
                assert closure == expected_closure

        class TestWhenTheImageHasNoExtra:
            @pytest.fixture
            def closure(self) -> tuple[LockedEntry, ...]:
                return close_extra(build_lock(ROOT_OF_EXTRAS), None)

            def test_should_hold_no_package(self, closure: tuple[LockedEntry, ...]) -> None:
                assert closure == ()

        class TestWhenTheLockLacksTheExtra:
            @pytest.fixture
            def close_act(self) -> Callable[[], tuple[LockedEntry, ...]]:
                lock_text = build_lock(
                    ROOT_OF_EXTRAS, build_package("runtime"), build_package("unrelated")
                )
                return partial(close_extra, lock_text, "missing")

            def test_should_refuse_it(
                self, close_act: Callable[[], tuple[LockedEntry, ...]]
            ) -> None:
                with pytest.raises(
                    LockError,
                    match=re.escape(f"the lock holds no extra missing of {ROOT}"),
                ):
                    close_act()

        class TestWhenTheLockFailsItsCheck:
            @pytest.fixture
            def close_act(self) -> Callable[[], tuple[LockedEntry, ...]]:
                lock_text = build_lock(
                    ROOT_OF_EXTRAS,
                    build_package("runtime").replace(FILES, "https://example.org"),
                    build_package("unrelated"),
                )
                return partial(close_extra, lock_text, "image")

            def test_should_refuse_it_as_outside_input(
                self, close_act: Callable[[], tuple[LockedEntry, ...]]
            ) -> None:
                with pytest.raises(LockError, match=re.escape("the lock is refused")):
                    close_act()

        class TestWhenEachExtraOfTheImagesLockIsClosed:
            @pytest.fixture(scope="class")
            @classmethod
            def closed_names(cls) -> dict[str, set[str]]:
                lock_text = IMAGES_LOCK_PATH.read_text(encoding="utf-8")
                return {
                    extra: {entry.name for entry in close_extra(lock_text, extra)}
                    for extra in EXTRA_REQUIREMENTS
                }

            def test_should_hold_the_declared_requirements_of_each_extra(
                self, closed_names: dict[str, set[str]]
            ) -> None:
                mismatches = [
                    extra
                    for extra, requirements in EXTRA_REQUIREMENTS.items()
                    if not requirements <= closed_names[extra]
                ]
                assert len(closed_names) == len(EXTRA_REQUIREMENTS)
                assert mismatches == []

            def test_should_keep_the_hub_image_free_of_the_libraries_of_a_model(
                self, closed_names: dict[str, set[str]]
            ) -> None:
                assert {"torch", "nemo-toolkit", "transformers"} & closed_names["core-hub"] == set()
