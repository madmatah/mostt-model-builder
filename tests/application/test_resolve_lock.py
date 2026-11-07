import contextlib
import re
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from functools import partial
from pathlib import PurePosixPath

import pytest
from run_step_fakes import (
    GROUP_ID,
    USER_ID,
    FakeContainerEngine,
    FakeStepFiles,
    build_step_output,
)

from mostt_model_builder.application.resolve_lock import LockRequest, LockResolution, resolve_lock
from mostt_model_builder.application.run_step import HostIdentity, ImageFacts
from mostt_model_builder.domain.containers.account_ranges import HostAccounts
from mostt_model_builder.domain.containers.container_files import ContainerFileError
from mostt_model_builder.domain.containers.container_steps import LOCK_ROW, ContainerRun
from mostt_model_builder.domain.containers.step_peaks import (
    PEAKS_FILE_NAME,
    StepPeaks,
    render_step_peaks,
)
from mostt_model_builder.domain.lock.lock_check import LockedPackage, LockError
from mostt_model_builder.domain.lock.lock_freshness import LockFreshnessError

WORK_ROOT = PurePosixPath("/data/builds/work")
CLONE_LOCK_PATH = PurePosixPath("/opt/clones/mostt-model-builder/images/environment.lock")
HEAD_TREE = "3f1d1ffe79bc3336eb583331cfbf9b189fd45f01"
IMAGE_ID = "sha256:" + "e" * 64
ROOT = "mostt-model-builder-images"
HASH = "sha256:" + "0" * 64
FILES = "https://files.pythonhosted.org/packages/00/00"
HOST = HostIdentity(USER_ID, GROUP_ID, 20, HostAccounts(False, (), ()))
MANIFEST = f"""[project]
name = "{ROOT}"
version = "0.1.0"
requires-python = "==3.12.*"
dependencies = []

[project.optional-dependencies]
core-onnx-inspect = ["onnx==1.23.2"]

[tool.uv]
package = false
no-build = false
environments = ["sys_platform == 'linux' and platform_machine == 'x86_64'"]
exclude-newer = "2026-11-02T16:10:00Z"
required-version = "==0.13.7"
"""


def build_package(name: str, version: str, wheel_tags: str = "py3-none-any") -> str:
    return (
        f'[[package]]\nname = "{name}"\nversion = "{version}"\n'
        f'source = {{ registry = "https://pypi.org/simple" }}\n'
        f'wheels = [{{ url = "{FILES}/{name}-{version}-{wheel_tags}.whl", hash = "{HASH}" }}]\n'
    )


def build_lock(*packages: str) -> str:
    return f"""version = 1
revision = 3
requires-python = "==3.12.*"
supported-markers = ["platform_machine == 'x86_64' and sys_platform == 'linux'"]

[options]
exclude-newer = "2026-11-02T16:10:00Z"

[[package]]
name = "{ROOT}"
version = "0.1.0"
source = {{ virtual = "." }}

[package.metadata]
requires-dist = [
    {{ name = "onnx", marker = "extra == 'core-onnx-inspect'", specifier = "==1.23.2" }},
]
provides-extras = ["core-onnx-inspect"]

""" + "\n".join(packages)


COMMITTED_LOCK = build_lock(
    build_package("numpy", "2.5.3"),
    build_package("onnx", "1.23.2"),
    build_package("six", "1.17.0"),
)
NEW_LOCK = build_lock(
    build_package("numpy", "2.5.4"),
    build_package("onnx", "1.23.2"),
    build_package("protobuf", "7.36.2"),
)
LOCK_WITHOUT_A_WHEEL = build_lock(
    build_package("onnx", "1.23.2"),
    build_package("antlr4-python3-runtime", "4.9.3", "cp312-cp312-musllinux_1_2_x86_64"),
)
STALE_LOCK = NEW_LOCK.replace('specifier = "==1.23.2"', 'specifier = "==1.23.1"')
PEAKS = StepPeaks(memory_peak_bytes=61 * 2**20, process_count_peak=6)
REQUEST = LockRequest(
    manifest_text=MANIFEST,
    committed_lock_text=COMMITTED_LOCK,
    images_lock_path=CLONE_LOCK_PATH,
    work_root=WORK_ROOT,
    head_tree=HEAD_TREE,
    host=HOST,
)


@dataclass(frozen=True)
class LockRecord:
    resolution: LockResolution | None
    shown_lines: list[str]
    runs: list[ContainerRun]
    files: FakeStepFiles


@dataclass
class ShownLines:
    lines: list[str] = field(default_factory=list)

    def show(self, line: str) -> None:
        self.lines.append(line)


def build_files() -> FakeStepFiles:
    files = FakeStepFiles()
    files.add_directory(WORK_ROOT)
    return files


def build_engine(files: FakeStepFiles, new_lock: str) -> FakeContainerEngine:
    return FakeContainerEngine(
        files=files,
        images={LOCK_ROW.image.tag: ImageFacts(IMAGE_ID, HEAD_TREE)},
        step_outputs={
            "environment.lock": build_step_output(new_lock.encode()),
            PEAKS_FILE_NAME: build_step_output(render_step_peaks(PEAKS)),
        },
    )


def resolve_quietly(request: LockRequest, new_lock: str, files: FakeStepFiles) -> LockRecord:
    engine = build_engine(files, new_lock)
    shown = ShownLines()
    resolution = None
    with contextlib.suppress(LockError, LockFreshnessError):
        resolution = resolve_lock(request, engine, files, shown.show)
    return LockRecord(resolution, shown.lines, engine.runs, files)


class TestTheResolveLock:
    class TestTheResolveLockFunction:
        class TestWhenALockIsCommitted:
            @pytest.fixture
            def lock_record(self) -> LockRecord:
                return resolve_quietly(REQUEST, NEW_LOCK, build_files())

            def test_should_hand_the_container_copies_of_the_manifest_and_the_lock(
                self, lock_record: LockRecord
            ) -> None:
                assert {
                    path.name: content.decode()
                    for path, content in lock_record.files.contents.items()
                    if path.parent == WORK_ROOT / "lock/1/request"
                } == {"environment.toml": MANIFEST, "environment.lock": COMMITTED_LOCK}

            def test_should_print_the_packages_added_removed_and_changed_in_version(
                self, lock_record: LockRecord
            ) -> None:
                assert lock_record.shown_lines == [
                    "added protobuf 7.36.2",
                    "removed six 1.17.0",
                    "changed numpy from 2.5.3 to 2.5.4",
                ]

            def test_should_write_the_checked_lock_into_the_clone(
                self, lock_record: LockRecord
            ) -> None:
                assert lock_record.files.contents[CLONE_LOCK_PATH] == NEW_LOCK.encode()

            def test_should_run_the_lock_row_with_the_network(
                self, lock_record: LockRecord
            ) -> None:
                assert [(run.entry_module, run.has_network) for run in lock_record.runs] == [
                    (LOCK_ROW.entry_module, True)
                ]

        class TestWhenNoLockIsCommittedYet:
            @pytest.fixture
            def lock_record(self) -> LockRecord:
                return resolve_quietly(
                    replace(REQUEST, committed_lock_text=None), NEW_LOCK, build_files()
                )

            def test_should_hand_the_container_the_manifest_alone(
                self, lock_record: LockRecord
            ) -> None:
                assert [
                    path.name
                    for path in lock_record.files.contents
                    if path.parent == WORK_ROOT / "lock/1/request"
                ] == ["environment.toml"]

            def test_should_print_every_package_as_added(self, lock_record: LockRecord) -> None:
                assert lock_record.shown_lines == [
                    "added numpy 2.5.4",
                    "added onnx 1.23.2",
                    "added protobuf 7.36.2",
                ]

        class TestWhenEarlierLocksRan:
            @pytest.fixture
            def lock_record(self) -> LockRecord:
                files = build_files()
                files.add_directory(WORK_ROOT / "lock/1")
                files.add_directory(WORK_ROOT / "lock/2")
                return resolve_quietly(REQUEST, NEW_LOCK, files)

            def test_should_run_in_a_new_directory_after_them(
                self, lock_record: LockRecord
            ) -> None:
                assert lock_record.resolution is not None
                assert lock_record.resolution.lock_run == PurePosixPath("lock/3")

        class TestWhenTheNewLockHoldsAPackageWithoutAWheelThatInstallsHere:
            @pytest.fixture
            def resolve_act(self) -> Callable[[], LockResolution]:
                files = build_files()
                return partial(
                    resolve_lock,
                    REQUEST,
                    build_engine(files, LOCK_WITHOUT_A_WHEEL),
                    files,
                    ShownLines().show,
                )

            @pytest.fixture
            def lock_record(self) -> LockRecord:
                return resolve_quietly(REQUEST, LOCK_WITHOUT_A_WHEEL, build_files())

            def test_should_report_it_for_the_user_to_decide(
                self, resolve_act: Callable[[], LockResolution]
            ) -> None:
                with pytest.raises(
                    LockError,
                    match=re.escape(
                        "antlr4-python3-runtime 4.9.3 has no wheel that installs on Python 3.12"
                    ),
                ):
                    resolve_act()

            def test_should_leave_the_clone_as_it_was(self, lock_record: LockRecord) -> None:
                assert (CLONE_LOCK_PATH in lock_record.files.contents, lock_record.shown_lines) == (
                    False,
                    [],
                )

        class TestWhenTheNewLockNoLongerRecordsTheManifest:
            @pytest.fixture
            def resolve_act(self) -> Callable[[], LockResolution]:
                files = build_files()
                return partial(
                    resolve_lock, REQUEST, build_engine(files, STALE_LOCK), files, ShownLines().show
                )

            def test_should_refuse_it(self, resolve_act: Callable[[], LockResolution]) -> None:
                with pytest.raises(LockFreshnessError, match="requires-dist: the lock records"):
                    resolve_act()

        class TestWhenTheContainerLeavesTheLockReadableByItselfAlone:
            @pytest.fixture
            def lock_record(self) -> LockRecord:
                files = build_files()
                engine = build_engine(files, NEW_LOCK)
                facts, content = engine.step_outputs["environment.lock"]
                engine.step_outputs["environment.lock"] = (
                    replace(facts, permission_bits=0o600),
                    content,
                )
                resolution = None
                with contextlib.suppress(ContainerFileError):
                    resolution = resolve_lock(REQUEST, engine, files, ShownLines().show)
                return LockRecord(resolution, [], engine.runs, files)

            def test_should_judge_it_as_any_output_and_write_nothing_into_the_clone(
                self, lock_record: LockRecord
            ) -> None:
                assert (lock_record.resolution, CLONE_LOCK_PATH in lock_record.files.contents) == (
                    None,
                    False,
                )

        class TestWhenTheLockIsWritten:
            @pytest.fixture
            def lock_record(self) -> LockRecord:
                return resolve_quietly(REQUEST, NEW_LOCK, build_files())

            def test_should_give_its_packages_the_extras_it_records_and_the_steps_peaks(
                self, lock_record: LockRecord
            ) -> None:
                assert lock_record.resolution is not None
                assert (
                    lock_record.resolution.checked_lock.packages,
                    lock_record.resolution.recorded_extras,
                    lock_record.resolution.peaks,
                ) == (
                    (
                        LockedPackage("numpy", "2.5.4"),
                        LockedPackage("onnx", "1.23.2"),
                        LockedPackage("protobuf", "7.36.2"),
                    ),
                    ("core-onnx-inspect",),
                    PEAKS,
                )
