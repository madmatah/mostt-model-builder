import re
from collections.abc import Callable
from dataclasses import replace
from functools import partial
from pathlib import PurePosixPath

import pytest

from mostt_model_builder.domain.containers.container_files import (
    ContainerFileError,
    FileFacts,
    FileKind,
    PathComponent,
    judge_mount,
    judge_output_entries,
    judge_written_file,
)

CONTAINER_USER_ID = 65531
USER_ID = 1000
OTHER_USER_ID = 1001
MAX_SIZE_BYTES = 4096
OUTPUT_NAME = "peaks.json"
GOOD_WRITTEN_FILE = FileFacts(FileKind.REGULAR, CONTAINER_USER_ID, 0o640, 56, 1)
REFUSED_WRITTEN_FILES = {
    "an absolute or relative link": (
        replace(GOOD_WRITTEN_FILE, kind=FileKind.LINK),
        "the output peaks.json is a link, not a regular file",
    ),
    "a fifo": (
        replace(GOOD_WRITTEN_FILE, kind=FileKind.FIFO),
        "the output peaks.json is a FIFO, not a regular file",
    ),
    "a socket": (
        replace(GOOD_WRITTEN_FILE, kind=FileKind.SOCKET),
        "the output peaks.json is a socket, not a regular file",
    ),
    "a device": (
        replace(GOOD_WRITTEN_FILE, kind=FileKind.DEVICE),
        "the output peaks.json is a device, not a regular file",
    ),
    "a directory in place of a file": (
        replace(GOOD_WRITTEN_FILE, kind=FileKind.DIRECTORY),
        "the output peaks.json is a directory, not a regular file",
    ),
    "a missing file": (
        replace(GOOD_WRITTEN_FILE, kind=FileKind.MISSING),
        "the output peaks.json is missing, not a regular file",
    ),
    "an unreadable file": (
        replace(GOOD_WRITTEN_FILE, kind=FileKind.UNREADABLE),
        "the output peaks.json cannot be read, not a regular file",
    ),
    "another owner": (
        replace(GOOD_WRITTEN_FILE, owner_id=USER_ID),
        "the output peaks.json is owned by the user id 1000, not by 65531",
    ),
    "a setuid file": (
        replace(GOOD_WRITTEN_FILE, permission_bits=0o4640),
        "the output peaks.json is setuid or setgid",
    ),
    "a setgid file": (
        replace(GOOD_WRITTEN_FILE, permission_bits=0o2640),
        "the output peaks.json is setuid or setgid",
    ),
    "another mode": (
        replace(GOOD_WRITTEN_FILE, permission_bits=0o600),
        "the output peaks.json has the mode 0600, not 0640",
    ),
    "a file beyond its size bound": (
        replace(GOOD_WRITTEN_FILE, size_bytes=MAX_SIZE_BYTES + 1),
        "the output peaks.json holds 4097 bytes, beyond its bound of 4096",
    ),
    "several links": (
        replace(GOOD_WRITTEN_FILE, link_count=2),
        "the output peaks.json has 2 links, not one",
    ),
}
DIRECTORY = FileFacts(FileKind.DIRECTORY, USER_ID, 0o700, 4096, 2)
SOURCE_FILE = FileFacts(FileKind.REGULAR, USER_ID, 0o444, 456_972, 1)
WORK_ROOT = PurePosixPath("/data/builds/work")


def build_components(*facts: FileFacts) -> tuple[PathComponent, ...]:
    return tuple(
        PathComponent(WORK_ROOT / f"part{part_index}", part_facts)
        for part_index, part_facts in enumerate(facts)
    )


GOOD_FILE_MOUNT = build_components(DIRECTORY, DIRECTORY, SOURCE_FILE)
GOOD_DIRECTORY_MOUNT = build_components(DIRECTORY, replace(DIRECTORY, permission_bits=0o2770))
REFUSED_MOUNTS = {
    "a link on the way": (
        build_components(DIRECTORY, replace(DIRECTORY, kind=FileKind.LINK), SOURCE_FILE),
        True,
        "part1 is a link, where a directory is declared",
    ),
    "a directory on the way that another user owns": (
        build_components(DIRECTORY, replace(DIRECTORY, owner_id=OTHER_USER_ID), SOURCE_FILE),
        True,
        "part1 is owned by the user id 1001, not by 1000",
    ),
    "a directory where a file is declared": (
        build_components(DIRECTORY, DIRECTORY),
        True,
        "part1 is a directory, where a file is declared",
    ),
    "a link where a file is declared": (
        build_components(DIRECTORY, replace(SOURCE_FILE, kind=FileKind.LINK)),
        True,
        "part1 is a link, where a file is declared",
    ),
    "a missing file": (
        build_components(DIRECTORY, replace(SOURCE_FILE, kind=FileKind.MISSING)),
        True,
        "part1 is missing, where a file is declared",
    ),
    "a file that another user owns": (
        build_components(DIRECTORY, replace(SOURCE_FILE, owner_id=OTHER_USER_ID)),
        True,
        "part1 is owned by the user id 1001, not by 1000",
    ),
    "a file where a directory is declared": (
        build_components(DIRECTORY, SOURCE_FILE),
        False,
        "part1 is a file, where a directory is declared",
    ),
    "a missing directory": (
        build_components(DIRECTORY, replace(DIRECTORY, kind=FileKind.MISSING)),
        False,
        "part1 is missing, where a directory is declared",
    ),
    "no path at all": ((), False, "a mount names no path"),
}


class TestTheContainerFiles:
    class TestTheJudgeWrittenFileFunction:
        class TestWhenTheFileIsAsAStepLeavesIt:
            @pytest.fixture
            def judge_act(self) -> Callable[[], object]:
                return partial(
                    judge_written_file,
                    OUTPUT_NAME,
                    GOOD_WRITTEN_FILE,
                    MAX_SIZE_BYTES,
                    CONTAINER_USER_ID,
                )

            def test_should_accept_it(self, judge_act: Callable[[], object]) -> None:
                assert judge_act() is None

        @pytest.mark.parametrize(
            ("facts", "reason"),
            list(REFUSED_WRITTEN_FILES.values()),
            ids=list(REFUSED_WRITTEN_FILES.keys()),
        )
        class TestWhenTheFileIsNotWhatAStepLeaves:
            @pytest.fixture
            def judge_act(self, facts: FileFacts) -> Callable[[], object]:
                return partial(
                    judge_written_file, OUTPUT_NAME, facts, MAX_SIZE_BYTES, CONTAINER_USER_ID
                )

            def test_should_refuse_it_and_say_why(
                self, judge_act: Callable[[], object], reason: str
            ) -> None:
                with pytest.raises(ContainerFileError, match=f"^{re.escape(reason)}$"):
                    judge_act()

    class TestTheJudgeOutputEntriesFunction:
        class TestWhenTheDirectoryHoldsExactlyTheDeclaredNames:
            @pytest.fixture
            def judge_act(self) -> Callable[[], object]:
                return partial(
                    judge_output_entries, ["a.json", OUTPUT_NAME], [OUTPUT_NAME, "a.json"]
                )

            def test_should_accept_it(self, judge_act: Callable[[], object]) -> None:
                assert judge_act() is None

        class TestWhenTheDirectoryHoldsAnUndeclaredEntry:
            @pytest.fixture
            def judge_act(self) -> Callable[[], object]:
                return partial(judge_output_entries, [OUTPUT_NAME, "\x1b[2J"], [OUTPUT_NAME])

            def test_should_refuse_it_and_show_its_name_escaped(
                self, judge_act: Callable[[], object]
            ) -> None:
                with pytest.raises(
                    ContainerFileError,
                    match=re.escape("the step wrote '\\x1b[2J', which its row does not declare"),
                ):
                    judge_act()

        class TestWhenADeclaredOutputIsMissing:
            @pytest.fixture
            def judge_act(self) -> Callable[[], object]:
                return partial(judge_output_entries, [], [OUTPUT_NAME])

            def test_should_refuse_it(self, judge_act: Callable[[], object]) -> None:
                with pytest.raises(
                    ContainerFileError,
                    match=re.escape("the step did not write its output peaks.json"),
                ):
                    judge_act()

    class TestTheJudgeMountFunction:
        @pytest.mark.parametrize(
            ("components", "is_file"),
            [
                pytest.param(GOOD_FILE_MOUNT, True, id="a file"),
                pytest.param(GOOD_DIRECTORY_MOUNT, False, id="a directory"),
            ],
        )
        class TestWhenEveryComponentIsTheUsersOwnAndNoLink:
            @pytest.fixture
            def judge_act(
                self, components: tuple[PathComponent, ...], is_file: bool
            ) -> Callable[[], object]:
                return partial(judge_mount, components, is_file, USER_ID)

            def test_should_accept_it(self, judge_act: Callable[[], object]) -> None:
                assert judge_act() is None

        @pytest.mark.parametrize(
            ("components", "is_file", "reason"),
            list(REFUSED_MOUNTS.values()),
            ids=list(REFUSED_MOUNTS.keys()),
        )
        class TestWhenAComponentIsNotWhatTheMountDeclares:
            @pytest.fixture
            def judge_act(
                self, components: tuple[PathComponent, ...], is_file: bool
            ) -> Callable[[], object]:
                return partial(judge_mount, components, is_file, USER_ID)

            def test_should_refuse_it_and_say_why(
                self, judge_act: Callable[[], object], reason: str
            ) -> None:
                with pytest.raises(ContainerFileError, match=re.escape(reason)):
                    judge_act()
