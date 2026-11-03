import contextlib
import re
from collections.abc import Callable, Mapping
from dataclasses import replace
from functools import partial

import pytest

from mostt_model_builder.application.open_work_directory import (
    WorkDirectoryRequest,
    open_work_directory,
)
from mostt_model_builder.domain.containers.work_directory import (
    DirectoryFacts,
    EntryKind,
    WorkDirectory,
    WorkDirectoryError,
)

USER_ID = 1000
WORK_PATH = "/data/builds/work"
GOOD_REQUEST = WorkDirectoryRequest(
    path=WORK_PATH,
    home_variable="/home/user",
    account_home="/home/user",
    clone_root="/opt/clones/mostt-model-builder",
    user_id=USER_ID,
)


class FakeDirectoryInspector:
    def __init__(self, canonical_paths: Mapping[str, str]) -> None:
        self.canonical_paths = canonical_paths
        self.inspected_paths: list[str] = []

    def inspect_directory(self, path: str) -> DirectoryFacts:
        self.inspected_paths.append(path)
        return DirectoryFacts(EntryKind.DIRECTORY, self.canonicalize(path), USER_ID, 0o700)

    def canonicalize(self, path: str) -> str:
        return self.canonical_paths.get(path, path)


def match_exactly(message: str) -> str:
    return f"^{re.escape(message)}$"


REQUESTS_REFUSED_BEFORE_ANY_INSPECTION = {
    "a path outside the grammar": (
        replace(GOOD_REQUEST, path="builds/work"),
        "the work directory 'builds/work' does not match /[A-Za-z0-9._/-]+",
    ),
    "a missing $HOME": (
        replace(GOOD_REQUEST, home_variable=None),
        "$HOME is missing, so the home directory cannot be avoided",
    ),
    "a relative $HOME": (
        replace(GOOD_REQUEST, home_variable="home/user"),
        "$HOME is 'home/user', which is not an absolute path",
    ),
    "an account without a home": (
        replace(GOOD_REQUEST, account_home=""),
        "the account's home directory is missing, so the home directory cannot be avoided",
    ),
}
REQUESTS_REFUSED_ONCE_CANONICAL = {
    "a $HOME reached through a link": (
        replace(GOOD_REQUEST, home_variable="/home/linked-user"),
        {"/home/linked-user": "/data/builds"},
        f"the work directory {WORK_PATH} lies under the home directory '/data/builds'",
    ),
    "a clone reached through a link": (
        replace(GOOD_REQUEST, clone_root="/opt/clones/linked"),
        {"/opt/clones/linked": "/data"},
        f"the work directory {WORK_PATH} lies in the clone '/data'",
    ),
}


class TestTheOpenWorkDirectory:
    class TestTheOpenWorkDirectoryFunction:
        class TestWhenTheRequestIsGood:
            @pytest.fixture
            def work_directory(self) -> WorkDirectory:
                return open_work_directory(GOOD_REQUEST, FakeDirectoryInspector({}))

            def test_should_open_the_requested_directory(
                self, work_directory: WorkDirectory
            ) -> None:
                assert work_directory == WorkDirectory(WORK_PATH)

        @pytest.mark.parametrize(
            ("request_", "message"),
            list(REQUESTS_REFUSED_BEFORE_ANY_INSPECTION.values()),
            ids=list(REQUESTS_REFUSED_BEFORE_ANY_INSPECTION.keys()),
        )
        class TestWhenTheRequestCannotBeJudged:
            @pytest.fixture
            def open_act(self, request_: WorkDirectoryRequest) -> Callable[[], WorkDirectory]:
                return partial(open_work_directory, request_, FakeDirectoryInspector({}))

            def test_should_refuse_it_and_say_why(
                self, open_act: Callable[[], WorkDirectory], message: str
            ) -> None:
                with pytest.raises(WorkDirectoryError, match=match_exactly(message)):
                    open_act()

        @pytest.mark.parametrize(
            "request_",
            [request_ for request_, _ in REQUESTS_REFUSED_BEFORE_ANY_INSPECTION.values()],
            ids=list(REQUESTS_REFUSED_BEFORE_ANY_INSPECTION.keys()),
        )
        class TestWhenTheRequestCannotBeJudgedBeforeAnyPathIsRead:
            @pytest.fixture
            def inspected_paths(self, request_: WorkDirectoryRequest) -> list[str]:
                inspector = FakeDirectoryInspector({})
                with contextlib.suppress(WorkDirectoryError):
                    open_work_directory(request_, inspector)
                return inspector.inspected_paths

            def test_should_inspect_no_path_it_has_not_checked(
                self, inspected_paths: list[str]
            ) -> None:
                assert inspected_paths == []

        @pytest.mark.parametrize(
            ("request_", "canonical_paths", "message"),
            list(REQUESTS_REFUSED_ONCE_CANONICAL.values()),
            ids=list(REQUESTS_REFUSED_ONCE_CANONICAL.keys()),
        )
        class TestWhenALinkHidesWhereTheHomeOrTheCloneLies:
            @pytest.fixture
            def open_act(
                self, request_: WorkDirectoryRequest, canonical_paths: Mapping[str, str]
            ) -> Callable[[], WorkDirectory]:
                return partial(
                    open_work_directory, request_, FakeDirectoryInspector(canonical_paths)
                )

            def test_should_judge_the_canonical_path(
                self, open_act: Callable[[], WorkDirectory], message: str
            ) -> None:
                with pytest.raises(WorkDirectoryError, match=match_exactly(message)):
                    open_act()
