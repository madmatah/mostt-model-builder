import os
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from pathlib import Path, PurePosixPath

import pytest

from mostt_model_builder.adapters.git_tree import GitError, GitTree, drop_git_variables

GIT_TIMEOUT_S = 60
COMMITTED_FILE = "tracked.txt"
UNTRACKED_FILE = "src/untracked.py"
IGNORED_FILE = "ignored.log"
RENAMED_FILE = "renamed.txt"


def run_git(repository: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repository), *arguments],
        capture_output=True,
        check=True,
        text=True,
        env={
            **drop_git_variables(os.environ),
            "GIT_AUTHOR_NAME": "Test",
            "GIT_AUTHOR_EMAIL": "test@example.org",
            "GIT_COMMITTER_NAME": "Test",
            "GIT_COMMITTER_EMAIL": "test@example.org",
        },
        timeout=GIT_TIMEOUT_S,
    )
    return completed.stdout.strip()


def build_repository(repository: Path) -> None:
    run_git(repository, "init", "--quiet")
    (repository / COMMITTED_FILE).write_text("content\n")
    (repository / ".gitignore").write_text("*.log\n")
    run_git(repository, "add", COMMITTED_FILE, ".gitignore")
    run_git(repository, "-c", "commit.gpgsign=false", "commit", "--quiet", "--message", "first")


@dataclass(frozen=True)
class CloneFacts:
    head_commit: str
    head_tree: str
    expected_commit: str
    expected_tree: str


class TestTheGitTree:
    class TestTheFindHeadMethods:
        class TestWhenTheCloneHasACommit:
            @pytest.fixture
            def clone_facts(self, tmp_path: Path) -> CloneFacts:
                build_repository(tmp_path)
                git_tree = GitTree(PurePosixPath(tmp_path))
                return CloneFacts(
                    git_tree.find_head_commit(),
                    git_tree.find_head_tree(),
                    run_git(tmp_path, "rev-parse", "HEAD"),
                    run_git(tmp_path, "rev-parse", "HEAD^{tree}"),
                )

            def test_should_name_the_commit_and_the_tree_of_head(
                self, clone_facts: CloneFacts
            ) -> None:
                assert (clone_facts.head_commit, clone_facts.head_tree) == (
                    clone_facts.expected_commit,
                    clone_facts.expected_tree,
                )

    class TestTheListChangesMethod:
        class TestWhenTheCloneIsClean:
            @pytest.fixture
            def changes(self, tmp_path: Path) -> tuple[str, ...]:
                build_repository(tmp_path)
                (tmp_path / IGNORED_FILE).write_text("ignored\n")
                return GitTree(PurePosixPath(tmp_path)).list_changes()

            def test_should_find_none_and_leave_ignored_files_out(
                self, changes: tuple[str, ...]
            ) -> None:
                assert changes == ()

        class TestWhenTheCloneHoldsAnUntrackedFileAndAChange:
            @pytest.fixture
            def changes(self, tmp_path: Path) -> tuple[str, ...]:
                build_repository(tmp_path)
                (tmp_path / "src").mkdir()
                (tmp_path / UNTRACKED_FILE).write_text("x = 1\n")
                (tmp_path / COMMITTED_FILE).write_text("changed\n")
                return GitTree(PurePosixPath(tmp_path)).list_changes()

            def test_should_count_both_so_no_image_holds_what_no_commit_names(
                self, changes: tuple[str, ...]
            ) -> None:
                assert sorted(changes) == sorted((COMMITTED_FILE, UNTRACKED_FILE))

        class TestWhenTheCloneRenamesAFile:
            @pytest.fixture
            def changes(self, tmp_path: Path) -> tuple[str, ...]:
                build_repository(tmp_path)
                run_git(tmp_path, "mv", COMMITTED_FILE, RENAMED_FILE)
                return GitTree(PurePosixPath(tmp_path)).list_changes()

            def test_should_name_both_paths_of_the_rename_whole(
                self, changes: tuple[str, ...]
            ) -> None:
                assert sorted(changes) == sorted((RENAMED_FILE, COMMITTED_FILE))

    class TestTheRunGitMethod:
        class TestWhenGitGivesNoAnswerInTime:
            @pytest.fixture
            def run_act(
                self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
            ) -> Callable[[], bytes]:
                def run_until_the_timeout(
                    arguments: list[str], *, timeout: float, **_options: object
                ) -> subprocess.CompletedProcess[bytes]:
                    raise subprocess.TimeoutExpired(arguments, timeout)

                monkeypatch.setattr(subprocess, "run", run_until_the_timeout)
                return partial(GitTree(PurePosixPath(tmp_path)).run_git, ["status"])

            def test_should_refuse_it_and_name_the_command_and_its_timeout(
                self, run_act: Callable[[], bytes]
            ) -> None:
                with pytest.raises(GitError, match=r"^git status gave no answer within 60 s$"):
                    run_act()

    class TestTheDropGitVariablesFunction:
        class TestWhenTheEnvironmentPointsGitElsewhere:
            @pytest.fixture
            def environment(self) -> dict[str, str]:
                return drop_git_variables(
                    {"GIT_DIR": "/elsewhere/.git", "GIT_INDEX_FILE": "/x", "PATH": "/usr/bin"}
                )

            def test_should_keep_only_what_is_not_gits_so_the_clone_itself_is_read(
                self, environment: dict[str, str]
            ) -> None:
                assert environment == {"PATH": "/usr/bin"}
