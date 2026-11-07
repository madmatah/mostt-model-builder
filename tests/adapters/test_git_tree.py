import io
import os
import re
import subprocess
import tarfile
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from pathlib import Path, PurePosixPath

import pytest

from mostt_model_builder.adapters.git_tree import GitError, GitTree, drop_git_variables
from mostt_model_builder.domain.containers.provenance import ProvenanceError

GIT_TIMEOUT_S = 60
COMMITTED_FILE = "tracked.txt"
UNTRACKED_FILE = "src/untracked.py"
IGNORED_FILE = "ignored.log"
IGNORED_SOURCE_FILE = "src/ignored.log"
RENAMED_FILE = "renamed.txt"
COMMITTED_CONTENT = b"content\n"
CHANGED_CONTENT = b"changed\n"
IDENT_CONTENT = b"$Id$\nfirst line\nsecond line\n"
IDENT_FILE = "ident.txt"
RESHAPING_ATTRIBUTES = "* ident text eol=crlf\n"
COMMITTED_TEXT_ATTRIBUTES = "* text=auto\n"
RESHAPING_UMASK = "0077"
COMMITTED_FILE_MODE = 0o644
MISSING_TREE = "0" * 40
MALFORMED_TREE = "HEAD"
BUILD_DIRECTORIES = frozenset({"src", "recipes", "images"})
BUILD_FILES = {
    "src/mostt_model_builder/steps/step.py": b"import json\n",
    "recipes/family/src/mostt_recipe_family/steps/run.py": b"x = 1\n",
    "images/bench/Containerfile": b"FROM python\n",
}
DOCUMENT_FILE = "docs/model-build.md"


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
    (repository / COMMITTED_FILE).write_bytes(COMMITTED_CONTENT)
    (repository / ".gitignore").write_text("*.log\n")
    run_git(repository, "add", COMMITTED_FILE, ".gitignore")
    run_git(repository, "-c", "commit.gpgsign=false", "commit", "--quiet", "--message", "first")


def commit_build_files(repository: Path) -> None:
    for name, content in BUILD_FILES.items():
        (repository / name).parent.mkdir(parents=True, exist_ok=True)
        (repository / name).write_bytes(content)
    (repository / DOCUMENT_FILE).parent.mkdir()
    (repository / DOCUMENT_FILE).write_text("# The build\n")
    run_git(repository, "add", *BUILD_FILES, DOCUMENT_FILE)
    run_git(repository, "-c", "commit.gpgsign=false", "commit", "--quiet", "--message", "code")


def write_reshaping_configuration(home: Path) -> None:
    attributes_path = home / "attributes"
    attributes_path.write_text(RESHAPING_ATTRIBUTES)
    (home / ".gitconfig").write_text(
        "[core]\n"
        "\tautocrlf = true\n"
        "\teol = crlf\n"
        f"\tattributesFile = {attributes_path}\n"
        "[tar]\n"
        f"\tumask = {RESHAPING_UMASK}\n"
    )


def read_archived_modes(archive: bytes) -> dict[str, int]:
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:") as archive_file:
        return {member.name: member.mode for member in archive_file.getmembers()}


def read_archived_files(archive: bytes) -> dict[str, bytes]:
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:") as archive_file:
        return {
            member.name: extracted.read()
            for member in archive_file.getmembers()
            if (extracted := archive_file.extractfile(member)) is not None
        }


@dataclass(frozen=True)
class ArchivedTree:
    files: dict[str, bytes]
    modes: dict[str, int]


@dataclass(frozen=True)
class CloneFacts:
    head_tree: str
    expected_tree: str


class TestTheGitTree:
    class TestTheFindHeadTreeMethod:
        class TestWhenTheCloneHasACommit:
            @pytest.fixture
            def clone_facts(self, tmp_path: Path) -> CloneFacts:
                build_repository(tmp_path)
                return CloneFacts(
                    GitTree(PurePosixPath(tmp_path)).find_head_tree(),
                    run_git(tmp_path, "rev-parse", "HEAD^{tree}"),
                )

            def test_should_name_the_tree_of_head(self, clone_facts: CloneFacts) -> None:
                assert clone_facts.head_tree == clone_facts.expected_tree

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

    class TestTheArchiveTreeMethod:
        class TestWhenTheCloneHoldsIgnoredUntrackedAndChangedFilesBesideItsCommit:
            @pytest.fixture
            def archived_files(self, tmp_path: Path) -> dict[str, bytes]:
                build_repository(tmp_path)
                (tmp_path / "src").mkdir()
                (tmp_path / IGNORED_FILE).write_text("ignored\n")
                (tmp_path / IGNORED_SOURCE_FILE).write_text("ignored\n")
                (tmp_path / UNTRACKED_FILE).write_text("x = 1\n")
                (tmp_path / COMMITTED_FILE).write_bytes(CHANGED_CONTENT)
                git_tree = GitTree(PurePosixPath(tmp_path))
                return read_archived_files(git_tree.archive_tree(git_tree.find_head_tree()))

            def test_should_hold_only_committed_files_so_no_ignored_file_enters_an_image(
                self, archived_files: dict[str, bytes]
            ) -> None:
                assert sorted(archived_files) == [".gitignore", COMMITTED_FILE]

            def test_should_hold_each_file_as_committed_rather_than_as_the_working_tree_holds_it(
                self, archived_files: dict[str, bytes]
            ) -> None:
                assert archived_files[COMMITTED_FILE] == COMMITTED_CONTENT

        class TestWhenALaterCommitChangesAFile:
            @pytest.fixture
            def archived_files(self, tmp_path: Path) -> dict[str, bytes]:
                build_repository(tmp_path)
                (tmp_path / COMMITTED_FILE).write_bytes(CHANGED_CONTENT)
                run_git(
                    tmp_path,
                    "-c",
                    "commit.gpgsign=false",
                    "commit",
                    "--quiet",
                    "--all",
                    "--message",
                    "second",
                )
                first_tree = run_git(tmp_path, "rev-parse", "HEAD~1^{tree}")
                return read_archived_files(
                    GitTree(PurePosixPath(tmp_path)).archive_tree(first_tree)
                )

            def test_should_hold_the_files_of_the_tree_it_is_given_rather_than_of_head(
                self, archived_files: dict[str, bytes]
            ) -> None:
                assert archived_files[COMMITTED_FILE] == COMMITTED_CONTENT

        class TestWhenAReplaceRefSwapsTheTreeForAnother:
            @pytest.fixture
            def archived_files(self, tmp_path: Path) -> dict[str, bytes]:
                build_repository(tmp_path)
                first_tree = run_git(tmp_path, "rev-parse", "HEAD^{tree}")
                (tmp_path / COMMITTED_FILE).write_bytes(CHANGED_CONTENT)
                run_git(
                    tmp_path,
                    "-c",
                    "commit.gpgsign=false",
                    "commit",
                    "--quiet",
                    "--all",
                    "--message",
                    "second",
                )
                run_git(
                    tmp_path, "replace", first_tree, run_git(tmp_path, "rev-parse", "HEAD^{tree}")
                )
                return read_archived_files(
                    GitTree(PurePosixPath(tmp_path)).archive_tree(first_tree)
                )

            def test_should_hold_the_files_of_the_tree_itself_so_the_label_names_its_bytes(
                self, archived_files: dict[str, bytes]
            ) -> None:
                assert archived_files[COMMITTED_FILE] == COMMITTED_CONTENT

        class TestWhenTheUsersGitConfigurationWouldReshapeItsFiles:
            @pytest.fixture
            def archived_tree(
                self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
            ) -> ArchivedTree:
                repository = tmp_path / "repository"
                home = tmp_path / "home"
                repository.mkdir()
                home.mkdir()
                build_repository(repository)
                (repository / IDENT_FILE).write_bytes(IDENT_CONTENT)
                (repository / ".gitattributes").write_text(COMMITTED_TEXT_ATTRIBUTES)
                run_git(repository, "add", IDENT_FILE, ".gitattributes")
                run_git(
                    repository,
                    "-c",
                    "commit.gpgsign=false",
                    "commit",
                    "--quiet",
                    "--message",
                    "ident",
                )
                write_reshaping_configuration(home)
                monkeypatch.setenv("HOME", str(home))
                monkeypatch.setenv("XDG_CONFIG_HOME", str(home))
                git_tree = GitTree(PurePosixPath(repository))
                archive = git_tree.archive_tree(git_tree.find_head_tree())
                return ArchivedTree(read_archived_files(archive), read_archived_modes(archive))

            def test_should_hold_each_file_as_committed_so_the_label_names_its_bytes(
                self, archived_tree: ArchivedTree
            ) -> None:
                assert (
                    archived_tree.files[COMMITTED_FILE],
                    archived_tree.files[IDENT_FILE],
                ) == (COMMITTED_CONTENT, IDENT_CONTENT)

            def test_should_give_each_file_the_mode_of_a_committed_file_whatever_the_users_umask(
                self, archived_tree: ArchivedTree
            ) -> None:
                assert archived_tree.modes[COMMITTED_FILE] == COMMITTED_FILE_MODE

        class TestWhenTheTreeIsNotAGitObjectName:
            @pytest.fixture
            def archive_act(self, tmp_path: Path) -> Callable[[], bytes]:
                build_repository(tmp_path)
                return partial(GitTree(PurePosixPath(tmp_path)).archive_tree, MALFORMED_TREE)

            def test_should_refuse_it_before_git_reads_it_as_a_revision(
                self, archive_act: Callable[[], bytes]
            ) -> None:
                with pytest.raises(
                    ProvenanceError,
                    match=rf"^the tree to archive '{re.escape(MALFORMED_TREE)}' "
                    r"is not a git object name$",
                ):
                    archive_act()

        class TestWhenTheCloneHoldsNoSuchTree:
            @pytest.fixture
            def archive_act(self, tmp_path: Path) -> Callable[[], bytes]:
                build_repository(tmp_path)
                return partial(GitTree(PurePosixPath(tmp_path)).archive_tree, MISSING_TREE)

            def test_should_refuse_it_and_show_what_git_says(
                self, archive_act: Callable[[], bytes]
            ) -> None:
                with pytest.raises(GitError, match=rf"^git archive failed: .*{MISSING_TREE}"):
                    archive_act()

    class TestTheReadTreeFilesMethod:
        class TestWhenTheCloneChangesAndAddsFilesBesideItsCommit:
            @pytest.fixture
            def tree_files(self, tmp_path: Path) -> dict[PurePosixPath, bytes]:
                build_repository(tmp_path)
                commit_build_files(tmp_path)
                (tmp_path / "src/mostt_model_builder/steps/step.py").write_bytes(CHANGED_CONTENT)
                (tmp_path / UNTRACKED_FILE).write_text("x = 2\n")
                git_tree = GitTree(PurePosixPath(tmp_path))
                return git_tree.read_tree_files(git_tree.find_head_tree(), BUILD_DIRECTORIES)

            def test_should_give_the_committed_files_of_the_code_and_the_images_alone(
                self, tree_files: dict[PurePosixPath, bytes]
            ) -> None:
                assert tree_files == {
                    PurePosixPath(name): content for name, content in BUILD_FILES.items()
                }

        class TestWhenTheTreeHoldsALinkAmongTheCode:
            @pytest.fixture
            def read_act(self, tmp_path: Path) -> Callable[[], dict[PurePosixPath, bytes]]:
                build_repository(tmp_path)
                (tmp_path / "src").mkdir()
                (tmp_path / "src/linked.py").symlink_to("/etc/passwd")
                run_git(tmp_path, "add", "src/linked.py")
                run_git(
                    tmp_path, "-c", "commit.gpgsign=false", "commit", "--quiet", "--message", "link"
                )
                git_tree = GitTree(PurePosixPath(tmp_path))
                return partial(
                    git_tree.read_tree_files, git_tree.find_head_tree(), BUILD_DIRECTORIES
                )

            def test_should_refuse_it_since_no_step_runs_a_link(
                self, read_act: Callable[[], dict[PurePosixPath, bytes]]
            ) -> None:
                with pytest.raises(
                    GitError,
                    match=re.escape(
                        "the tree holds 'src/linked.py', neither a file nor a directory, among "
                        "the build's code and images"
                    ),
                ):
                    read_act()

        class TestWhenTheTreeIsNotAGitObjectName:
            @pytest.fixture
            def read_act(self, tmp_path: Path) -> Callable[[], dict[PurePosixPath, bytes]]:
                build_repository(tmp_path)
                return partial(
                    GitTree(PurePosixPath(tmp_path)).read_tree_files,
                    MALFORMED_TREE,
                    BUILD_DIRECTORIES,
                )

            def test_should_refuse_it_before_git_reads_it_as_a_revision(
                self, read_act: Callable[[], dict[PurePosixPath, bytes]]
            ) -> None:
                with pytest.raises(ProvenanceError, match="is not a git object name"):
                    read_act()

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
