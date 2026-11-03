import os
import subprocess
from collections.abc import Mapping
from pathlib import PurePosixPath

from mostt_model_builder.domain.containers.printable_output import show_printable
from mostt_model_builder.domain.containers.provenance import require_object_name

GIT_PROGRAM = "git"
GIT_VARIABLE_PREFIX = "GIT_"
STATUS_SEPARATOR = b"\0"
# The two characters of a change's status, then a space, come before its path. A rename or a
# copy gives its original path as the next entry, without a status (git-status(1), "Porcelain
# Format Version 1").
STATUS_PREFIX_SIZE_BYTES = 3
STATUSES_WITH_AN_ORIGINAL_PATH = frozenset(b"RC")
GIT_TIMEOUT_S = 60


class GitError(Exception):
    pass


def drop_git_variables(environment: Mapping[str, str]) -> dict[str, str]:
    return {
        name: value
        for name, value in environment.items()
        if not name.startswith(GIT_VARIABLE_PREFIX)
    }


def parse_status(output: bytes) -> tuple[str, ...]:
    entries = iter(entry for entry in output.split(STATUS_SEPARATOR) if entry)
    paths = []
    for entry in entries:
        paths.append(show_printable(entry[STATUS_PREFIX_SIZE_BYTES:]))
        if STATUSES_WITH_AN_ORIGINAL_PATH & set(entry[: STATUS_PREFIX_SIZE_BYTES - 1]):
            paths.append(show_printable(next(entries, b"")))
    return tuple(paths)


class GitTree:
    def __init__(self, clone_root: PurePosixPath) -> None:
        self.clone_root = clone_root

    def run_git(self, arguments: list[str]) -> bytes:
        try:
            completed = subprocess.run(
                [GIT_PROGRAM, "-C", str(self.clone_root), *arguments],
                capture_output=True,
                check=False,
                stdin=subprocess.DEVNULL,
                env=drop_git_variables(os.environ),
                timeout=GIT_TIMEOUT_S,
            )
        except FileNotFoundError as error:
            raise GitError(f"{GIT_PROGRAM} is not installed") from error
        except subprocess.TimeoutExpired as error:
            raise GitError(
                f"{GIT_PROGRAM} {arguments[0]} gave no answer within {GIT_TIMEOUT_S} s"
            ) from error
        if completed.returncode != 0:
            raise GitError(
                f"{GIT_PROGRAM} {arguments[0]} failed: {show_printable(completed.stderr[-2000:])}"
            )
        return completed.stdout

    def find_head_commit(self) -> str:
        output = self.run_git(["rev-parse", "--verify", "HEAD^{commit}"])
        return require_object_name(show_printable(output).strip(), "the commit of HEAD")

    def find_head_tree(self) -> str:
        output = self.run_git(["rev-parse", "--verify", "HEAD^{tree}"])
        return require_object_name(show_printable(output).strip(), "the tree of HEAD")

    def list_changes(self) -> tuple[str, ...]:
        output = self.run_git(["status", "--porcelain=v1", "-z", "--untracked-files=all"])
        return parse_status(output)
