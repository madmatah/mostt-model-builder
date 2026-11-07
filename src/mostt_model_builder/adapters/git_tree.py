import io
import os
import subprocess
import tarfile
from collections.abc import Collection, Mapping, Sequence
from pathlib import PurePosixPath

from mostt_model_builder.application.build_images import TarArchive
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
# A replace ref would make git read another object in place of the one the label names.
NO_REPLACE_OBJECTS_OPTION = "--no-replace-objects"
# The user's git settings would otherwise reshape the archive: its line endings, a global
# attributes file with `ident`, `export-ignore` or a filter, and the modes of its files.
ARCHIVE_SETTINGS = (
    "core.autocrlf=false",
    "core.eol=lf",
    "core.attributesFile=/dev/null",
    "tar.umask=0022",
)


class GitError(Exception):
    pass


def drop_git_variables(environment: Mapping[str, str]) -> dict[str, str]:
    return {
        name: value
        for name, value in environment.items()
        if not name.startswith(GIT_VARIABLE_PREFIX)
    }


def read_build_files(archive: bytes, directories: Collection[str]) -> dict[PurePosixPath, bytes]:
    files: dict[PurePosixPath, bytes] = {}
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:") as archive_file:
        for member in archive_file:
            path = PurePosixPath(member.name)
            if path.parts[0] not in directories or member.isdir():
                continue
            extracted = archive_file.extractfile(member) if member.isfile() else None
            if extracted is None:
                raise GitError(
                    f"the tree holds {member.name!a}, neither a file nor a directory, among the "
                    "build's code and images"
                )
            files[path] = extracted.read()
    return files


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

    def run_git(self, arguments: list[str], settings: Sequence[str] = ()) -> bytes:
        setting_options = [option for setting in settings for option in ("-c", setting)]
        try:
            completed = subprocess.run(
                [
                    GIT_PROGRAM,
                    "-C",
                    str(self.clone_root),
                    NO_REPLACE_OBJECTS_OPTION,
                    *setting_options,
                    *arguments,
                ],
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

    def archive_tree(self, tree_hash: str) -> TarArchive:
        tree = require_object_name(tree_hash, "the tree to archive")
        return TarArchive(self.run_git(["archive", "--format=tar", tree], ARCHIVE_SETTINGS))

    def read_tree_files(
        self, tree_hash: str, directories: Collection[str]
    ) -> dict[PurePosixPath, bytes]:
        return read_build_files(self.archive_tree(tree_hash), directories)
