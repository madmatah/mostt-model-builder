import hashlib
import os
import stat
from pathlib import PurePosixPath

from mostt_model_builder.application.fetch_sources import FileDigest, SourceFileError
from mostt_model_builder.domain.containers.work_directory import DirectoryFacts, EntryKind

CREATED_DIRECTORY_MODE = 0o700
READ_ONLY_FILE_MODE = 0o444
DIGEST_ALGORITHM = "sha256"
NO_OWNER_ID = -1
NO_PERMISSION_BITS = 0
OPEN_WITHOUT_FOLLOWING = os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC


class WorkFiles:
    def inspect_directory(self, path: str) -> DirectoryFacts:
        canonical_path = os.path.realpath(path)
        try:
            entry_status = os.lstat(path)
        except (FileNotFoundError, NotADirectoryError):
            return DirectoryFacts(
                EntryKind.MISSING, canonical_path, NO_OWNER_ID, NO_PERMISSION_BITS
            )
        return DirectoryFacts(
            find_entry_kind(entry_status.st_mode),
            canonical_path,
            entry_status.st_uid,
            stat.S_IMODE(entry_status.st_mode),
        )

    def canonicalize(self, path: str) -> str:
        return os.path.realpath(path)

    def make_directories(self, root: PurePosixPath, relative_path: PurePosixPath) -> None:
        directory = root
        for part in relative_path.parts:
            directory = directory / part
            try:
                os.mkdir(directory, CREATED_DIRECTORY_MODE)
                continue
            except FileExistsError:
                entry_status = os.lstat(directory)
            if find_entry_kind(entry_status.st_mode) is not EntryKind.DIRECTORY:
                raise SourceFileError(f"{directory} is not a directory, or is a link")

    def find_file_size(self, path: PurePosixPath) -> int | None:
        try:
            entry_status = os.lstat(path)
        except FileNotFoundError:
            return None
        if not stat.S_ISREG(entry_status.st_mode):
            raise SourceFileError(f"{path} is not a regular file, or is a link")
        return entry_status.st_size

    def digest_file(self, path: PurePosixPath) -> FileDigest:
        descriptor = open_regular_file(path)
        with os.fdopen(descriptor, "rb") as stream:
            digest = hashlib.file_digest(stream, DIGEST_ALGORITHM)
            return FileDigest(os.fstat(stream.fileno()).st_size, digest.hexdigest())

    def rename_file(self, source: PurePosixPath, target: PurePosixPath) -> None:
        os.rename(source, target)

    def remove_file(self, path: PurePosixPath) -> None:
        os.unlink(path)

    def make_read_only(self, path: PurePosixPath) -> None:
        descriptor = open_regular_file(path)
        try:
            os.fchmod(descriptor, READ_ONLY_FILE_MODE)
        finally:
            os.close(descriptor)


def find_entry_kind(mode: int) -> EntryKind:
    if stat.S_ISLNK(mode):
        return EntryKind.LINK
    if stat.S_ISDIR(mode):
        return EntryKind.DIRECTORY
    return EntryKind.OTHER


def open_regular_file(path: PurePosixPath) -> int:
    try:
        descriptor = os.open(path, os.O_RDONLY | OPEN_WITHOUT_FOLLOWING)
    except OSError as error:
        raise SourceFileError(f"{path} cannot be opened as a regular file: {error}") from error
    if not stat.S_ISREG(os.fstat(descriptor).st_mode):
        os.close(descriptor)
        raise SourceFileError(f"{path} is not a regular file")
    return descriptor
