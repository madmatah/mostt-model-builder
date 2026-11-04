import contextlib
import errno
import hashlib
import os
import stat
from collections.abc import Callable, Mapping
from dataclasses import replace
from pathlib import PurePosixPath
from typing import BinaryIO

from mostt_model_builder.application.assemble_candidate import HashedFile
from mostt_model_builder.application.fetch_sources import FileDigest, SourceFileError
from mostt_model_builder.application.run_step import WrittenFile, WrittenOutputs
from mostt_model_builder.domain.containers.container_files import (
    FileFacts,
    FileKind,
    PathComponent,
)
from mostt_model_builder.domain.containers.work_directory import DirectoryFacts, EntryKind

CREATED_DIRECTORY_MODE = 0o700
READ_ONLY_FILE_MODE = 0o444
REPLACED_FILE_MODE = 0o644
DIGEST_ALGORITHM = "sha256"
NO_OWNER_ID = -1
NO_PERMISSION_BITS = 0
OPEN_WITHOUT_FOLLOWING = os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC
OPEN_DIRECTORY_WITHOUT_FOLLOWING = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
REPLACEMENT_SUFFIX = ".new"
CREATE_EXCLUSIVELY = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC
PRIVATE_FILE_MODE = 0o600
COPY_CHUNK_BYTES = 8 * 2**20
NO_FACTS = FileFacts(FileKind.MISSING, NO_OWNER_ID, NO_PERMISSION_BITS, 0, 0)


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

    def make_step_directory(self, path: PurePosixPath, mode: int) -> bool:
        try:
            os.mkdir(path, CREATED_DIRECTORY_MODE)
        except FileExistsError:
            return False
        os.chmod(path, mode, follow_symlinks=False)
        return True

    def write_new_file(self, path: PurePosixPath, content: bytes, mode: int) -> None:
        descriptor = os.open(
            path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC, mode
        )
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
            os.fchmod(stream.fileno(), mode)

    def inspect_components(
        self, root: PurePosixPath, relative_path: PurePosixPath
    ) -> tuple[PathComponent, ...]:
        components = []
        path = root
        for part in relative_path.parts:
            path = path / part
            facts = inspect_entry(path)
            components.append(PathComponent(path, facts))
            if facts.kind is not FileKind.DIRECTORY:
                break
        return tuple(components)

    def read_written_outputs(
        self,
        directory: PurePosixPath,
        max_sizes_bytes: Mapping[str, int],
        parsed_names: frozenset[str],
    ) -> WrittenOutputs:
        directory_descriptor = os.open(directory, OPEN_DIRECTORY_WITHOUT_FOLLOWING)
        try:
            entry_names = tuple(sorted(os.listdir(directory_descriptor)))
            files = {
                name: read_written_file(
                    directory_descriptor, name, max_size_bytes, is_parsed=name in parsed_names
                )
                for name, max_size_bytes in max_sizes_bytes.items()
                if name in entry_names
            }
        finally:
            os.close(directory_descriptor)
        return WrittenOutputs(entry_names, files)

    def copy_written_file(
        self,
        directory: PurePosixPath,
        name: str,
        max_size_bytes: int,
        target: PurePosixPath,
        observe_chunk: Callable[[bytes], None],
    ) -> FileFacts:
        directory_descriptor = os.open(directory, OPEN_DIRECTORY_WITHOUT_FOLLOWING)
        try:
            entry_facts = describe_status(
                os.stat(name, dir_fd=directory_descriptor, follow_symlinks=False)
            )
            if entry_facts.kind is not FileKind.REGULAR:
                return entry_facts
            descriptor = os.open(
                name, os.O_RDONLY | OPEN_WITHOUT_FOLLOWING, dir_fd=directory_descriptor
            )
        finally:
            os.close(directory_descriptor)
        with os.fdopen(descriptor, "rb") as stream:
            facts = describe_status(os.fstat(stream.fileno()))
            if facts.kind is not FileKind.REGULAR or facts.size_bytes > max_size_bytes:
                return facts
            copy_stream(stream, target, max_size_bytes, observe_chunk)
        return facts

    def read_host_file(
        self, directory: PurePosixPath, name: str, max_size_bytes: int
    ) -> WrittenFile:
        directory_descriptor = os.open(directory, OPEN_DIRECTORY_WITHOUT_FOLLOWING)
        try:
            return read_written_file(directory_descriptor, name, max_size_bytes, is_parsed=True)
        except FileNotFoundError:
            return WrittenFile(NO_FACTS, None)
        finally:
            os.close(directory_descriptor)

    def digest_host_file(
        self, directory: PurePosixPath, name: str, max_size_bytes: int
    ) -> HashedFile:
        directory_descriptor = os.open(directory, OPEN_DIRECTORY_WITHOUT_FOLLOWING)
        try:
            entry_facts, descriptor = open_regular_entry(directory_descriptor, name)
        except FileNotFoundError:
            return HashedFile(NO_FACTS, None)
        finally:
            os.close(directory_descriptor)
        if descriptor is None:
            return HashedFile(entry_facts, None)
        with os.fdopen(descriptor, "rb") as stream:
            facts = describe_status(os.fstat(stream.fileno()))
            if facts.kind is not FileKind.REGULAR or facts.size_bytes > max_size_bytes:
                return HashedFile(facts, None)
            return hash_stream(stream, facts, max_size_bytes)

    def list_entry_names(self, directory: PurePosixPath) -> tuple[str, ...]:
        directory_descriptor = os.open(directory, OPEN_DIRECTORY_WITHOUT_FOLLOWING)
        try:
            return tuple(sorted(os.listdir(directory_descriptor)))
        finally:
            os.close(directory_descriptor)

    def replace_file(self, path: PurePosixPath, content: bytes) -> None:
        replacement = path.with_name(path.name + REPLACEMENT_SUFFIX)
        with contextlib.suppress(FileNotFoundError):
            os.unlink(replacement)
        descriptor = os.open(
            replacement,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
            REPLACED_FILE_MODE,
        )
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(content)
            os.replace(replacement, path)
        except BaseException:
            with contextlib.suppress(FileNotFoundError):
                os.unlink(replacement)
            raise

    def remove_empty_directory(self, path: PurePosixPath) -> None:
        os.rmdir(path)

    def remove_request_directory(self, path: PurePosixPath) -> None:
        directory_descriptor = os.open(path, OPEN_DIRECTORY_WITHOUT_FOLLOWING)
        try:
            for name in os.listdir(directory_descriptor):
                os.unlink(name, dir_fd=directory_descriptor)
        finally:
            os.close(directory_descriptor)
        os.rmdir(path)

    def remove_file_if_present(self, path: PurePosixPath) -> bool:
        try:
            os.unlink(path)
        except FileNotFoundError:
            return False
        return True


def describe_status(entry_status: os.stat_result) -> FileFacts:
    return FileFacts(
        kind=find_file_kind(entry_status.st_mode),
        owner_id=entry_status.st_uid,
        permission_bits=stat.S_IMODE(entry_status.st_mode),
        size_bytes=entry_status.st_size,
        link_count=entry_status.st_nlink,
    )


def find_file_kind(mode: int) -> FileKind:
    if stat.S_ISREG(mode):
        return FileKind.REGULAR
    if stat.S_ISDIR(mode):
        return FileKind.DIRECTORY
    if stat.S_ISLNK(mode):
        return FileKind.LINK
    if stat.S_ISFIFO(mode):
        return FileKind.FIFO
    if stat.S_ISSOCK(mode):
        return FileKind.SOCKET
    return FileKind.DEVICE


def inspect_entry(path: PurePosixPath) -> FileFacts:
    try:
        return describe_status(os.lstat(path))
    except FileNotFoundError:
        return NO_FACTS


def open_regular_entry(directory_descriptor: int, name: str) -> tuple[FileFacts, int | None]:
    entry_facts = describe_status(os.stat(name, dir_fd=directory_descriptor, follow_symlinks=False))
    if entry_facts.kind is not FileKind.REGULAR:
        return entry_facts, None
    try:
        descriptor = os.open(
            name, os.O_RDONLY | OPEN_WITHOUT_FOLLOWING, dir_fd=directory_descriptor
        )
    except PermissionError:
        return replace_kind(entry_facts, FileKind.UNREADABLE), None
    except OSError as error:
        if error.errno == errno.ELOOP:
            return replace_kind(entry_facts, FileKind.LINK), None
        raise
    return entry_facts, descriptor


def read_written_file(
    directory_descriptor: int, name: str, max_size_bytes: int, *, is_parsed: bool
) -> WrittenFile:
    entry_facts, descriptor = open_regular_entry(directory_descriptor, name)
    if descriptor is None:
        return WrittenFile(entry_facts, None)
    with os.fdopen(descriptor, "rb") as stream:
        facts = describe_status(os.fstat(stream.fileno()))
        if not is_parsed or facts.kind is not FileKind.REGULAR or facts.size_bytes > max_size_bytes:
            return WrittenFile(facts, None)
        content = stream.read(max_size_bytes + 1)
    if len(content) > max_size_bytes:
        return WrittenFile(replace(facts, size_bytes=len(content)), None)
    return WrittenFile(facts, content)


def hash_stream(stream: BinaryIO, facts: FileFacts, max_size_bytes: int) -> HashedFile:
    digest = hashlib.new(DIGEST_ALGORITHM)
    hashed_bytes = 0
    while chunk := stream.read(COPY_CHUNK_BYTES):
        hashed_bytes += len(chunk)
        if hashed_bytes > max_size_bytes:
            return HashedFile(replace(facts, size_bytes=hashed_bytes), None)
        digest.update(chunk)
    return HashedFile(replace(facts, size_bytes=hashed_bytes), digest.hexdigest())


def copy_stream(
    stream: BinaryIO,
    target: PurePosixPath,
    max_size_bytes: int,
    observe_chunk: Callable[[bytes], None],
) -> None:
    target_descriptor = os.open(target, CREATE_EXCLUSIVELY, PRIVATE_FILE_MODE)
    copied_bytes = 0
    with os.fdopen(target_descriptor, "wb") as copy:
        while chunk := stream.read(COPY_CHUNK_BYTES):
            copied_bytes += len(chunk)
            if copied_bytes > max_size_bytes:
                raise SourceFileError(
                    f"{target.name} grew beyond {max_size_bytes} bytes as it was copied"
                )
            observe_chunk(chunk)
            copy.write(chunk)
        os.fchmod(copy.fileno(), READ_ONLY_FILE_MODE)


def replace_kind(facts: FileFacts, kind: FileKind) -> FileFacts:
    return FileFacts(
        kind, facts.owner_id, facts.permission_bits, facts.size_bytes, facts.link_count
    )


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
