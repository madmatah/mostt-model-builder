from dataclasses import dataclass
from enum import Enum, auto
from pathlib import PurePosixPath
from typing import Protocol

from mostt_model_builder.domain.containers.work_directory import WorkDirectory, locate_source
from mostt_model_builder.domain.sources.source_pins import SourcePin, is_pinned_content
from mostt_model_builder.domain.sources.source_urls import build_source_url

PARTIAL_SUFFIX = ".partial"


class FetchError(Exception):
    pass


class DownloadError(Exception):
    pass


class SourceFileError(Exception):
    pass


@dataclass(frozen=True, slots=True)
class FileDigest:
    size_bytes: int
    sha256: str


class SourceDownloader(Protocol):
    def download(self, url: str, partial_path: PurePosixPath, size_bytes: int) -> None: ...


class SourceFiles(Protocol):
    def make_directories(self, root: PurePosixPath, relative_path: PurePosixPath) -> None: ...

    def find_file_size(self, path: PurePosixPath) -> int | None: ...

    def digest_file(self, path: PurePosixPath) -> FileDigest: ...

    def rename_file(self, source: PurePosixPath, target: PurePosixPath) -> None: ...

    def remove_file(self, path: PurePosixPath) -> None: ...

    def make_read_only(self, path: PurePosixPath) -> None: ...


class FetchOutcome(Enum):
    FETCHED = auto()
    ALREADY_FETCHED = auto()


def fetch_source(
    pin: SourcePin,
    work_directory: WorkDirectory,
    downloader: SourceDownloader,
    files: SourceFiles,
) -> FetchOutcome:
    root = PurePosixPath(work_directory.path)
    source_path = locate_source(pin)
    target = root / source_path
    if files.find_file_size(target) is not None:
        refuse_unpinned_content(pin, target, files.digest_file(target))
        files.make_read_only(target)
        return FetchOutcome.ALREADY_FETCHED
    files.make_directories(root, source_path.parent)
    partial = target.with_name(target.name + PARTIAL_SUFFIX)
    partial_size_bytes = files.find_file_size(partial)
    if partial_size_bytes is None or partial_size_bytes < pin.size_bytes:
        downloader.download(build_source_url(pin), partial, pin.size_bytes)
    partial_digest = files.digest_file(partial)
    if not is_pinned_content(pin, partial_digest.size_bytes, partial_digest.sha256):
        files.remove_file(partial)
        raise FetchError(
            f"the download of {source_path} does not match its pin, so it was removed: "
            f"{describe_mismatch(pin, partial_digest)}"
        )
    files.rename_file(partial, target)
    files.make_read_only(target)
    return FetchOutcome.FETCHED


def refuse_unpinned_content(pin: SourcePin, target: PurePosixPath, digest: FileDigest) -> None:
    if not is_pinned_content(pin, digest.size_bytes, digest.sha256):
        raise FetchError(
            f"{target} does not match its pin: {describe_mismatch(pin, digest)}; "
            "remove it, and fetch again"
        )


def describe_mismatch(pin: SourcePin, digest: FileDigest) -> str:
    return (
        f"{digest.size_bytes} bytes of SHA-256 {digest.sha256}, "
        f"where the pin gives {pin.size_bytes} bytes of SHA-256 {pin.sha256}"
    )
