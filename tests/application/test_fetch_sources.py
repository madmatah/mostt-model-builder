import contextlib
import hashlib
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import partial
from pathlib import PurePosixPath

import pytest
from source_fakes import FakeSourceFiles

from mostt_model_builder.application.fetch_sources import (
    DownloadError,
    FetchError,
    FetchOutcome,
    fetch_source,
)
from mostt_model_builder.domain.containers.work_directory import WorkDirectory
from mostt_model_builder.domain.sources.source_pins import RepositoryType, SourcePin

WORK_ROOT = PurePosixPath("/data/builds/work")
WORK_DIRECTORY = WorkDirectory(str(WORK_ROOT))
REVISION = "70bb2e84b976b7e960aa89f1c648e09c59f894dd"
PINNED_CONTENT = b"id\tfile\ttranscription\n" * 64
OTHER_CONTENT = b"x" * len(PINNED_CONTENT)
PARTIAL_LENGTH_BYTES = 300
FAILED_TRANSFER_LENGTH_BYTES = 10
PIN = SourcePin(
    repository_type=RepositoryType.DATASET,
    repository="google/fleurs",
    revision=REVISION,
    path="data/fr_fr/test.tsv",
    size_bytes=len(PINNED_CONTENT),
    sha256=hashlib.sha256(PINNED_CONTENT).hexdigest(),
)
URL = f"https://huggingface.co/datasets/google/fleurs/resolve/{REVISION}/data/fr_fr/test.tsv"
SOURCE_DIRECTORY = WORK_ROOT / f"sources/google/fleurs/{REVISION}/data/fr_fr"
TARGET = SOURCE_DIRECTORY / "test.tsv"
PARTIAL = SOURCE_DIRECTORY / "test.tsv.partial"


@dataclass(frozen=True)
class Download:
    url: str
    partial_path: PurePosixPath
    size_bytes: int
    resumed_from_bytes: int


@dataclass
class FakeDownloader:
    files: FakeSourceFiles
    served_content: bytes
    is_failing: bool = False
    downloads: list[Download] = field(default_factory=list)

    def download(self, url: str, partial_path: PurePosixPath, size_bytes: int) -> None:
        received = self.files.contents.get(partial_path, b"")
        self.downloads.append(Download(url, partial_path, size_bytes, len(received)))
        if self.is_failing:
            end = len(received) + FAILED_TRANSFER_LENGTH_BYTES
            self.files.write(partial_path, received + self.served_content[len(received) : end])
            raise DownloadError("the transfer broke")
        self.files.write(partial_path, received + self.served_content[len(received) :])


@dataclass(frozen=True)
class FetchRecord:
    outcome: FetchOutcome
    downloads: list[Download]
    contents: dict[PurePosixPath, bytes]
    read_only_files: set[PurePosixPath]


def run_fetch(files: FakeSourceFiles, downloader: FakeDownloader) -> FetchRecord:
    outcome = fetch_source(PIN, WORK_DIRECTORY, downloader, files)
    return FetchRecord(outcome, downloader.downloads, files.contents, files.read_only_files)


def build_files_with_the_source_directory(contents: dict[PurePosixPath, bytes]) -> FakeSourceFiles:
    files = FakeSourceFiles(WORK_ROOT, contents=contents)
    files.make_directories(WORK_ROOT, SOURCE_DIRECTORY.relative_to(WORK_ROOT))
    return files


def match_exactly(message: str) -> str:
    return f"^{re.escape(message)}$"


class TestTheFetchSources:
    class TestTheFetchSourceFunction:
        class TestWhenNothingIsFetchedYet:
            @pytest.fixture
            def fetch_record(self) -> FetchRecord:
                files = FakeSourceFiles(WORK_ROOT)
                return run_fetch(files, FakeDownloader(files, PINNED_CONTENT))

            def test_should_download_the_pinned_url_into_a_partial_file_bounded_by_the_size(
                self, fetch_record: FetchRecord
            ) -> None:
                assert fetch_record.downloads == [Download(URL, PARTIAL, PIN.size_bytes, 0)]

            def test_should_give_the_checked_file_its_name_in_the_layout(
                self, fetch_record: FetchRecord
            ) -> None:
                assert fetch_record.contents == {TARGET: PINNED_CONTENT}

            def test_should_leave_it_read_only(self, fetch_record: FetchRecord) -> None:
                assert fetch_record.read_only_files == {TARGET}

            def test_should_report_it_fetched(self, fetch_record: FetchRecord) -> None:
                assert fetch_record.outcome is FetchOutcome.FETCHED

        class TestWhenAnEarlierDownloadStoppedHalfway:
            @pytest.fixture
            def fetch_record(self) -> FetchRecord:
                files = build_files_with_the_source_directory(
                    {PARTIAL: PINNED_CONTENT[:PARTIAL_LENGTH_BYTES]}
                )
                return run_fetch(files, FakeDownloader(files, PINNED_CONTENT))

            def test_should_resume_it_where_it_stopped(self, fetch_record: FetchRecord) -> None:
                assert fetch_record.downloads == [
                    Download(URL, PARTIAL, PIN.size_bytes, PARTIAL_LENGTH_BYTES)
                ]

            def test_should_keep_the_whole_file_once_checked(
                self, fetch_record: FetchRecord
            ) -> None:
                assert fetch_record.contents == {TARGET: PINNED_CONTENT}

        class TestWhenAnEarlierDownloadEndedBeforeItsCheck:
            @pytest.fixture
            def fetch_record(self) -> FetchRecord:
                files = build_files_with_the_source_directory({PARTIAL: PINNED_CONTENT})
                return run_fetch(files, FakeDownloader(files, PINNED_CONTENT))

            def test_should_download_nothing(self, fetch_record: FetchRecord) -> None:
                assert fetch_record.downloads == []

            def test_should_keep_the_file_once_checked(self, fetch_record: FetchRecord) -> None:
                assert fetch_record.contents == {TARGET: PINNED_CONTENT}

        class TestWhenTheDownloadDoesNotMatchItsPin:
            @pytest.fixture
            def fetch_act(self) -> Callable[[], FetchRecord]:
                files = FakeSourceFiles(WORK_ROOT)
                return partial(run_fetch, files, FakeDownloader(files, OTHER_CONTENT))

            @pytest.fixture
            def contents_after_the_fetch(self) -> dict[PurePosixPath, bytes]:
                files = FakeSourceFiles(WORK_ROOT)
                with contextlib.suppress(FetchError):
                    run_fetch(files, FakeDownloader(files, OTHER_CONTENT))
                return files.contents

            def test_should_refuse_it_and_give_both_sizes_and_hashes(
                self, fetch_act: Callable[[], FetchRecord]
            ) -> None:
                with pytest.raises(
                    FetchError,
                    match=match_exactly(
                        f"the download of {TARGET.relative_to(WORK_ROOT)} does not match its "
                        f"pin, so it was removed: {len(OTHER_CONTENT)} bytes of SHA-256 "
                        f"{hashlib.sha256(OTHER_CONTENT).hexdigest()}, where the pin gives "
                        f"{PIN.size_bytes} bytes of SHA-256 {PIN.sha256}"
                    ),
                ):
                    fetch_act()

            def test_should_remove_it_and_name_no_file_after_it(
                self, contents_after_the_fetch: dict[PurePosixPath, bytes]
            ) -> None:
                assert contents_after_the_fetch == {}

        class TestWhenTheTransferBreaks:
            @pytest.fixture
            def fetch_act(self) -> Callable[[], FetchRecord]:
                files = FakeSourceFiles(WORK_ROOT)
                return partial(
                    run_fetch, files, FakeDownloader(files, PINNED_CONTENT, is_failing=True)
                )

            @pytest.fixture
            def contents_after_the_fetch(self) -> dict[PurePosixPath, bytes]:
                files = FakeSourceFiles(WORK_ROOT)
                with contextlib.suppress(DownloadError):
                    run_fetch(files, FakeDownloader(files, PINNED_CONTENT, is_failing=True))
                return files.contents

            def test_should_fail_with_the_downloaders_error(
                self, fetch_act: Callable[[], FetchRecord]
            ) -> None:
                with pytest.raises(DownloadError, match=match_exactly("the transfer broke")):
                    fetch_act()

            def test_should_keep_the_partial_file_for_the_next_fetch_to_resume(
                self, contents_after_the_fetch: dict[PurePosixPath, bytes]
            ) -> None:
                assert contents_after_the_fetch == {
                    PARTIAL: PINNED_CONTENT[:FAILED_TRANSFER_LENGTH_BYTES]
                }

        class TestWhenTheSourceIsAlreadyFetched:
            @pytest.fixture
            def fetch_record(self) -> FetchRecord:
                files = build_files_with_the_source_directory({TARGET: PINNED_CONTENT})
                return run_fetch(files, FakeDownloader(files, PINNED_CONTENT))

            def test_should_download_nothing(self, fetch_record: FetchRecord) -> None:
                assert fetch_record.downloads == []

            def test_should_report_it_already_fetched(self, fetch_record: FetchRecord) -> None:
                assert fetch_record.outcome is FetchOutcome.ALREADY_FETCHED

            def test_should_leave_it_read_only(self, fetch_record: FetchRecord) -> None:
                assert fetch_record.read_only_files == {TARGET}

        class TestWhenAFetchedSourceNoLongerMatchesItsPin:
            @pytest.fixture
            def fetch_act(self) -> Callable[[], FetchRecord]:
                files = build_files_with_the_source_directory({TARGET: OTHER_CONTENT})
                return partial(run_fetch, files, FakeDownloader(files, PINNED_CONTENT))

            @pytest.fixture
            def contents_after_the_fetch(self) -> dict[PurePosixPath, bytes]:
                files = build_files_with_the_source_directory({TARGET: OTHER_CONTENT})
                with contextlib.suppress(FetchError):
                    run_fetch(files, FakeDownloader(files, PINNED_CONTENT))
                return files.contents

            def test_should_refuse_it_and_ask_for_its_removal(
                self, fetch_act: Callable[[], FetchRecord]
            ) -> None:
                with pytest.raises(
                    FetchError,
                    match=f"^{re.escape(str(TARGET))} does not match its pin: .*; "
                    "remove it, and fetch again$",
                ):
                    fetch_act()

            def test_should_leave_it_for_the_user_to_look_at(
                self, contents_after_the_fetch: dict[PurePosixPath, bytes]
            ) -> None:
                assert contents_after_the_fetch == {TARGET: OTHER_CONTENT}
