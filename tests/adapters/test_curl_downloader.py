import os
import re
from collections.abc import Callable
from functools import partial
from pathlib import Path, PurePosixPath

import pytest

from mostt_model_builder.adapters.curl_downloader import (
    CurlDownloader,
    build_curl_arguments,
    build_listing_arguments,
)
from mostt_model_builder.application.check_published import RefusedListingError
from mostt_model_builder.application.fetch_sources import DownloadError

URL = "https://huggingface.co/datasets/google/fleurs/resolve/0123/data/fr_fr/test.tsv"
PARTIAL_PATH = PurePosixPath("/data/builds/work/sources/test.tsv.partial")
SIZE_BYTES = 456972
FAILING_CURL_STATUS = 22
TIMED_OUT_CURL_STATUS = 28
FIRST_ARGUMENTS = ["curl", "--disable"]
LAST_ARGUMENTS = ["--", URL]
ARGUMENT_SEQUENCES = {
    "no configuration file of the user's, which only the first option can turn off": (
        FIRST_ARGUMENTS
    ),
    "https alone, redirects included, over TLS 1.2 or later": [
        "--proto",
        "=https",
        "--proto-redir",
        "=https",
        "--tlsv1.2",
    ],
    "no globbing of the url": ["--globoff"],
    "an error status from the server fails the transfer": ["--fail"],
    "redirects followed, as the hub sends its files to its storage": ["--location"],
    "errors shown without a progress meter": ["--silent", "--show-error"],
    "three retries of a transient failure": ["--retry", "3"],
    "no more bytes than the pin gives": ["--max-filesize", str(SIZE_BYTES)],
    "a transfer resumed where the partial file ends": ["--continue-at", "-"],
    "the partial file as the output": ["--output", str(PARTIAL_PATH)],
    "the url after the end of the options": LAST_ARGUMENTS,
}


LISTING_URL = (
    "https://huggingface.co/api/models/example-user/parakeet-ultra-onnx/tree/"
    "0123456789abcdef0123456789abcdef01234567"
)
LISTING_MAX_SIZE_BYTES = 2**20
LISTING = b'[{"type":"file","path":"README.md","size":12}]'
LISTING_ARGUMENT_SEQUENCES = {
    "no configuration file of the user's, which only the first option can turn off": (
        FIRST_ARGUMENTS
    ),
    "https alone, redirects included, over TLS 1.2 or later": [
        "--proto",
        "=https",
        "--proto-redir",
        "=https",
        "--tlsv1.2",
    ],
    "no globbing of the url": ["--globoff"],
    "an error status from the server fails the transfer": ["--fail"],
    "redirects followed": ["--location"],
    "errors shown without a progress meter": ["--silent", "--show-error"],
    "no more bytes than the listing's bound": ["--max-filesize", str(LISTING_MAX_SIZE_BYTES)],
    "a single attempt bounded in time, since a retry would restart the bound": [
        "--max-time",
        "60",
    ],
    "the url after the end of the options": ["--", LISTING_URL],
}


def write_printing_curl(directory: Path, output: bytes, status: int) -> None:
    (directory / "listing").write_bytes(output)
    fake_curl = directory / "curl"
    fake_curl.write_text(
        f"#!/usr/bin/env bash\ncat -- '{directory / 'listing'}'\necho 'curl: (22) refused' >&2\n"
        f"exit {status}\n"
    )
    fake_curl.chmod(0o755)


def read_with_fake_curl(
    directory: Path,
    monkeypatch: pytest.MonkeyPatch,
    output: bytes,
    status: int,
    max_size_bytes: int,
) -> Callable[[], bytes]:
    write_printing_curl(directory, output, status)
    monkeypatch.setenv("PATH", f"{directory}{os.pathsep}{os.environ['PATH']}")
    return partial(CurlDownloader().read_listing, LISTING_URL, max_size_bytes)


def find_sequence(arguments: list[str], sequence: list[str]) -> int | None:
    for start in range(len(arguments) - len(sequence) + 1):
        if arguments[start : start + len(sequence)] == sequence:
            return start
    return None


def write_failing_curl(directory: Path) -> None:
    fake_curl = directory / "curl"
    fake_curl.write_text(f"#!/usr/bin/env bash\nexit {FAILING_CURL_STATUS}\n")
    fake_curl.chmod(0o755)


class TestTheCurlDownloader:
    class TestTheBuildCurlArgumentsFunction:
        @pytest.mark.parametrize(
            "sequence", list(ARGUMENT_SEQUENCES.values()), ids=list(ARGUMENT_SEQUENCES.keys())
        )
        class TestWhenBuiltForAPin:
            @pytest.fixture
            def curl_arguments(self) -> list[str]:
                return build_curl_arguments(URL, PARTIAL_PATH, SIZE_BYTES)

            def test_should_hold_each_option_of_a_safe_resumable_download(
                self, curl_arguments: list[str], sequence: list[str]
            ) -> None:
                assert find_sequence(curl_arguments, sequence) is not None

        class TestWhenBuiltForAnyPin:
            @pytest.fixture
            def curl_arguments(self) -> list[str]:
                return build_curl_arguments(URL, PARTIAL_PATH, SIZE_BYTES)

            def test_should_start_with_disable_which_curl_honors_only_as_its_first_option(
                self, curl_arguments: list[str]
            ) -> None:
                assert curl_arguments[: len(FIRST_ARGUMENTS)] == FIRST_ARGUMENTS

            def test_should_end_with_the_url_after_the_end_of_the_options(
                self, curl_arguments: list[str]
            ) -> None:
                assert curl_arguments[-len(LAST_ARGUMENTS) :] == LAST_ARGUMENTS

            def test_should_hold_no_other_argument(self, curl_arguments: list[str]) -> None:
                assert len(curl_arguments) == sum(
                    len(sequence) for sequence in ARGUMENT_SEQUENCES.values()
                )

    class TestTheDownloadMethod:
        class TestWhenCurlFails:
            @pytest.fixture
            def download_act(
                self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
            ) -> Callable[[], None]:
                write_failing_curl(tmp_path)
                monkeypatch.setenv("PATH", f"{tmp_path}{os.pathsep}{os.environ['PATH']}")
                return partial(
                    CurlDownloader().download,
                    URL,
                    PurePosixPath(tmp_path / "test.tsv.partial"),
                    SIZE_BYTES,
                )

            def test_should_fail_with_its_status_and_say_a_fetch_resumes_it(
                self, download_act: Callable[[], None]
            ) -> None:
                with pytest.raises(
                    DownloadError,
                    match=re.escape(
                        f"curl exited with the status {FAILING_CURL_STATUS} while downloading "
                        f"{URL}; fetch again to resume it"
                    ),
                ):
                    download_act()

    class TestTheBuildListingArgumentsFunction:
        @pytest.mark.parametrize(
            "sequence",
            list(LISTING_ARGUMENT_SEQUENCES.values()),
            ids=list(LISTING_ARGUMENT_SEQUENCES.keys()),
        )
        class TestWhenBuiltForATreeUrl:
            @pytest.fixture
            def listing_arguments(self) -> list[str]:
                return build_listing_arguments(LISTING_URL, LISTING_MAX_SIZE_BYTES)

            def test_should_hold_each_option_of_a_safe_bounded_read(
                self, listing_arguments: list[str], sequence: list[str]
            ) -> None:
                assert find_sequence(listing_arguments, sequence) is not None

        class TestWhenBuiltForAnyTreeUrl:
            @pytest.fixture
            def listing_arguments(self) -> list[str]:
                return build_listing_arguments(LISTING_URL, LISTING_MAX_SIZE_BYTES)

            def test_should_hold_no_other_argument_and_start_with_disable(
                self, listing_arguments: list[str]
            ) -> None:
                assert (
                    listing_arguments[: len(FIRST_ARGUMENTS)],
                    len(listing_arguments),
                ) == (
                    FIRST_ARGUMENTS,
                    sum(len(sequence) for sequence in LISTING_ARGUMENT_SEQUENCES.values()),
                )

    class TestTheReadListingMethod:
        class TestWhenCurlPrintsAListingWithinItsBound:
            @pytest.fixture
            def listing(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> bytes:
                return read_with_fake_curl(
                    tmp_path, monkeypatch, LISTING, 0, LISTING_MAX_SIZE_BYTES
                )()

            def test_should_give_its_bytes(self, listing: bytes) -> None:
                assert listing == LISTING

        class TestWhenCurlPrintsMoreThanTheBound:
            @pytest.fixture
            def read_act(
                self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
            ) -> Callable[[], bytes]:
                return read_with_fake_curl(tmp_path, monkeypatch, LISTING, 0, len(LISTING) - 1)

            def test_should_refuse_it_without_reading_the_rest(
                self, read_act: Callable[[], bytes]
            ) -> None:
                with pytest.raises(
                    DownloadError, match=re.escape(f"holds more than {len(LISTING) - 1} bytes")
                ):
                    read_act()

        class TestWhenCurlFails:
            @pytest.fixture
            def read_act(
                self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
            ) -> Callable[[], bytes]:
                return read_with_fake_curl(
                    tmp_path, monkeypatch, b"", FAILING_CURL_STATUS, LISTING_MAX_SIZE_BYTES
                )

            def test_should_fail_with_its_status_and_its_message(
                self, read_act: Callable[[], bytes]
            ) -> None:
                with pytest.raises(
                    DownloadError,
                    match=re.escape(
                        f"curl exited with the status {FAILING_CURL_STATUS} while reading "
                        f"{LISTING_URL}: curl: (22) refused"
                    ),
                ):
                    read_act()

            def test_should_say_the_hub_refused_the_listing(
                self, read_act: Callable[[], bytes]
            ) -> None:
                with pytest.raises(RefusedListingError, match="exited with the status"):
                    read_act()

        class TestWhenCurlFailsOtherwiseThanOnAnHttpError:
            @pytest.fixture
            def read_act(
                self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
            ) -> Callable[[], bytes]:
                return read_with_fake_curl(
                    tmp_path, monkeypatch, b"", TIMED_OUT_CURL_STATUS, LISTING_MAX_SIZE_BYTES
                )

            def test_should_fail_with_its_status_and_its_message_as_no_refusal_of_the_hub(
                self, read_act: Callable[[], bytes]
            ) -> None:
                with pytest.raises(
                    DownloadError,
                    match=re.escape(
                        f"curl exited with the status {TIMED_OUT_CURL_STATUS} while reading "
                        f"{LISTING_URL}: curl: (22) refused"
                    ),
                ) as failure:
                    read_act()
                assert failure.type is DownloadError
