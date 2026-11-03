import os
import re
from collections.abc import Callable
from functools import partial
from pathlib import Path, PurePosixPath

import pytest

from mostt_model_builder.adapters.curl_downloader import CurlDownloader, build_curl_arguments
from mostt_model_builder.application.fetch_sources import DownloadError

URL = "https://huggingface.co/datasets/google/fleurs/resolve/0123/data/fr_fr/test.tsv"
PARTIAL_PATH = PurePosixPath("/data/builds/work/sources/test.tsv.partial")
SIZE_BYTES = 456972
FAILING_CURL_STATUS = 22
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
