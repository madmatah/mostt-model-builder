import subprocess
from pathlib import PurePosixPath

from mostt_model_builder.application.fetch_sources import DownloadError
from mostt_model_builder.domain.containers.printable_output import show_printable

CURL_PROGRAM = "curl"
RETRY_COUNT = 3
LISTING_TIMEOUT_S = 60
MAX_ERROR_BYTES = 4096


def build_curl_arguments(url: str, partial_path: PurePosixPath, size_bytes: int) -> list[str]:
    return [
        CURL_PROGRAM,
        # curl honors it only as the first option, and then reads no configuration file of the
        # user's, which could add options of its own.
        "--disable",
        "--proto",
        "=https",
        "--proto-redir",
        "=https",
        "--tlsv1.2",
        "--globoff",
        "--fail",
        "--location",
        "--silent",
        "--show-error",
        "--retry",
        str(RETRY_COUNT),
        "--max-filesize",
        str(size_bytes),
        "--continue-at",
        "-",
        "--output",
        str(partial_path),
        "--",
        url,
    ]


def build_listing_arguments(url: str, max_size_bytes: int) -> list[str]:
    return [
        CURL_PROGRAM,
        # curl honors it only as the first option.
        "--disable",
        "--proto",
        "=https",
        "--proto-redir",
        "=https",
        "--tlsv1.2",
        "--globoff",
        "--fail",
        "--location",
        "--silent",
        "--show-error",
        "--max-filesize",
        str(max_size_bytes),
        "--max-time",
        str(LISTING_TIMEOUT_S),
        "--",
        url,
    ]


class CurlDownloader:
    def download(self, url: str, partial_path: PurePosixPath, size_bytes: int) -> None:
        try:
            completed_download = subprocess.run(
                build_curl_arguments(url, partial_path, size_bytes),
                check=False,
                stdin=subprocess.DEVNULL,
            )
        except FileNotFoundError as error:
            raise DownloadError(f"{CURL_PROGRAM} is not installed") from error
        if completed_download.returncode != 0:
            raise DownloadError(
                f"{CURL_PROGRAM} exited with the status {completed_download.returncode} "
                f"while downloading {url}; fetch again to resume it"
            )

    def read_listing(self, url: str, max_size_bytes: int) -> bytes:
        try:
            process = subprocess.Popen(
                build_listing_arguments(url, max_size_bytes),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except FileNotFoundError as error:
            raise DownloadError(f"{CURL_PROGRAM} is not installed") from error
        with process:
            if process.stdout is None or process.stderr is None:
                raise DownloadError(f"{CURL_PROGRAM} was started without its pipes")
            listing = process.stdout.read(max_size_bytes + 1)
            if len(listing) > max_size_bytes:
                process.kill()
                raise DownloadError(f"the listing at {url} holds more than {max_size_bytes} bytes")
            error_output = process.stderr.read(MAX_ERROR_BYTES)
            status = process.wait()
        if status != 0:
            raise DownloadError(
                f"{CURL_PROGRAM} exited with the status {status} while reading {url}: "
                f"{show_printable(error_output).strip()}"
            )
        return listing
