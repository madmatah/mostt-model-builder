import subprocess
from pathlib import PurePosixPath

from mostt_model_builder.application.fetch_sources import DownloadError

CURL_PROGRAM = "curl"
RETRY_COUNT = 3


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
