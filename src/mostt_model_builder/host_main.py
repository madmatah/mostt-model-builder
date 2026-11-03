import argparse
import os
import pwd
import sys
from collections.abc import Sequence
from pathlib import Path

from mostt_model_builder.adapters.curl_downloader import CurlDownloader
from mostt_model_builder.adapters.work_files import WorkFiles
from mostt_model_builder.application.fetch_sources import (
    DownloadError,
    FetchError,
    FetchOutcome,
    SourceFileError,
    fetch_source,
)
from mostt_model_builder.application.open_work_directory import (
    WorkDirectoryRequest,
    open_work_directory,
)
from mostt_model_builder.domain.containers.work_directory import (
    WorkDirectory,
    WorkDirectoryError,
    locate_source,
)
from mostt_model_builder.domain.sources.source_pins import SourcePinError, parse_source_pins

PROGRAM_NAME = "model-build.sh"
DESCRIPTION = (
    "Builds speech recognition models from their original weights, in confined containers, "
    "and verifies them on FLEURS."
)
CLONE_ROOT = Path(__file__).resolve().parents[2]
SOURCES_FILE_NAME = "sources.toml"
FETCH_COMMAND = "fetch"
FAILURE_EXIT_STATUS = 1
NO_ACCOUNT_HOME = ""
OUTCOME_WORDS = {FetchOutcome.FETCHED: "fetched", FetchOutcome.ALREADY_FETCHED: "already fetched"}
COMMAND_ERRORS = (
    SourcePinError,
    WorkDirectoryError,
    FetchError,
    DownloadError,
    SourceFileError,
    OSError,
)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=PROGRAM_NAME, description=DESCRIPTION, add_help=False, allow_abbrev=False
    )
    parser.add_argument("--help", action="help", help="show this message and exit")
    parser.add_argument(
        "--work",
        required=True,
        metavar="<dir>",
        help="the work directory: an existing directory of yours, in mode 0700, outside the "
        "home directory and the clone",
    )
    commands = parser.add_subparsers(dest="command", required=True, metavar="<command>")
    commands.add_parser(
        FETCH_COMMAND,
        add_help=False,
        help="download the pinned sources into the work directory, each checked by its size "
        "and SHA-256",
    )
    return parser


def find_account_home() -> str:
    try:
        return pwd.getpwuid(os.getuid()).pw_dir
    except KeyError:
        return NO_ACCOUNT_HOME


def open_requested_work_directory(path: str, files: WorkFiles) -> WorkDirectory:
    request = WorkDirectoryRequest(
        path=path,
        home_variable=os.environ.get("HOME"),
        account_home=find_account_home(),
        clone_root=str(CLONE_ROOT),
        user_id=os.getuid(),
    )
    return open_work_directory(request, files)


def fetch(work_path: str) -> None:
    files = WorkFiles()
    work_directory = open_requested_work_directory(work_path, files)
    pins = parse_source_pins((CLONE_ROOT / SOURCES_FILE_NAME).read_text(encoding="utf-8"))
    downloader = CurlDownloader()
    for pin in pins:
        source_path = locate_source(pin)
        print(f"checking {source_path}, {pin.size_bytes} bytes", flush=True)
        outcome = fetch_source(pin, work_directory, downloader, files)
        print(f"{OUTCOME_WORDS[outcome]} {source_path}", flush=True)


def main(arguments: Sequence[str]) -> None:
    parsed_arguments = build_argument_parser().parse_args(arguments)
    try:
        fetch(parsed_arguments.work)
    except COMMAND_ERRORS as error:
        print(f"model-build: {error}", file=sys.stderr)
        raise SystemExit(FAILURE_EXIT_STATUS) from error


if __name__ == "__main__":
    main(sys.argv[1:])
