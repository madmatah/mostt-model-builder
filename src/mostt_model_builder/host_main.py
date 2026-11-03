import argparse
import os
import pwd
import sys
from collections.abc import Sequence
from pathlib import Path, PurePosixPath

from mostt_model_builder.adapters.curl_downloader import CurlDownloader
from mostt_model_builder.adapters.docker_engine import ContainerEngineError, DockerEngine
from mostt_model_builder.adapters.git_tree import GitError, GitTree
from mostt_model_builder.adapters.work_files import WorkFiles
from mostt_model_builder.application.build_images import ImageBuildError, build_images
from mostt_model_builder.application.check_fleurs import FleursCheckRun, check_fleurs_bench
from mostt_model_builder.application.clean_run import CleanOutcome, CleanRequest, clean_step
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
from mostt_model_builder.application.resolve_lock import (
    LockRequest,
    LockResolution,
    resolve_lock,
)
from mostt_model_builder.application.run_step import HostIdentity, StepError
from mostt_model_builder.domain.bench.fleurs_check_report import FleursCheckReportError
from mostt_model_builder.domain.containers.account_ranges import (
    AccountRangeError,
    HostAccounts,
    parse_subordinate_ranges,
)
from mostt_model_builder.domain.containers.container_files import ContainerFileError
from mostt_model_builder.domain.containers.container_steps import (
    CLEAN_STEP_NAME,
    CONTAINER_USER_ID,
    CORE_IMAGES,
    FLEURS_CHECK_STEP_NAME,
    LOCK_STEP_NAME,
    RUNNABLE_CORE_ROWS,
    StepTableError,
    find_runnable_core_row,
)
from mostt_model_builder.domain.containers.provenance import ProvenanceError
from mostt_model_builder.domain.containers.step_peaks import StepPeaks, StepPeaksError
from mostt_model_builder.domain.containers.work_directory import (
    WorkDirectory,
    WorkDirectoryError,
    locate_source,
)
from mostt_model_builder.domain.lock.lock_check import (
    LOCK_FILE_NAME,
    MANIFEST_FILE_NAME,
    LockError,
)
from mostt_model_builder.domain.lock.lock_freshness import LockFreshnessError
from mostt_model_builder.domain.sources.source_pins import (
    SourcePin,
    SourcePinError,
    parse_source_pins,
)

PROGRAM_NAME = "model-build.sh"
DESCRIPTION = (
    "Builds speech recognition models from their original weights, in confined containers, "
    "and verifies them on FLEURS."
)
CLONE_ROOT = Path(__file__).resolve().parents[2]
SOURCES_FILE_NAME = "sources.toml"
IMAGES_ENVIRONMENT_DIRECTORY = CLONE_ROOT / "images"
SUBORDINATE_USER_IDS_PATH = Path("/etc/subuid")
SUBORDINATE_GROUP_IDS_PATH = Path("/etc/subgid")
FETCH_COMMAND = "fetch"
IMAGES_COMMAND = "images"
STEP_COMMAND = "step"
TEXT_ENCODING = "utf-8"
FAILURE_EXIT_STATUS = 1
NO_ACCOUNT_HOME = ""
OUTCOME_WORDS = {FetchOutcome.FETCHED: "fetched", FetchOutcome.ALREADY_FETCHED: "already fetched"}
CLEAN_OUTCOME_WORDS = {
    CleanOutcome.CLEANED: "cleaned",
    CleanOutcome.NOTHING_TO_CLEAN: "nothing to clean for",
}
COMMAND_ERRORS = (
    SourcePinError,
    WorkDirectoryError,
    FetchError,
    DownloadError,
    SourceFileError,
    StepError,
    StepTableError,
    StepPeaksError,
    ContainerFileError,
    ContainerEngineError,
    AccountRangeError,
    ProvenanceError,
    GitError,
    ImageBuildError,
    FleursCheckReportError,
    LockError,
    LockFreshnessError,
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
    commands.add_parser(
        LOCK_STEP_NAME,
        add_help=False,
        help="resolve the lock of the images' environment in its container, check it, print "
        "its change, and write it to images/uv.lock",
    )
    commands.add_parser(
        IMAGES_COMMAND,
        add_help=False,
        help="build the images of the containers from a clean clone, labeled with its tree",
    )
    step_names = [row.name for row in RUNNABLE_CORE_ROWS]
    step_parser = commands.add_parser(
        STEP_COMMAND, add_help=False, help="run a step in its confined container"
    )
    step_parser.add_argument(
        "step", choices=step_names, metavar="<step>", help=", ".join(step_names)
    )
    clean_parser = commands.add_parser(
        CLEAN_STEP_NAME,
        add_help=False,
        help="remove the outputs, the request and the log of a step, or of every step",
    )
    clean_parser.add_argument(
        "step", nargs="?", choices=step_names, metavar="<step>", help=", ".join(step_names)
    )
    return parser


def find_account_home() -> str:
    try:
        return pwd.getpwuid(os.getuid()).pw_dir
    except KeyError:
        return NO_ACCOUNT_HOME


def has_account(user_id: int) -> bool:
    try:
        pwd.getpwuid(user_id)
    except KeyError:
        return False
    return True


def read_optional_text(path: Path) -> str:
    try:
        return path.read_text(encoding=TEXT_ENCODING)
    except FileNotFoundError:
        return ""


def read_host_identity() -> HostIdentity:
    return HostIdentity(
        user_id=os.getuid(),
        group_id=os.getgid(),
        cpu_count=len(os.sched_getaffinity(0)),
        accounts=HostAccounts(
            has_container_user_account=has_account(CONTAINER_USER_ID),
            subordinate_user_ranges=parse_subordinate_ranges(
                read_optional_text(SUBORDINATE_USER_IDS_PATH), str(SUBORDINATE_USER_IDS_PATH)
            ),
            subordinate_group_ranges=parse_subordinate_ranges(
                read_optional_text(SUBORDINATE_GROUP_IDS_PATH), str(SUBORDINATE_GROUP_IDS_PATH)
            ),
        ),
    )


def open_requested_work_directory(path: str, files: WorkFiles) -> WorkDirectory:
    request = WorkDirectoryRequest(
        path=path,
        home_variable=os.environ.get("HOME"),
        account_home=find_account_home(),
        clone_root=str(CLONE_ROOT),
        user_id=os.getuid(),
    )
    return open_work_directory(request, files)


def read_core_pins() -> tuple[SourcePin, ...]:
    return parse_source_pins((CLONE_ROOT / SOURCES_FILE_NAME).read_text(encoding=TEXT_ENCODING))


def describe_peaks(peaks: StepPeaks) -> str:
    return f"peaks: {peaks.memory_peak_mib:.0f} MiB of memory, {peaks.process_count_peak} processes"


def describe_lock_resolution(resolution: LockResolution) -> str:
    return (
        f"wrote images/{LOCK_FILE_NAME}, {len(resolution.checked_lock.packages)} packages, "
        f"{len(resolution.recorded_extras)} extras, from {resolution.lock_run}; "
        "review its change in the diff"
    )


def fetch(work_directory: WorkDirectory, files: WorkFiles) -> None:
    downloader = CurlDownloader()
    for pin in read_core_pins():
        source_path = locate_source(pin)
        print(f"checking {source_path}, {pin.size_bytes} bytes", flush=True)
        outcome = fetch_source(pin, work_directory, downloader, files)
        print(f"{OUTCOME_WORDS[outcome]} {source_path}", flush=True)


def build_container_images(work_directory: WorkDirectory, files: WorkFiles) -> None:
    clone_root = PurePosixPath(CLONE_ROOT)
    built_images = build_images(
        CORE_IMAGES,
        PurePosixPath(work_directory.path),
        GitTree(clone_root),
        DockerEngine(clone_root),
        files,
    )
    for image in built_images:
        print(f"built {image.tag}, {image.image_id}", flush=True)


def check_fleurs(work_directory: WorkDirectory, files: WorkFiles) -> None:
    clone_root = PurePosixPath(CLONE_ROOT)
    git_tree = GitTree(clone_root)
    outcome = check_fleurs_bench(
        FleursCheckRun(
            pins=read_core_pins(),
            commit=git_tree.find_head_commit(),
            work_root=PurePosixPath(work_directory.path),
            head_tree=git_tree.find_head_tree(),
            host=read_host_identity(),
        ),
        DockerEngine(clone_root),
        files,
    )
    for language in outcome.report.languages:
        print(
            f"{language.language}: {len(language.recordings)} recordings, "
            f"{language.duration_s:.0f} s",
            flush=True,
        )
    print(describe_peaks(outcome.peaks), flush=True)


def clean(step_name: str | None, work_directory: WorkDirectory, files: WorkFiles) -> None:
    clone_root = PurePosixPath(CLONE_ROOT)
    git_tree = GitTree(clone_root)
    rows = RUNNABLE_CORE_ROWS if step_name is None else (find_runnable_core_row(step_name),)
    for row in rows:
        outcome = clean_step(
            CleanRequest(
                row=row,
                commit=git_tree.find_head_commit(),
                work_root=PurePosixPath(work_directory.path),
                head_tree=git_tree.find_head_tree(),
                host=read_host_identity(),
            ),
            DockerEngine(clone_root),
            files,
        )
        print(f"{CLEAN_OUTCOME_WORDS[outcome]} {row.name}", flush=True)


def lock(work_directory: WorkDirectory, files: WorkFiles) -> None:
    clone_root = PurePosixPath(CLONE_ROOT)
    images_lock_path = IMAGES_ENVIRONMENT_DIRECTORY / LOCK_FILE_NAME
    resolution = resolve_lock(
        LockRequest(
            manifest_text=(IMAGES_ENVIRONMENT_DIRECTORY / MANIFEST_FILE_NAME).read_text(
                encoding=TEXT_ENCODING
            ),
            committed_lock_text=read_optional_text(images_lock_path) or None,
            images_lock_path=PurePosixPath(images_lock_path),
            work_root=PurePosixPath(work_directory.path),
            head_tree=GitTree(clone_root).find_head_tree(),
            host=read_host_identity(),
        ),
        DockerEngine(clone_root),
        files,
        lambda line: print(line, flush=True),
    )
    print(describe_lock_resolution(resolution), flush=True)
    print(describe_peaks(resolution.peaks), flush=True)


STEP_RUNNERS = {FLEURS_CHECK_STEP_NAME: check_fleurs}


def run_command(arguments: argparse.Namespace) -> None:
    files = WorkFiles()
    work_directory = open_requested_work_directory(arguments.work, files)
    if arguments.command == FETCH_COMMAND:
        fetch(work_directory, files)
    elif arguments.command == IMAGES_COMMAND:
        build_container_images(work_directory, files)
    elif arguments.command == STEP_COMMAND:
        STEP_RUNNERS[arguments.step](work_directory, files)
    elif arguments.command == CLEAN_STEP_NAME:
        clean(arguments.step, work_directory, files)
    elif arguments.command == LOCK_STEP_NAME:
        lock(work_directory, files)


def main(arguments: Sequence[str]) -> None:
    parsed_arguments = build_argument_parser().parse_args(arguments)
    try:
        run_command(parsed_arguments)
    except COMMAND_ERRORS as error:
        print(f"model-build: {error}", file=sys.stderr)
        raise SystemExit(FAILURE_EXIT_STATUS) from error


if __name__ == "__main__":
    main(sys.argv[1:])
