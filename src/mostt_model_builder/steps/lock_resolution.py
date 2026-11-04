import shutil
import subprocess
from pathlib import Path

from mostt_model_builder.domain.containers.container_steps import (
    OUTPUT_IN_THE_CONTAINER,
    REQUEST_IN_THE_CONTAINER,
    TEMPORARY_DIRECTORY_IN_THE_CONTAINER,
)
from mostt_model_builder.domain.lock.lock_check import LOCK_FILE_NAME, MANIFEST_FILE_NAME
from mostt_model_builder.steps.output_modes import finish_outputs

UV_PROGRAM = "/usr/local/bin/uv"
PROJECT_IN_THE_CONTAINER = TEMPORARY_DIRECTORY_IN_THE_CONTAINER / "project"
UV_CACHE_IN_THE_CONTAINER = TEMPORARY_DIRECTORY_IN_THE_CONTAINER / "uv-cache"
UV_PROJECT_FILE_NAME = "pyproject.toml"
UV_LOCK_FILE_NAME = "uv.lock"


def build_lock_arguments(uv_program: str, cache_directory: Path) -> list[str]:
    return [uv_program, "lock", "--cache-dir", str(cache_directory)]


def resolve_lock(
    request_directory: Path,
    project_directory: Path,
    output_directory: Path,
    lock_arguments: list[str],
) -> None:
    project_directory.mkdir()
    shutil.copyfile(
        request_directory / MANIFEST_FILE_NAME, project_directory / UV_PROJECT_FILE_NAME
    )
    committed_lock = request_directory / LOCK_FILE_NAME
    if committed_lock.exists():
        shutil.copyfile(committed_lock, project_directory / UV_LOCK_FILE_NAME)
    subprocess.run(lock_arguments, check=True, cwd=project_directory, stdin=subprocess.DEVNULL)
    shutil.copyfile(project_directory / UV_LOCK_FILE_NAME, output_directory / LOCK_FILE_NAME)


def main() -> None:
    output_directory = Path(OUTPUT_IN_THE_CONTAINER)
    resolve_lock(
        Path(REQUEST_IN_THE_CONTAINER),
        Path(PROJECT_IN_THE_CONTAINER),
        output_directory,
        build_lock_arguments(UV_PROGRAM, Path(UV_CACHE_IN_THE_CONTAINER)),
    )
    finish_outputs(output_directory)


if __name__ == "__main__":
    main()
