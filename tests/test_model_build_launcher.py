import os
import shlex
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
LAUNCHER_PATH = REPO_ROOT / "model-build.sh"
BOOTSTRAP_PATH = REPO_ROOT / "src" / "mostt_model_builder" / "bootstrap.py"
HOST_ENTRY_MODULE = "mostt_model_builder.host_main"
ISOLATION_OPTIONS = ["-I", "-S", "-B"]
USER_ARGUMENTS = ["--work", "a directory with a space", "-a-leading-dash"]
LAUNCHER_TIMEOUT_S = 60


def write_recording_python(fake_directory: Path, record_path: Path) -> None:
    fake_python = fake_directory / "python3"
    fake_python.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        f"printf '%s\\0' \"$@\" > {shlex.quote(str(record_path))}\n"
    )
    fake_python.chmod(0o755)


def read_recorded_arguments(record_path: Path) -> list[str]:
    return record_path.read_text().split("\0")[:-1]


class TestTheModelBuildLauncher:
    class TestWhenItRunsFromAnotherDirectory:
        @pytest.fixture(scope="class")
        @classmethod
        def python_arguments(cls, tmp_path_factory: pytest.TempPathFactory) -> list[str]:
            fake_directory = tmp_path_factory.mktemp("fake-bin")
            working_directory = tmp_path_factory.mktemp("elsewhere")
            record_path = tmp_path_factory.mktemp("record") / "python-arguments"
            write_recording_python(fake_directory, record_path)
            subprocess.run(
                [str(LAUNCHER_PATH), *USER_ARGUMENTS],
                check=True,
                cwd=working_directory,
                env={"PATH": f"{fake_directory}{os.pathsep}{os.environ['PATH']}"},
                timeout=LAUNCHER_TIMEOUT_S,
            )
            return read_recorded_arguments(record_path)

        def test_should_isolate_python_from_the_environment_and_the_site_packages(
            self, python_arguments: list[str]
        ) -> None:
            assert python_arguments[:3] == ISOLATION_OPTIONS

        def test_should_run_its_own_bootstrap_on_the_command_line_module(
            self, python_arguments: list[str]
        ) -> None:
            assert python_arguments[3:5] == [str(BOOTSTRAP_PATH), HOST_ENTRY_MODULE]

        def test_should_pass_the_user_arguments_unchanged(
            self, python_arguments: list[str]
        ) -> None:
            assert python_arguments[5:] == USER_ARGUMENTS
