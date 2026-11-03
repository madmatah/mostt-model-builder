import subprocess
import sys
from pathlib import Path

import pytest

BOOTSTRAP_PATH = (
    Path(__file__).resolve().parent.parent / "src" / "mostt_model_builder" / "bootstrap.py"
)
HOST_ENTRY_MODULE = "mostt_model_builder.host_main"
USAGE_START = "usage: model-build.sh"
UNKNOWN_COMMAND = "unknown-command"
ARGUMENT_ERROR_EXIT_STATUS = 2
COMMAND_LINE_TIMEOUT_S = 60


def run_command_line(arguments: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-I", "-S", "-B", str(BOOTSTRAP_PATH), HOST_ENTRY_MODULE, *arguments],
        capture_output=True,
        check=False,
        text=True,
        timeout=COMMAND_LINE_TIMEOUT_S,
    )


class TestTheHostMain:
    class TestTheMainFunction:
        class TestWhenAskedForHelp:
            @pytest.fixture(scope="class")
            @classmethod
            def help_run(cls) -> subprocess.CompletedProcess[str]:
                return run_command_line(["--help"])

            def test_should_print_the_usage(
                self, help_run: subprocess.CompletedProcess[str]
            ) -> None:
                assert help_run.stdout.startswith(USAGE_START)

            def test_should_succeed(self, help_run: subprocess.CompletedProcess[str]) -> None:
                assert help_run.returncode == 0

        class TestWhenGivenAnUnknownCommand:
            @pytest.fixture(scope="class")
            @classmethod
            def unknown_command_run(cls) -> subprocess.CompletedProcess[str]:
                return run_command_line([UNKNOWN_COMMAND])

            def test_should_print_the_usage_on_the_error_stream(
                self, unknown_command_run: subprocess.CompletedProcess[str]
            ) -> None:
                assert unknown_command_run.stderr.startswith(USAGE_START)

            def test_should_name_the_unknown_command(
                self, unknown_command_run: subprocess.CompletedProcess[str]
            ) -> None:
                assert UNKNOWN_COMMAND in unknown_command_run.stderr

            def test_should_fail_as_an_argument_error(
                self, unknown_command_run: subprocess.CompletedProcess[str]
            ) -> None:
                assert unknown_command_run.returncode == ARGUMENT_ERROR_EXIT_STATUS
