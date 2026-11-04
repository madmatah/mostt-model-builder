import subprocess
import sys
from pathlib import Path, PurePosixPath

import pytest

from mostt_model_builder.application.resolve_lock import LockResolution
from mostt_model_builder.domain.containers.step_peaks import StepPeaks
from mostt_model_builder.domain.lock.lock_check import CheckedLock, LockedPackage
from mostt_model_builder.host_main import describe_lock_resolution

BOOTSTRAP_PATH = (
    Path(__file__).resolve().parent.parent / "src" / "mostt_model_builder" / "bootstrap.py"
)
HOST_ENTRY_MODULE = "mostt_model_builder.host_main"
USAGE_START = "usage: model-build.sh"
UNKNOWN_COMMAND = "unknown-command"
COMMANDS = ("fetch", "lock", "images", "step", "verify", "build", "card", "clean")
RECIPE_NAME = "parakeet-tdt"
RECIPE_STEPS = (
    "fleurs-check",
    "describe",
    "export",
    "convert",
    "inspect",
    "reference",
    "transcribe",
)
ARGUMENT_ERROR_EXIT_STATUS = 2
COMMAND_FAILURE_EXIT_STATUS = 1
COMMAND_LINE_TIMEOUT_S = 60
LOCKED_PACKAGES = (LockedPackage("numpy", "2.3.4"), LockedPackage("torch", "2.9.0"))
RECORDED_EXTRAS = ("export", "inspect", "transcribe")
LOCK_RUN = PurePosixPath("runs/lock/run-1")
LOCK_RESOLUTION = LockResolution(
    lock_run=LOCK_RUN,
    checked_lock=CheckedLock(root_name="mostt-model-builder-images", packages=LOCKED_PACKAGES),
    recorded_extras=RECORDED_EXTRAS,
    change_lines=(),
    peaks=StepPeaks(memory_peak_bytes=0, process_count_peak=0),
)


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

            @pytest.mark.parametrize("command", COMMANDS)
            class TestWhenItListsTheCommands:
                def test_should_name_each_one(
                    self, help_run: subprocess.CompletedProcess[str], command: str
                ) -> None:
                    assert f"    {command} " in help_run.stdout

            def test_should_name_the_recipes_that_the_option_of_the_recipe_takes(
                self, help_run: subprocess.CompletedProcess[str]
            ) -> None:
                assert "--recipe <name>" in help_run.stdout
                assert RECIPE_NAME in help_run.stdout

        class TestWhenGivenAnUnknownStep:
            @pytest.fixture(scope="class")
            @classmethod
            def step_run(cls) -> subprocess.CompletedProcess[str]:
                return run_command_line(
                    ["--work", "/data/work", "--recipe", RECIPE_NAME, "step", UNKNOWN_COMMAND]
                )

            @pytest.mark.parametrize("step", RECIPE_STEPS)
            class TestWhenItListsTheSteps:
                def test_should_name_the_steps_of_the_core_and_of_the_recipe(
                    self, step_run: subprocess.CompletedProcess[str], step: str
                ) -> None:
                    assert step in step_run.stderr

            def test_should_fail_as_an_argument_error(
                self, step_run: subprocess.CompletedProcess[str]
            ) -> None:
                assert step_run.returncode == ARGUMENT_ERROR_EXIT_STATUS

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

        class TestWhenFetchIsGivenAWorkDirectoryThatDoesNotExist:
            @pytest.fixture(scope="class")
            @classmethod
            def missing_work_directory(cls, tmp_path_factory: pytest.TempPathFactory) -> Path:
                return tmp_path_factory.mktemp("parent").resolve() / "missing-work"

            @pytest.fixture(scope="class")
            @classmethod
            def fetch_run(cls, missing_work_directory: Path) -> subprocess.CompletedProcess[str]:
                return run_command_line(
                    ["--work", str(missing_work_directory), "--recipe", RECIPE_NAME, "fetch"]
                )

            def test_should_say_why_on_the_error_stream(
                self,
                fetch_run: subprocess.CompletedProcess[str],
                missing_work_directory: Path,
            ) -> None:
                assert fetch_run.stderr == (
                    f"model-build: the work directory {missing_work_directory} does not exist\n"
                )

            def test_should_fail(self, fetch_run: subprocess.CompletedProcess[str]) -> None:
                assert fetch_run.returncode == COMMAND_FAILURE_EXIT_STATUS

        class TestWhenACommandOtherThanTheLockIsGivenNoRecipe:
            @pytest.fixture(scope="class")
            @classmethod
            def fetch_run(cls) -> subprocess.CompletedProcess[str]:
                return run_command_line(["--work", "/data/work", "fetch"])

            def test_should_say_that_the_command_needs_a_recipe(
                self, fetch_run: subprocess.CompletedProcess[str]
            ) -> None:
                assert "the command fetch needs --recipe <name>" in fetch_run.stderr

            def test_should_fail_as_an_argument_error(
                self, fetch_run: subprocess.CompletedProcess[str]
            ) -> None:
                assert fetch_run.returncode == ARGUMENT_ERROR_EXIT_STATUS

    class TestTheDescribeLockResolutionFunction:
        @pytest.fixture
        def lock_description(self) -> str:
            return describe_lock_resolution(LOCK_RESOLUTION)

        def test_should_count_the_packages_and_the_extras_the_lock_records_and_name_its_run(
            self, lock_description: str
        ) -> None:
            assert lock_description == (
                f"wrote images/environment.lock, {len(LOCKED_PACKAGES)} packages, "
                f"{len(RECORDED_EXTRAS)} extras, from {LOCK_RUN}; review its change in the diff"
            )
