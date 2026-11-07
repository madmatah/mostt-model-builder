import subprocess
import sys
from pathlib import Path, PurePosixPath

import pytest

from mostt_model_builder.application.clean_run import BuildFile, BuildFileCleaned, CleanOutcome
from mostt_model_builder.application.prune_cache import PrunedEntry
from mostt_model_builder.application.resolve_lock import LockResolution
from mostt_model_builder.domain.cache.hardware_facts import NO_GPU_REASON
from mostt_model_builder.domain.containers.step_peaks import StepPeaks
from mostt_model_builder.domain.contract.model_description import ModelDescription
from mostt_model_builder.domain.contract.recipe_contract import (
    RECIPES_DIRECTORY_NAME,
    RecipeContract,
)
from mostt_model_builder.domain.lock.lock_check import CheckedLock, LockedPackage
from mostt_model_builder.domain.publication.publication_report import (
    PublicationReport,
    ReportedFile,
)
from mostt_model_builder.host_main import (
    describe_clean_event,
    describe_cleaned_entry,
    describe_lock_resolution,
    describe_prune_total,
    describe_pruned_entry,
    describe_publication,
    describe_removed_unfinished_entry,
    describe_reported_file,
    describe_reused_entry,
    describe_skipped_entry,
    is_card_template_present,
)

BOOTSTRAP_PATH = (
    Path(__file__).resolve().parent.parent / "src" / "mostt_model_builder" / "bootstrap.py"
)
HOST_ENTRY_MODULE = "mostt_model_builder.host_main"
USAGE_START = "usage: model-build.sh"
UNKNOWN_COMMAND = "unknown-command"
COMMANDS = (
    "fetch",
    "lock",
    "images",
    "step",
    "verify",
    "build",
    "card",
    "publish",
    "check-published",
    "clean",
    "prune",
)
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
TEMPLATE_RECIPE_NAME = "family"
# The keys of the mock of the commands' lines, each completed to its 64 hexadecimal digits.
FLEURS_CHECK_KEY = "7b1e04c9a2f3" + "0" * 52
CANDIDATE_KEY = "91aa3c0e7d45" + "1" * 52
UNFINISHED_KEY = "6a0f3e81c9d4" + "2" * 52
CLEANED_KEY = "9e41c2d07a3b" + "3" * 52
HUB_REPOSITORY = "<user>/parakeet-ultra-onnx"
PARENT_REVISION = "9a0b1c2d3e4f5a6b7c8d9e0f1a2b3c4d5e6f7a8b"
NEW_REVISION = "5e1f0c2a9b8d7e6f5a4b3c2d1e0f9a8b7c6d5e4f"
KEPT_ENCODER = ReportedFile("encoder-model.onnx", 902417, "a" * 64, is_uploaded=False)
UPLOADED_CARD = ReportedFile("README.md", 9412, "b" * 64, is_uploaded=True)
KEPT_CARD = ReportedFile("README.md", 9412, "b" * 64, is_uploaded=False)
PRUNED_ENTRIES = (
    PrunedEntry("convert-bfloat16", "5c0d8a3b9e21" + "4" * 52, 1241503812),
    PrunedEntry("candidate", "4e7a2b90c1d3" + "5" * 52, 5934410225),
    PrunedEntry("inspect", "8f02d1b6e4a7" + "6" * 52, 61448),
    PrunedEntry("transcribe-bfloat16-gpu-fr_fr-0", "b3c90e5a7d12" + "7" * 52, 398220),
    PrunedEntry("transcribe-bfloat16-gpu-en_us-0", "1d7e4f0a9c63" + "8" * 52, 371904),
)
CARD_TEMPLATE_NAME = "card-template.md"
CARD_TEMPLATE_TEXT = "# {title}\n"
CONTRACT_WITHOUT_MODEL = RecipeContract(
    name=TEMPLATE_RECIPE_NAME, images=(), steps=(), candidate=()
)
CONTRACT_WITH_MODEL = RecipeContract(
    name=TEMPLATE_RECIPE_NAME,
    images=(),
    steps=(),
    candidate=(),
    model=ModelDescription(
        title="Family",
        template_path=PurePosixPath(CARD_TEMPLATE_NAME),
        measuring_library_names=(),
        variants=(),
    ),
)


def make_recipe_directory(clone_root: Path) -> Path:
    recipe_directory = clone_root / RECIPES_DIRECTORY_NAME / TEMPLATE_RECIPE_NAME
    recipe_directory.mkdir(parents=True)
    return recipe_directory


def run_command_line(arguments: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-I", "-S", "-B", str(BOOTSTRAP_PATH), HOST_ENTRY_MODULE, *arguments],
        capture_output=True,
        check=False,
        stdin=subprocess.DEVNULL,
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

        class TestWhenThePublicationIsGivenNoTerminal:
            @pytest.fixture(scope="class")
            @classmethod
            def publish_run(
                cls, tmp_path_factory: pytest.TempPathFactory
            ) -> subprocess.CompletedProcess[str]:
                missing_work_directory = tmp_path_factory.mktemp("parent") / "missing-work"
                return run_command_line(
                    [
                        "--work",
                        str(missing_work_directory),
                        "--recipe",
                        RECIPE_NAME,
                        "publish",
                        "--repository",
                        "example-user/parakeet-ultra-onnx",
                    ]
                )

            def test_should_refuse_it_before_it_opens_the_work_directory(
                self, publish_run: subprocess.CompletedProcess[str]
            ) -> None:
                assert (
                    "publish reads the token at a terminal" in publish_run.stderr,
                    "work directory" in publish_run.stderr,
                ) == (True, False)

            def test_should_fail_as_an_argument_error(
                self, publish_run: subprocess.CompletedProcess[str]
            ) -> None:
                assert publish_run.returncode == ARGUMENT_ERROR_EXIT_STATUS

        class TestWhenThePublicationIsGivenNoRepository:
            @pytest.fixture(scope="class")
            @classmethod
            def publish_run(cls) -> subprocess.CompletedProcess[str]:
                return run_command_line(
                    ["--work", "/data/work", "--recipe", RECIPE_NAME, "publish"]
                )

            def test_should_say_that_it_needs_the_repository(
                self, publish_run: subprocess.CompletedProcess[str]
            ) -> None:
                assert "--repository" in publish_run.stderr

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

    class TestTheIsCardTemplatePresentFunction:
        class TestWhenTheTemplateIsARegularFile:
            @pytest.fixture
            def is_present(self, tmp_path: Path) -> bool:
                recipe_directory = make_recipe_directory(tmp_path)
                (recipe_directory / CARD_TEMPLATE_NAME).write_text(CARD_TEMPLATE_TEXT)
                return is_card_template_present(CONTRACT_WITH_MODEL, tmp_path)

            def test_should_find_the_template(self, is_present: bool) -> None:
                assert is_present

        class TestWhenTheTemplateIsALinkToARegularFile:
            @pytest.fixture
            def is_present(self, tmp_path: Path) -> bool:
                recipe_directory = make_recipe_directory(tmp_path)
                linked_file = tmp_path / CARD_TEMPLATE_NAME
                linked_file.write_text(CARD_TEMPLATE_TEXT)
                (recipe_directory / CARD_TEMPLATE_NAME).symlink_to(linked_file)
                return is_card_template_present(CONTRACT_WITH_MODEL, tmp_path)

            def test_should_refuse_the_link_so_that_the_host_never_reads_through_it(
                self, is_present: bool
            ) -> None:
                assert not is_present

        class TestWhenTheTemplateIsADirectory:
            @pytest.fixture
            def is_present(self, tmp_path: Path) -> bool:
                recipe_directory = make_recipe_directory(tmp_path)
                (recipe_directory / CARD_TEMPLATE_NAME).mkdir()
                return is_card_template_present(CONTRACT_WITH_MODEL, tmp_path)

            def test_should_refuse_the_directory(self, is_present: bool) -> None:
                assert not is_present

        class TestWhenTheTemplateIsMissing:
            @pytest.fixture
            def is_present(self, tmp_path: Path) -> bool:
                make_recipe_directory(tmp_path)
                return is_card_template_present(CONTRACT_WITH_MODEL, tmp_path)

            def test_should_not_find_the_template(self, is_present: bool) -> None:
                assert not is_present

        class TestWhenTheContractDescribesNoModel:
            @pytest.fixture
            def is_present(self, tmp_path: Path) -> bool:
                recipe_directory = make_recipe_directory(tmp_path)
                (recipe_directory / CARD_TEMPLATE_NAME).write_text(CARD_TEMPLATE_TEXT)
                return is_card_template_present(CONTRACT_WITHOUT_MODEL, tmp_path)

            def test_should_find_no_template_since_no_card_is_built(self, is_present: bool) -> None:
                assert not is_present

    class TestTheDescribeReusedEntryFunction:
        class TestWhenARowIsReused:
            @pytest.fixture
            def line(self) -> str:
                return describe_reused_entry("fleurs-check", FLEURS_CHECK_KEY)

            def test_should_name_the_row_and_the_start_of_its_key(self, line: str) -> None:
                assert line == "reused fleurs-check, key 7b1e04c9a2f3"

        class TestWhenTheCandidateIsReused:
            @pytest.fixture
            def line(self) -> str:
                return describe_reused_entry("candidate", CANDIDATE_KEY)

            def test_should_call_it_the_candidate(self, line: str) -> None:
                assert line == "reused the candidate, key 91aa3c0e7d45"

    class TestTheDescribeRemovedUnfinishedEntryFunction:
        @pytest.fixture
        def line(self) -> str:
            return describe_removed_unfinished_entry(
                "transcribe-float32-cpu-fr_fr-2", UNFINISHED_KEY
            )

        def test_should_name_the_row_and_the_start_of_its_key(self, line: str) -> None:
            assert line == (
                "removed the unfinished entry of transcribe-float32-cpu-fr_fr-2, key 6a0f3e81c9d4"
            )

    class TestTheDescribeCleanedEntryFunction:
        @pytest.fixture
        def line(self) -> str:
            return describe_cleaned_entry(
                "transcribe-float32-cpu-fr_fr-0", CLEANED_KEY, CleanOutcome.CLEANED
            )

        def test_should_name_the_row_and_the_start_of_its_key(self, line: str) -> None:
            assert line == "cleaned transcribe-float32-cpu-fr_fr-0, key 9e41c2d07a3b"

    class TestTheDescribeSkippedEntryFunction:
        @pytest.fixture
        def line(self) -> str:
            return describe_skipped_entry("transcribe-float16-gpu-fr_fr-0", NO_GPU_REASON)

        def test_should_name_the_row_and_the_reason_it_has_no_key(self, line: str) -> None:
            assert line == (
                "skipped transcribe-float16-gpu-fr_fr-0: /proc/driver/nvidia holds no GPU, which "
                "the key of a run on the GPU needs"
            )

    class TestTheDescribeCleanEventFunction:
        @pytest.mark.parametrize(
            ("event", "expected_line"),
            [
                pytest.param(
                    BuildFileCleaned(BuildFile.VERIFICATION_REPORT, CleanOutcome.CLEANED),
                    "cleaned the verification's report",
                    id="the-report",
                ),
                pytest.param(
                    BuildFileCleaned(BuildFile.MODEL_CARD, CleanOutcome.CLEANED),
                    "cleaned the model card",
                    id="the-card",
                ),
            ],
        )
        class TestWhenAFileOfTheBuildIsCleaned:
            @pytest.fixture
            def line(self, event: BuildFileCleaned) -> str:
                return describe_clean_event(event)

            def test_should_name_it(self, line: str, expected_line: str) -> None:
                assert line == expected_line

    class TestTheDescribePrunedEntryFunction:
        @pytest.fixture
        def lines(self) -> list[str]:
            return [describe_pruned_entry(entry) for entry in PRUNED_ENTRIES]

        def test_should_name_each_entry_its_key_and_its_bytes(self, lines: list[str]) -> None:
            assert lines == [
                "removed convert-bfloat16, key 5c0d8a3b9e21, 1241503812 bytes",
                "removed the candidate, key 4e7a2b90c1d3, 5934410225 bytes",
                "removed inspect, key 8f02d1b6e4a7, 61448 bytes",
                "removed transcribe-bfloat16-gpu-fr_fr-0, key b3c90e5a7d12, 398220 bytes",
                "removed transcribe-bfloat16-gpu-en_us-0, key 1d7e4f0a9c63, 371904 bytes",
            ]

    class TestTheDescribePruneTotalFunction:
        @pytest.fixture
        def line(self) -> str:
            return describe_prune_total(PRUNED_ENTRIES)

        def test_should_count_the_entries_and_the_bytes_freed(self, line: str) -> None:
            assert line == "pruned 5 entries, 7176745609 bytes freed"

    class TestTheDescribeReportedFileFunction:
        @pytest.fixture
        def lines(self) -> list[str]:
            return [describe_reported_file(reported) for reported in (KEPT_ENCODER, UPLOADED_CARD)]

        def test_should_say_whether_each_file_was_kept_or_uploaded(self, lines: list[str]) -> None:
            assert lines == [
                "kept encoder-model.onnx, already on the Hub with its sha256",
                "uploaded README.md, 9412 bytes",
            ]

    class TestTheDescribePublicationFunction:
        class TestWhenAFileWasUploaded:
            @pytest.fixture
            def line(self) -> str:
                return describe_publication(
                    PublicationReport(
                        HUB_REPOSITORY,
                        True,
                        PARENT_REVISION,
                        NEW_REVISION,
                        (KEPT_ENCODER, UPLOADED_CARD),
                    )
                )

            def test_should_name_the_new_revision_and_its_parent(self, line: str) -> None:
                assert line == (
                    "published <user>/parakeet-ultra-onnx at the revision "
                    "5e1f0c2a9b8d7e6f5a4b3c2d1e0f9a8b7c6d5e4f, on "
                    "9a0b1c2d3e4f5a6b7c8d9e0f1a2b3c4d5e6f7a8b; the repository is private"
                )

        class TestWhenNothingDiffered:
            @pytest.fixture
            def line(self) -> str:
                return describe_publication(
                    PublicationReport(
                        HUB_REPOSITORY,
                        True,
                        NEW_REVISION,
                        NEW_REVISION,
                        (KEPT_ENCODER, KEPT_CARD),
                    )
                )

            def test_should_say_there_was_nothing_to_publish(self, line: str) -> None:
                assert line == (
                    "nothing to publish: <user>/parakeet-ultra-onnx already holds every file of "
                    "the build at the revision 5e1f0c2a9b8d7e6f5a4b3c2d1e0f9a8b7c6d5e4f; the "
                    "repository is private"
                )
