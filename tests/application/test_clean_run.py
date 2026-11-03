import re
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from pathlib import PurePosixPath

import pytest
from run_step_fakes import (
    GROUP_ID,
    USER_ID,
    FakeContainerEngine,
    FakeStepFiles,
    build_directory_facts,
    build_file_facts,
    build_step_output,
)

from mostt_model_builder.application.clean_run import (
    CleanOutcome,
    CleanRequest,
    clean_step,
    remove_candidate,
    remove_verification_report,
)
from mostt_model_builder.application.run_step import HostIdentity, ImageFacts, StepError
from mostt_model_builder.domain.containers.account_ranges import HostAccounts
from mostt_model_builder.domain.containers.base_images import ENVIRONMENT_PYTHON
from mostt_model_builder.domain.containers.container_files import (
    ContainerFileError,
    FileFacts,
    FileKind,
)
from mostt_model_builder.domain.containers.container_steps import (
    FLEURS_CHECK_ROW,
    INSPECT_ROW,
    OUTPUT_IN_THE_CONTAINER,
    PEAKS_OUTPUT,
    BindMount,
    ContainerImage,
    ContainerRun,
    Device,
    StepLimits,
    StepRow,
)

WORK_ROOT = PurePosixPath("/data/builds/work")
COMMIT = "e949fa3a2eb8bf218e96c2c03625bbd36e4342b1"
HEAD_TREE = "3f1d1ffe79bc3336eb583331cfbf9b189fd45f01"
IMAGE_ID = "sha256:" + "d" * 64
RECIPE_NAME = "parakeet-tdt"
RUN_DIRECTORY = PurePosixPath(f"runs/{RECIPE_NAME}/{COMMIT}")
OUTPUT_DIRECTORY = WORK_ROOT / RUN_DIRECTORY / "fleurs-check"
REQUEST_DIRECTORY = WORK_ROOT / RUN_DIRECTORY / "requests/fleurs-check"
LOGS_DIRECTORY = WORK_ROOT / RUN_DIRECTORY / "logs"
LOG_PATH = LOGS_DIRECTORY / "fleurs-check.log"
CLEAN_LOG_PATH = WORK_ROOT / RUN_DIRECTORY / "logs/clean-fleurs-check.log"
LINK_FACTS = FileFacts(FileKind.LINK, USER_ID, 0o777, 10, 1)
CANDIDATE_DIRECTORY = WORK_ROOT / RUN_DIRECTORY / "candidate"
CANDIDATE_FILE_NAMES = ("encoder-model.onnx", "vocab.txt")
REPORT_PATH = WORK_ROOT / RUN_DIRECTORY / "verification.json"
HOST = HostIdentity(USER_ID, GROUP_ID, 20, HostAccounts(False, (), ()))
INSPECT_REQUEST = CleanRequest(INSPECT_ROW, RECIPE_NAME, COMMIT, WORK_ROOT, HEAD_TREE, HOST)
REQUEST = CleanRequest(FLEURS_CHECK_ROW, RECIPE_NAME, COMMIT, WORK_ROOT, HEAD_TREE, HOST)
RECIPE_ROW = StepRow(
    name="describe",
    image=ContainerImage(RECIPE_NAME, "export", Device.CPU, ENVIRONMENT_PYTHON),
    entry_module="mostt_recipe_parakeet_tdt.steps.nemo_description",
    sources=(),
    has_request=False,
    outputs=(PEAKS_OUTPUT,),
    limits=StepLimits(memory_mib=1024, process_count=64, cpu_count=2, temporary_directory_mib=64),
)


@dataclass(frozen=True)
class CleanRecord:
    outcome: CleanOutcome
    runs: list[ContainerRun]
    log_paths: list[PurePosixPath]
    files: FakeStepFiles


def build_files_with_a_run() -> FakeStepFiles:
    files = FakeStepFiles()
    files.add_directory(WORK_ROOT)
    files.add_directory(OUTPUT_DIRECTORY, build_directory_facts(0o2770))
    files.add_file(OUTPUT_DIRECTORY / "peaks.json", *build_step_output(b"{}"))
    files.add_directory(REQUEST_DIRECTORY, build_directory_facts(0o750))
    files.add_file(REQUEST_DIRECTORY / "request.json", build_file_facts(2), b"{}")
    files.add_file(LOG_PATH, build_file_facts(0, 0o600), b"")
    return files


@dataclass(frozen=True)
class RefusedClean:
    clean_act: Callable[[], CleanOutcome]
    engine: FakeContainerEngine
    files: FakeStepFiles


def build_files_with_a_link(link_path: PurePosixPath) -> FakeStepFiles:
    files = build_files_with_a_run()
    for path in [path for path in files.entries if link_path in path.parents]:
        del files.entries[path]
        files.contents.pop(path, None)
    files.entries[link_path] = LINK_FACTS
    return files


def build_refused_clean(link_path: PurePosixPath) -> RefusedClean:
    files = build_files_with_a_link(link_path)
    engine = build_engine(files)
    return RefusedClean(partial(clean_step, REQUEST, engine, files), engine, files)


@dataclass(frozen=True)
class CandidateRemoval:
    outcome: CleanOutcome
    files: FakeStepFiles


@dataclass(frozen=True)
class ReportRemoval:
    outcome: CleanOutcome
    files: FakeStepFiles


def build_files_with_a_candidate() -> FakeStepFiles:
    files = FakeStepFiles()
    files.add_directory(WORK_ROOT)
    files.add_directory(CANDIDATE_DIRECTORY, build_directory_facts(0o750))
    for name in CANDIDATE_FILE_NAMES:
        files.add_file(CANDIDATE_DIRECTORY / name, build_file_facts(4, 0o444), b"copy")
    return files


def build_engine(files: FakeStepFiles, status: int = 0) -> FakeContainerEngine:
    return FakeContainerEngine(
        files=files,
        images={FLEURS_CHECK_ROW.image.tag: ImageFacts(IMAGE_ID, HEAD_TREE)},
        status=status,
    )


class TestTheCleanRun:
    class TestTheCleanStepFunction:
        class TestWhenTheStepLeftOutputsARequestAndALog:
            @pytest.fixture
            def clean_record(self) -> CleanRecord:
                files = build_files_with_a_run()
                engine = build_engine(files)
                outcome = clean_step(REQUEST, engine, files)
                return CleanRecord(outcome, engine.runs, engine.log_paths, files)

            def test_should_empty_the_outputs_in_the_image_of_the_step_mounting_them_alone(
                self, clean_record: CleanRecord
            ) -> None:
                assert [
                    (run.image_id, run.entry_module, run.mounts, run.has_network)
                    for run in clean_record.runs
                ] == [
                    (
                        IMAGE_ID,
                        "mostt_model_builder.steps.clean_output",
                        (
                            BindMount(
                                RUN_DIRECTORY / "fleurs-check",
                                OUTPUT_IN_THE_CONTAINER,
                                is_read_only=False,
                                is_file=False,
                            ),
                        ),
                        False,
                    )
                ]

            def test_should_remove_the_outputs_the_request_and_the_log_of_the_step(
                self, clean_record: CleanRecord
            ) -> None:
                assert {OUTPUT_DIRECTORY, REQUEST_DIRECTORY, LOG_PATH} & set(
                    clean_record.files.entries
                ) == set()

            def test_should_log_the_cleaning_apart(self, clean_record: CleanRecord) -> None:
                assert clean_record.log_paths == [CLEAN_LOG_PATH]

            def test_should_say_it_cleaned(self, clean_record: CleanRecord) -> None:
                assert clean_record.outcome is CleanOutcome.CLEANED

        class TestWhenTheStepNeverRan:
            @pytest.fixture
            def clean_record(self) -> CleanRecord:
                files = FakeStepFiles()
                files.add_directory(WORK_ROOT)
                engine = build_engine(files)
                outcome = clean_step(REQUEST, engine, files)
                return CleanRecord(outcome, engine.runs, engine.log_paths, files)

            def test_should_run_no_container(self, clean_record: CleanRecord) -> None:
                assert clean_record.runs == []

            def test_should_say_there_was_nothing_to_clean(self, clean_record: CleanRecord) -> None:
                assert clean_record.outcome is CleanOutcome.NOTHING_TO_CLEAN

        class TestWhenTheCleaningContainerFails:
            @pytest.fixture
            def clean_act(self) -> Callable[[], CleanOutcome]:
                files = build_files_with_a_run()
                return partial(clean_step, REQUEST, build_engine(files, status=1), files)

            def test_should_stop_and_say_so(self, clean_act: Callable[[], CleanOutcome]) -> None:
                with pytest.raises(
                    StepError, match=re.escape("the step clean exited with the status 1")
                ):
                    clean_act()

        @pytest.mark.parametrize(
            "link_path", [OUTPUT_DIRECTORY, REQUEST_DIRECTORY], ids=["the outputs", "the request"]
        )
        class TestWhenALinkStandsWhereADirectoryOfTheStepGoes:
            @pytest.fixture
            def clean_act(self, link_path: PurePosixPath) -> Callable[[], CleanOutcome]:
                files = FakeStepFiles()
                files.add_directory(WORK_ROOT)
                files.add_directory(link_path.parent)
                files.entries[link_path] = LINK_FACTS
                return partial(clean_step, REQUEST, build_engine(files), files)

            def test_should_refuse_it_before_any_container_or_removal_follows_it(
                self, clean_act: Callable[[], CleanOutcome], link_path: PurePosixPath
            ) -> None:
                with pytest.raises(
                    ContainerFileError,
                    match=re.escape(f"{link_path} is a link, where a directory is declared"),
                ):
                    clean_act()

        class TestWhenTheRequestDirectoryIsALink:
            @pytest.fixture
            def refused_clean(self) -> RefusedClean:
                return build_refused_clean(REQUEST_DIRECTORY)

            def test_should_refuse_it_before_any_container_or_removal(
                self, refused_clean: RefusedClean
            ) -> None:
                with pytest.raises(
                    ContainerFileError,
                    match=re.escape(
                        f"{REQUEST_DIRECTORY} is a link, where a directory is declared"
                    ),
                ):
                    refused_clean.clean_act()
                assert (
                    refused_clean.engine.runs,
                    OUTPUT_DIRECTORY in refused_clean.files.entries,
                ) == (
                    [],
                    True,
                )

        class TestWhenTheLogsDirectoryIsALink:
            @pytest.fixture
            def refused_clean(self) -> RefusedClean:
                return build_refused_clean(LOGS_DIRECTORY)

            def test_should_refuse_it_before_any_container_or_removal(
                self, refused_clean: RefusedClean
            ) -> None:
                with pytest.raises(
                    ContainerFileError,
                    match=re.escape(f"{LOGS_DIRECTORY} is a link, where a directory is declared"),
                ):
                    refused_clean.clean_act()
                assert (
                    refused_clean.engine.runs,
                    OUTPUT_DIRECTORY in refused_clean.files.entries,
                ) == (
                    [],
                    True,
                )

        class TestWhenTheStepIsTheRecipesOwn:
            @pytest.fixture
            def clean_record(self) -> CleanRecord:
                files = FakeStepFiles()
                files.add_directory(WORK_ROOT)
                describe_directory = WORK_ROOT / RUN_DIRECTORY / "describe"
                files.add_directory(describe_directory, build_directory_facts(0o2770))
                files.add_file(describe_directory / "peaks.json", *build_step_output(b"{}"))
                engine = FakeContainerEngine(
                    files=files, images={RECIPE_ROW.image.tag: ImageFacts(IMAGE_ID, HEAD_TREE)}
                )
                outcome = clean_step(
                    CleanRequest(RECIPE_ROW, RECIPE_NAME, COMMIT, WORK_ROOT, HEAD_TREE, HOST),
                    engine,
                    files,
                )
                return CleanRecord(outcome, engine.runs, engine.log_paths, files)

            def test_should_run_the_cores_cleaning_in_the_recipes_image(
                self, clean_record: CleanRecord
            ) -> None:
                assert [(run.image_id, run.entry_module) for run in clean_record.runs] == [
                    (IMAGE_ID, "mostt_model_builder.steps.clean_output")
                ]

    class TestTheRemoveCandidateFunction:
        class TestWhenTheRunHoldsACandidate:
            @pytest.fixture
            def removal(self) -> CandidateRemoval:
                files = build_files_with_a_candidate()
                return CandidateRemoval(remove_candidate(INSPECT_REQUEST, files), files)

            def test_should_remove_its_files_and_its_directory(
                self, removal: CandidateRemoval
            ) -> None:
                assert [
                    path
                    for path in removal.files.entries
                    if path == CANDIDATE_DIRECTORY or CANDIDATE_DIRECTORY in path.parents
                ] == []

            def test_should_say_it_cleaned(self, removal: CandidateRemoval) -> None:
                assert removal.outcome is CleanOutcome.CLEANED

        class TestWhenTheRunHoldsNoCandidate:
            @pytest.fixture
            def outcome(self) -> CleanOutcome:
                files = FakeStepFiles()
                files.add_directory(WORK_ROOT)
                return remove_candidate(INSPECT_REQUEST, files)

            def test_should_say_there_was_nothing_to_clean(self, outcome: CleanOutcome) -> None:
                assert outcome is CleanOutcome.NOTHING_TO_CLEAN

        class TestWhenTheCandidateIsALink:
            @pytest.fixture
            def remove_act(self) -> Callable[[], CleanOutcome]:
                files = FakeStepFiles()
                files.add_directory(WORK_ROOT / RUN_DIRECTORY)
                files.entries[CANDIDATE_DIRECTORY] = FileFacts(FileKind.LINK, USER_ID, 0o777, 9, 1)
                return partial(remove_candidate, INSPECT_REQUEST, files)

            def test_should_refuse_to_follow_it(
                self, remove_act: Callable[[], CleanOutcome]
            ) -> None:
                with pytest.raises(
                    ContainerFileError,
                    match=re.escape(
                        f"{CANDIDATE_DIRECTORY} is a link, where a directory is declared"
                    ),
                ):
                    remove_act()

    class TestTheRemoveVerificationReportFunction:
        class TestWhenTheRunHoldsAReport:
            @pytest.fixture
            def removal(self) -> ReportRemoval:
                files = build_files_with_a_run()
                files.add_file(REPORT_PATH, build_file_facts(2, 0o644), b"{}")
                return ReportRemoval(remove_verification_report(REQUEST, files), files)

            def test_should_remove_it(self, removal: ReportRemoval) -> None:
                assert REPORT_PATH not in removal.files.entries

            def test_should_say_it_cleaned(self, removal: ReportRemoval) -> None:
                assert removal.outcome is CleanOutcome.CLEANED

        class TestWhenTheRunHoldsNoReport:
            @pytest.fixture
            def outcome(self) -> CleanOutcome:
                return remove_verification_report(REQUEST, build_files_with_a_run())

            def test_should_say_there_was_nothing_to_clean(self, outcome: CleanOutcome) -> None:
                assert outcome is CleanOutcome.NOTHING_TO_CLEAN

        class TestWhenTheRunNeverRan:
            @pytest.fixture
            def outcome(self) -> CleanOutcome:
                files = FakeStepFiles()
                files.add_directory(WORK_ROOT)
                return remove_verification_report(REQUEST, files)

            def test_should_say_there_was_nothing_to_clean(self, outcome: CleanOutcome) -> None:
                assert outcome is CleanOutcome.NOTHING_TO_CLEAN

        class TestWhenTheRunIsALink:
            @pytest.fixture
            def remove_act(self) -> Callable[[], CleanOutcome]:
                files = FakeStepFiles()
                files.add_directory(WORK_ROOT / RUN_DIRECTORY.parent)
                files.entries[WORK_ROOT / RUN_DIRECTORY] = FileFacts(
                    FileKind.LINK, USER_ID, 0o777, 9, 1
                )
                return partial(remove_verification_report, REQUEST, files)

            def test_should_refuse_to_follow_it(
                self, remove_act: Callable[[], CleanOutcome]
            ) -> None:
                with pytest.raises(
                    ContainerFileError,
                    match=re.escape(
                        f"{WORK_ROOT / RUN_DIRECTORY} is a link, where a directory is declared"
                    ),
                ):
                    remove_act()
