import re
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from pathlib import PurePosixPath

import pytest
from cache_arrangement import (
    GPU_HARDWARE,
    HARDWARE_WITHOUT_GPU,
    CacheBuild,
    clean_cache,
    list_entries,
    list_ran_rows,
    rewrite_output,
)
from cache_arrangement import RECIPE_NAME as CACHE_RECIPE_NAME
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
    BuildFile,
    BuildFileCleaned,
    CleanEvent,
    CleanOutcome,
    EntryCleaned,
    EntryCleaning,
    EntrySkipped,
    clean_entry,
    remove_candidate_entry,
    remove_model_card,
    remove_verification_report,
)
from mostt_model_builder.application.run_step import HostIdentity, ImageFacts, StepError
from mostt_model_builder.domain.cache.hardware_facts import NO_GPU_REASON, HardwareFacts
from mostt_model_builder.domain.containers.account_ranges import HostAccounts
from mostt_model_builder.domain.containers.base_images import ENVIRONMENT_PYTHON
from mostt_model_builder.domain.containers.container_files import (
    ContainerFileError,
    FileFacts,
    FileKind,
)
from mostt_model_builder.domain.containers.container_paths import OUTPUT_IN_THE_CONTAINER
from mostt_model_builder.domain.containers.container_steps import (
    FLEURS_CHECK_ROW,
    PEAKS_OUTPUT,
    BindMount,
    ContainerImage,
    ContainerRun,
    StepLimits,
    StepRow,
)
from mostt_model_builder.domain.containers.devices import Device

WORK_ROOT = PurePosixPath("/data/builds/work")
KEY = "7b1e04c9a2f3" + "0" * 52
HEAD_TREE = "3f1d1ffe79bc3336eb583331cfbf9b189fd45f01"
IMAGE_ID = "sha256:" + "d" * 64
RECIPE_NAME = "parakeet-tdt"
ENTRY = WORK_ROOT / "cache" / RECIPE_NAME / "fleurs-check" / KEY
OUTPUT_DIRECTORY = ENTRY / "output"
REQUEST_DIRECTORY = ENTRY / "request"
LOG_PATH = ENTRY / "step.log"
RECORD_PATH = ENTRY / "record.json"
RECORD_REPLACEMENT_PATH = ENTRY / "record.json.new"
CLEAN_LOG_PATH = ENTRY.parent / f"clean-{KEY}.log"
LINK_FACTS = FileFacts(FileKind.LINK, USER_ID, 0o777, 10, 1)
CANDIDATE_ENTRY = WORK_ROOT / "cache" / RECIPE_NAME / "candidate" / KEY
CANDIDATE_DIRECTORY = CANDIDATE_ENTRY / "files"
CANDIDATE_FILE_NAMES = ("encoder-model.onnx", "vocab.txt")
BUILD_DIRECTORY = WORK_ROOT / "builds" / RECIPE_NAME
REPORT_PATH = BUILD_DIRECTORY / "verification.json"
RECORD_NAME = "record.json"
UNREADABLE_RECORD = b"{not json"
CARD_PATH = BUILD_DIRECTORY / "README.md"
HOST = HostIdentity(USER_ID, GROUP_ID, 20, HostAccounts(False, (), ()))
REQUEST = EntryCleaning(
    RECIPE_NAME, FLEURS_CHECK_ROW.name, KEY, FLEURS_CHECK_ROW.image, WORK_ROOT, HEAD_TREE, HOST
)
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
    files.add_file(RECORD_PATH, build_file_facts(2, 0o444), b"{}")
    files.add_directory(BUILD_DIRECTORY)
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
    return RefusedClean(partial(clean_entry, REQUEST, engine, files), engine, files)


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
    files.add_file(CANDIDATE_ENTRY / "record.json", build_file_facts(2, 0o444), b"{}")
    return files


def build_engine(files: FakeStepFiles, status: int = 0) -> FakeContainerEngine:
    return FakeContainerEngine(
        files=files,
        images={FLEURS_CHECK_ROW.image.tag: ImageFacts(IMAGE_ID, HEAD_TREE)},
        status=status,
    )


@dataclass(frozen=True)
class CleanedCache:
    events: list[CleanEvent]
    entries_before: list[PurePosixPath]
    entries_after: list[PurePosixPath]
    files: FakeStepFiles


def clean_built_cache(
    steps: frozenset[str] | None, hardware: HardwareFacts = GPU_HARDWARE
) -> CleanedCache:
    cache = CacheBuild()
    cache.resolve()
    build_directory = WORK_ROOT / "builds" / CACHE_RECIPE_NAME
    cache.files.add_file(build_directory / "verification.json", build_file_facts(2, 0o644), b"{}")
    cache.files.add_file(build_directory / "README.md", build_file_facts(6, 0o444), b"# card")
    entries_before = list_entries(cache.files)
    events = clean_cache(cache, steps, hardware)
    return CleanedCache(events, entries_before, list_entries(cache.files), cache.files)


class TestTheCleanRun:
    class TestTheCleanEntryFunction:
        class TestWhenTheEntryHoldsOutputsARequestALogAndARecord:
            @pytest.fixture
            def clean_record(self) -> CleanRecord:
                files = build_files_with_a_run()
                engine = build_engine(files)
                outcome = clean_entry(REQUEST, engine, files)
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
                                OUTPUT_DIRECTORY.relative_to(WORK_ROOT),
                                OUTPUT_IN_THE_CONTAINER,
                                is_read_only=False,
                                is_file=False,
                            ),
                        ),
                        False,
                    )
                ]

            def test_should_remove_the_whole_entry(self, clean_record: CleanRecord) -> None:
                assert [
                    path
                    for path in clean_record.files.entries
                    if path == ENTRY or ENTRY in path.parents
                ] == []

            def test_should_log_the_cleaning_beside_the_entry(
                self, clean_record: CleanRecord
            ) -> None:
                assert clean_record.log_paths == [CLEAN_LOG_PATH]

            def test_should_remove_the_log_of_the_cleaning_once_it_succeeded(
                self, clean_record: CleanRecord
            ) -> None:
                assert CLEAN_LOG_PATH not in clean_record.files.entries

            def test_should_say_it_cleaned(self, clean_record: CleanRecord) -> None:
                assert clean_record.outcome is CleanOutcome.CLEANED

        class TestWhenAnInterruptedWriteOfTheRecordLeftItsReplacement:
            @pytest.fixture
            def clean_record(self) -> CleanRecord:
                files = build_files_with_a_run()
                files.remove_file_if_present(RECORD_PATH)
                files.add_file(RECORD_REPLACEMENT_PATH, build_file_facts(2, 0o444), b"{}")
                engine = build_engine(files)
                outcome = clean_entry(REQUEST, engine, files)
                return CleanRecord(outcome, engine.runs, engine.log_paths, files)

            def test_should_remove_the_whole_entry(self, clean_record: CleanRecord) -> None:
                assert [
                    path
                    for path in clean_record.files.entries
                    if path == ENTRY or ENTRY in path.parents
                ] == []

        class TestWhenTheEntryIsMissing:
            @pytest.fixture
            def clean_record(self) -> CleanRecord:
                files = FakeStepFiles()
                files.add_directory(WORK_ROOT)
                engine = build_engine(files)
                outcome = clean_entry(REQUEST, engine, files)
                return CleanRecord(outcome, engine.runs, engine.log_paths, files)

            def test_should_run_no_container(self, clean_record: CleanRecord) -> None:
                assert clean_record.runs == []

            def test_should_say_there_was_nothing_to_clean(self, clean_record: CleanRecord) -> None:
                assert clean_record.outcome is CleanOutcome.NOTHING_TO_CLEAN

        class TestWhenTheCleaningContainerFails:
            @pytest.fixture
            def clean_act(self) -> Callable[[], CleanOutcome]:
                files = build_files_with_a_run()
                return partial(clean_entry, REQUEST, build_engine(files, status=1), files)

            def test_should_stop_and_say_so(self, clean_act: Callable[[], CleanOutcome]) -> None:
                with pytest.raises(
                    StepError, match=re.escape("the step clean exited with the status 1")
                ):
                    clean_act()

        @pytest.mark.parametrize(
            "link_path",
            [ENTRY, OUTPUT_DIRECTORY, REQUEST_DIRECTORY],
            ids=["the entry", "the outputs", "the request"],
        )
        class TestWhenALinkStandsWhereADirectoryOfTheEntryGoes:
            @pytest.fixture
            def clean_act(self, link_path: PurePosixPath) -> Callable[[], CleanOutcome]:
                files = FakeStepFiles()
                files.add_directory(WORK_ROOT)
                files.add_directory(link_path.parent)
                files.entries[link_path] = LINK_FACTS
                return partial(clean_entry, REQUEST, build_engine(files), files)

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

        class TestWhenTheEntryIsOfARecipesStep:
            @pytest.fixture
            def clean_record(self) -> CleanRecord:
                files = FakeStepFiles()
                files.add_directory(WORK_ROOT)
                describe_directory = WORK_ROOT / "cache" / RECIPE_NAME / "describe" / KEY / "output"
                files.add_directory(describe_directory, build_directory_facts(0o2770))
                files.add_file(describe_directory / "peaks.json", *build_step_output(b"{}"))
                engine = FakeContainerEngine(
                    files=files, images={RECIPE_ROW.image.tag: ImageFacts(IMAGE_ID, HEAD_TREE)}
                )
                outcome = clean_entry(
                    EntryCleaning(
                        RECIPE_NAME,
                        RECIPE_ROW.name,
                        KEY,
                        RECIPE_ROW.image,
                        WORK_ROOT,
                        HEAD_TREE,
                        HOST,
                    ),
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

    class TestTheRemoveCandidateEntryFunction:
        class TestWhenTheCacheHoldsTheCandidate:
            @pytest.fixture
            def removal(self) -> CandidateRemoval:
                files = build_files_with_a_candidate()
                return CandidateRemoval(
                    remove_candidate_entry(RECIPE_NAME, KEY, WORK_ROOT, USER_ID, files), files
                )

            def test_should_remove_its_files_its_record_and_its_entry(
                self, removal: CandidateRemoval
            ) -> None:
                assert [
                    path
                    for path in removal.files.entries
                    if path == CANDIDATE_ENTRY or CANDIDATE_ENTRY in path.parents
                ] == []

            def test_should_say_it_cleaned(self, removal: CandidateRemoval) -> None:
                assert removal.outcome is CleanOutcome.CLEANED

        class TestWhenAnInterruptedWriteOfTheRecordLeftItsReplacement:
            @pytest.fixture
            def removal(self) -> CandidateRemoval:
                files = build_files_with_a_candidate()
                files.remove_file_if_present(CANDIDATE_ENTRY / "record.json")
                files.add_file(
                    CANDIDATE_ENTRY / "record.json.new", build_file_facts(2, 0o444), b"{}"
                )
                return CandidateRemoval(
                    remove_candidate_entry(RECIPE_NAME, KEY, WORK_ROOT, USER_ID, files), files
                )

            def test_should_remove_the_whole_entry(self, removal: CandidateRemoval) -> None:
                assert [
                    path
                    for path in removal.files.entries
                    if path == CANDIDATE_ENTRY or CANDIDATE_ENTRY in path.parents
                ] == []

        class TestWhenTheCacheHoldsNoCandidateAtThatKey:
            @pytest.fixture
            def outcome(self) -> CleanOutcome:
                files = FakeStepFiles()
                files.add_directory(WORK_ROOT)
                return remove_candidate_entry(RECIPE_NAME, KEY, WORK_ROOT, USER_ID, files)

            def test_should_say_there_was_nothing_to_clean(self, outcome: CleanOutcome) -> None:
                assert outcome is CleanOutcome.NOTHING_TO_CLEAN

        class TestWhenTheCandidateIsALink:
            @pytest.fixture
            def remove_act(self) -> Callable[[], CleanOutcome]:
                files = FakeStepFiles()
                files.add_directory(CANDIDATE_ENTRY)
                files.entries[CANDIDATE_DIRECTORY] = FileFacts(FileKind.LINK, USER_ID, 0o777, 9, 1)
                return partial(remove_candidate_entry, RECIPE_NAME, KEY, WORK_ROOT, USER_ID, files)

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
        class TestWhenTheBuildHoldsAReport:
            @pytest.fixture
            def removal(self) -> ReportRemoval:
                files = build_files_with_a_run()
                files.add_file(REPORT_PATH, build_file_facts(2, 0o644), b"{}")
                return ReportRemoval(
                    remove_verification_report(RECIPE_NAME, WORK_ROOT, USER_ID, files), files
                )

            def test_should_remove_it(self, removal: ReportRemoval) -> None:
                assert REPORT_PATH not in removal.files.entries

            def test_should_say_it_cleaned(self, removal: ReportRemoval) -> None:
                assert removal.outcome is CleanOutcome.CLEANED

        class TestWhenTheBuildHoldsNoReport:
            @pytest.fixture
            def outcome(self) -> CleanOutcome:
                return remove_verification_report(
                    RECIPE_NAME, WORK_ROOT, USER_ID, build_files_with_a_run()
                )

            def test_should_say_there_was_nothing_to_clean(self, outcome: CleanOutcome) -> None:
                assert outcome is CleanOutcome.NOTHING_TO_CLEAN

        class TestWhenTheBuildNeverVerified:
            @pytest.fixture
            def outcome(self) -> CleanOutcome:
                files = FakeStepFiles()
                files.add_directory(WORK_ROOT)
                return remove_verification_report(RECIPE_NAME, WORK_ROOT, USER_ID, files)

            def test_should_say_there_was_nothing_to_clean(self, outcome: CleanOutcome) -> None:
                assert outcome is CleanOutcome.NOTHING_TO_CLEAN

        class TestWhenTheBuildDirectoryIsALink:
            @pytest.fixture
            def remove_act(self) -> Callable[[], CleanOutcome]:
                files = FakeStepFiles()
                files.add_directory(BUILD_DIRECTORY.parent)
                files.entries[BUILD_DIRECTORY] = FileFacts(FileKind.LINK, USER_ID, 0o777, 9, 1)
                return partial(remove_verification_report, RECIPE_NAME, WORK_ROOT, USER_ID, files)

            def test_should_refuse_to_follow_it(
                self, remove_act: Callable[[], CleanOutcome]
            ) -> None:
                with pytest.raises(
                    ContainerFileError,
                    match=re.escape(f"{BUILD_DIRECTORY} is a link, where a directory is declared"),
                ):
                    remove_act()

    class TestTheRemoveModelCardFunction:
        class TestWhenTheBuildHoldsTheCardForReview:
            @pytest.fixture
            def removal(self) -> ReportRemoval:
                files = build_files_with_a_run()
                files.add_file(CARD_PATH, build_file_facts(5, 0o444), b"# card")
                return ReportRemoval(
                    remove_model_card(RECIPE_NAME, WORK_ROOT, USER_ID, files), files
                )

            def test_should_remove_it_since_it_was_rendered_from_the_verification(
                self, removal: ReportRemoval
            ) -> None:
                assert (CARD_PATH in removal.files.entries, removal.outcome) == (
                    False,
                    CleanOutcome.CLEANED,
                )

        class TestWhenTheBuildHoldsNoCard:
            @pytest.fixture
            def outcome(self) -> CleanOutcome:
                return remove_model_card(RECIPE_NAME, WORK_ROOT, USER_ID, build_files_with_a_run())

            def test_should_say_there_was_nothing_to_clean(self, outcome: CleanOutcome) -> None:
                assert outcome is CleanOutcome.NOTHING_TO_CLEAN

    class TestTheCleanBuildFunction:
        class TestWhenEveryStepIsCleaned:
            @pytest.fixture
            def cleaned(self) -> CleanedCache:
                return clean_built_cache(steps=None)

            def test_should_remove_every_entry_of_the_current_build(
                self, cleaned: CleanedCache
            ) -> None:
                assert (len(cleaned.entries_before), cleaned.entries_after) == (8, [])

            def test_should_leave_nothing_of_the_entries_in_the_cache(
                self, cleaned: CleanedCache
            ) -> None:
                assert [
                    path
                    for path in cleaned.files.entries
                    if path.is_relative_to(WORK_ROOT / "cache")
                    and len(path.relative_to(WORK_ROOT / "cache").parts) > 2
                ] == []

            def test_should_remove_the_report_and_the_card(self, cleaned: CleanedCache) -> None:
                assert [
                    event for event in cleaned.events if isinstance(event, BuildFileCleaned)
                ] == [
                    BuildFileCleaned(BuildFile.VERIFICATION_REPORT, CleanOutcome.CLEANED),
                    BuildFileCleaned(BuildFile.MODEL_CARD, CleanOutcome.CLEANED),
                ]

        class TestWhenOneStepIsCleaned:
            @pytest.fixture
            def cleaned(self) -> CleanedCache:
                return clean_built_cache(steps=frozenset({"transcribe"}))

            def test_should_remove_the_entries_of_that_step_alone(
                self, cleaned: CleanedCache
            ) -> None:
                assert [entry.parts[0] for entry in cleaned.entries_after] == [
                    "candidate",
                    "export",
                    "fleurs-check",
                    "inspect",
                ]

            def test_should_say_which_entries_it_cleaned(self, cleaned: CleanedCache) -> None:
                assert [
                    (event.name, event.outcome)
                    for event in cleaned.events
                    if isinstance(event, EntryCleaned)
                ] == [
                    (name, CleanOutcome.CLEANED)
                    for name in (
                        "transcribe-float32-cpu-fr_fr-0",
                        "transcribe-float32-cpu-en_us-0",
                        "transcribe-float32-gpu-fr_fr-0",
                        "transcribe-float32-gpu-en_us-0",
                    )
                ]

        class TestWhenAnEntryWasCorruptedAfterItsRecord:
            @pytest.fixture
            def rerun_rows(self) -> list[str]:
                cache = CacheBuild()
                resolved, _ = cache.resolve()
                entry = PurePosixPath("cache/family/export", resolved.entries["export"].key)
                rewrite_output(cache.files, entry, "vocab.txt", b"x" * 13)
                clean_cache(cache, frozenset({"export"}))
                _, engine = cache.resolve()
                return list_ran_rows(engine)

            def test_should_remove_it_unjudged_so_that_the_next_build_runs_it_again(
                self, rerun_rows: list[str]
            ) -> None:
                assert rerun_rows == ["export"]

        class TestWhenTheRecordOfAnEntryIsRefused:
            @pytest.fixture
            def cleaned(self) -> CleanedCache:
                cache = CacheBuild()
                resolved, _ = cache.resolve()
                record_path = (
                    WORK_ROOT / "cache/family/export" / resolved.entries["export"].key / RECORD_NAME
                )
                cache.files.contents[record_path] = UNREADABLE_RECORD
                cache.files.entries[record_path] = build_file_facts(len(UNREADABLE_RECORD), 0o444)
                entries_before = list_entries(cache.files)
                events = clean_cache(cache, steps=None)
                return CleanedCache(events, entries_before, list_entries(cache.files), cache.files)

            def test_should_remove_it_unjudged(self, cleaned: CleanedCache) -> None:
                assert [
                    (event.name, event.outcome)
                    for event in cleaned.events
                    if isinstance(event, EntryCleaned) and event.name == "export"
                ] == [("export", CleanOutcome.CLEANED)]

            def test_should_skip_the_entries_whose_keys_need_its_digests_and_say_why(
                self, cleaned: CleanedCache
            ) -> None:
                skipped = [event for event in cleaned.events if isinstance(event, EntrySkipped)]
                assert skipped[:1] == [
                    EntrySkipped(
                        "candidate", "the record of export, which its key needs, is refused"
                    )
                ]

            def test_should_leave_only_the_skipped_entries(self, cleaned: CleanedCache) -> None:
                assert sorted({entry.parts[0] for entry in cleaned.entries_after}) == sorted(
                    entry.parts[0]
                    for entry in cleaned.entries_before
                    if entry.parts[0] not in {"export", "fleurs-check"}
                )

        class TestWhenNoGpuIsReadable:
            @pytest.fixture
            def cleaned(self) -> CleanedCache:
                return clean_built_cache(steps=None, hardware=HARDWARE_WITHOUT_GPU)

            def test_should_skip_the_runs_on_the_gpu_and_say_why(
                self, cleaned: CleanedCache
            ) -> None:
                assert [event for event in cleaned.events if isinstance(event, EntrySkipped)] == [
                    EntrySkipped(name, NO_GPU_REASON)
                    for name in ("transcribe-float32-gpu-fr_fr-0", "transcribe-float32-gpu-en_us-0")
                ]

            def test_should_remove_every_other_entry(self, cleaned: CleanedCache) -> None:
                assert [entry.parts[0] for entry in cleaned.entries_after] == [
                    "transcribe-float32-gpu-en_us-0",
                    "transcribe-float32-gpu-fr_fr-0",
                ]
