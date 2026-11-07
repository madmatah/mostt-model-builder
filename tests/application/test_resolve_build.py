import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import partial
from pathlib import PurePosixPath

import pytest
from cache_arrangement import (
    CLEAN_MODULE,
    EXPORT_MODULE,
    GPU_HARDWARE,
    HARDWARE_WITHOUT_GPU,
    LANGUAGES,
    TRANSCRIBE_MODULE,
    TREE_FILES,
    VOCABULARY,
    VOCABULARY_NAME,
    WEIGHTS_PIN,
    WORK_ROOT,
    CacheBuild,
    RecipeShape,
    build_engine,
    build_recipe,
    find_request_content,
    list_ran_rows,
    list_reused_entries,
    rewrite_output,
    rewrite_record,
    write_transcription,
)
from run_step_fakes import FakeStepFiles

from mostt_model_builder.application.cache_entries import CacheEntryError
from mostt_model_builder.application.resolve_build import (
    ResolvedBuild,
    RunLanguageTranscribed,
    UnfinishedEntryRemoved,
)
from mostt_model_builder.domain.bench.fleurs_check_report import (
    FLEURS_CHECK_REPORT_FILE_NAME,
    FleursCheckReport,
    LanguageCheck,
    RecordingDuration,
    render_fleurs_check_report,
)
from mostt_model_builder.domain.bench.run_lines import RUN_LINES_FILE_NAME
from mostt_model_builder.domain.cache.hardware_facts import HardwareFacts
from mostt_model_builder.domain.cache.output_keys import OutputKeyError
from mostt_model_builder.domain.containers.container_files import ContainerFileError, FileFacts
from mostt_model_builder.domain.containers.container_steps import ContainerRun
from mostt_model_builder.domain.contract.shard_request import (
    SHARD_REQUEST_FILE_NAME,
    parse_shard_request,
)

CPU_ROWS = [f"transcribe-float32-cpu-{language}-0" for language in LANGUAGES]
GPU_ROWS = [f"transcribe-float32-gpu-{language}-0" for language in LANGUAGES]
EVERY_ROW = ["fleurs-check", "export", "inspect", *CPU_ROWS, *GPU_ROWS]
EVERY_ENTRY = ["fleurs-check", "export", "candidate", "inspect", *CPU_ROWS, *GPU_ROWS]
TRANSCRIBE_READER = PurePosixPath("recipes/family/src/mostt_recipe_family/domain/reader.py")
EXPORT_CONTAINERFILE = PurePosixPath("recipes/family/images/export/Containerfile")
IMAGES_LOCK = PurePosixPath("images/environment.lock")
ONNX_ENTRY_START = '[[package]]\nname = "onnx"\n'


def change_onnx_hash(lock: bytes) -> bytes:
    text = lock.decode()
    start = text.index(ONNX_ENTRY_START)
    digest_start = text.index('hash = "sha256:', start) + len('hash = "sha256:')
    return (text[:digest_start] + "f" * 64 + text[digest_start + 64 :]).encode()


def locate_entry(resolved: ResolvedBuild, name: str) -> PurePosixPath:
    return PurePosixPath("cache/family", name, resolved.entries[name].key)


@dataclass(frozen=True)
class KeyComponentChange:
    tree_files: Mapping[PurePosixPath, bytes]
    hardware: HardwareFacts
    shape: RecipeShape
    rerun_rows: list[str]


COMPONENT_CHANGES = {
    "a file in the import closure of the transcriptions": KeyComponentChange(
        {**TREE_FILES, TRANSCRIBE_READER: b"READ = 2\n"},
        GPU_HARDWARE,
        RecipeShape(),
        CPU_ROWS + GPU_ROWS,
    ),
    "a package of the inspection image's extra in the lock": KeyComponentChange(
        {**TREE_FILES, IMAGES_LOCK: change_onnx_hash(TREE_FILES[IMAGES_LOCK])},
        GPU_HARDWARE,
        RecipeShape(),
        ["inspect"],
    ),
    "the export's Containerfile, whose rerun writes the same bytes": KeyComponentChange(
        {**TREE_FILES, EXPORT_CONTAINERFILE: b"FROM python:3.12\n"},
        GPU_HARDWARE,
        RecipeShape(),
        ["export"],
    ),
    "a pin that the export binds": KeyComponentChange(
        TREE_FILES, GPU_HARDWARE, RecipeShape(export_sources=(WEIGHTS_PIN,)), ["export"]
    ),
    "the request of the runs on the CPU": KeyComponentChange(
        TREE_FILES, GPU_HARDWARE, RecipeShape(cpu_variant="float32-again"), CPU_ROWS
    ),
    "the driver's version, for the runs on the GPU": KeyComponentChange(
        TREE_FILES,
        HardwareFacts(GPU_HARDWARE.cpu_models, GPU_HARDWARE.gpu_models, "560.35.03"),
        RecipeShape(),
        GPU_ROWS,
    ),
}


TWO_SHARDS = RecipeShape(cpu_shard_count=2)
FIRST_FRENCH_SHARD = "transcribe-float32-cpu-fr_fr-0"
SECOND_FRENCH_SHARD = "transcribe-float32-cpu-fr_fr-1"


def write_lines_of_first_shard(
    run: ContainerRun, files: FakeStepFiles
) -> dict[str, tuple[FileFacts, bytes]]:
    outputs = write_transcription(run, files)
    request = parse_shard_request(find_request_content(run, files, SHARD_REQUEST_FILE_NAME))
    return {
        name: written
        for name, written in outputs.items()
        if request.shard == 0 or name != RUN_LINES_FILE_NAME
    }


@dataclass(frozen=True)
class FailedBuild:
    resolve: Callable[[], object]
    cache: CacheBuild


@dataclass(frozen=True)
class RerunEntry:
    removed: list[UnfinishedEntryRemoved]
    entry_modules: list[str]


@dataclass(frozen=True)
class RefusedResolution:
    resolve: Callable[[], object]
    entry: PurePosixPath


@dataclass(frozen=True)
class TwoBuilds:
    first_rows: list[str]
    second_rows: list[str]
    second_reused: list[str]


class TestTheBuildResolution:
    class TestTheResolveBuildFunction:
        class TestWhenTheCacheIsEmpty:
            @pytest.fixture
            def two_builds(self) -> TwoBuilds:
                cache = CacheBuild()
                _, first_engine = cache.resolve()
                _, second_engine = cache.resolve()
                return TwoBuilds(
                    list_ran_rows(first_engine),
                    list_ran_rows(second_engine),
                    list_reused_entries(cache.events),
                )

            def test_should_run_every_row_once_in_the_build_order(
                self, two_builds: TwoBuilds
            ) -> None:
                assert two_builds.first_rows == EVERY_ROW

            def test_should_run_no_container_when_built_again(self, two_builds: TwoBuilds) -> None:
                assert two_builds.second_rows == []

            def test_should_reuse_every_entry_when_built_again(self, two_builds: TwoBuilds) -> None:
                assert two_builds.second_reused == EVERY_ENTRY

        @pytest.mark.parametrize("change", COMPONENT_CHANGES.values(), ids=COMPONENT_CHANGES.keys())
        class TestWhenAComponentOfAKeyChanges:
            @pytest.fixture
            def rerun_rows(self, change: KeyComponentChange) -> list[str]:
                cache = CacheBuild()
                cache.resolve()
                _, engine = cache.resolve(
                    recipe=build_recipe(change.shape),
                    tree_files=change.tree_files,
                    hardware=change.hardware,
                )
                return list_ran_rows(engine)

            def test_should_run_exactly_the_rows_whose_key_it_changes(
                self, rerun_rows: list[str], change: KeyComponentChange
            ) -> None:
                assert rerun_rows == change.rerun_rows

        class TestWhenAnOutputChangedAfterItsRecordAtTheSameSize:
            @pytest.fixture
            def refused_resolution(self) -> RefusedResolution:
                cache = CacheBuild()
                resolved, _ = cache.resolve()
                entry = locate_entry(resolved, "export")
                rewrite_output(cache.files, entry, VOCABULARY_NAME, b"x" * len(VOCABULARY))
                return RefusedResolution(cache.resolve, entry)

            def test_should_refuse_it_and_name_the_entry_and_the_cleaning_of_its_step(
                self, refused_resolution: RefusedResolution
            ) -> None:
                with pytest.raises(
                    CacheEntryError,
                    match=re.escape(
                        f"the entry {refused_resolution.entry} of export is refused: the output "
                        "vocab.txt"
                    )
                    + ".*"
                    + re.escape("clean export"),
                ):
                    refused_resolution.resolve()

        class TestWhenOneShardOfALanguageIsReusedAndAnotherRuns:
            @pytest.fixture
            def utterances(self) -> list[str]:
                cache = CacheBuild()
                resolved, _ = cache.resolve(recipe=build_recipe(TWO_SHARDS))
                entry = locate_entry(resolved, SECOND_FRENCH_SHARD)
                cache.files.remove_file_if_present(WORK_ROOT / entry / "record.json")
                cache.resolve(recipe=build_recipe(TWO_SHARDS))
                return [
                    hypothesis.utterance_id
                    for event in cache.events
                    if isinstance(event, RunLanguageTranscribed)
                    for hypothesis in event.hypotheses
                ]

            def test_should_give_the_figures_of_the_whole_language_reused_shards_included(
                self, utterances: list[str]
            ) -> None:
                assert utterances == ["1.wav", "2.wav"]

        class TestWhenOneShardOfALanguageFailsAndAnotherSucceeds:
            @pytest.fixture
            def failed_build(self) -> FailedBuild:
                cache = CacheBuild()
                engine = build_engine(cache.files)
                engine.outputs_by_entry_module[TRANSCRIBE_MODULE] = write_lines_of_first_shard
                return FailedBuild(
                    partial(cache.resolve, recipe=build_recipe(TWO_SHARDS), engine=engine), cache
                )

            def test_should_fail_and_still_record_the_shard_that_ran_for_the_next_build(
                self, failed_build: FailedBuild
            ) -> None:
                with pytest.raises(ContainerFileError, match=re.escape(RUN_LINES_FILE_NAME)):
                    failed_build.resolve()
                assert [
                    path.parts[-3]
                    for path in failed_build.cache.files.entries
                    if path.name == "record.json" and "-cpu-fr_fr-" in path.parts[-3]
                ] == [FIRST_FRENCH_SHARD]

        class TestWhenAnEntryHasNoRecord:
            @pytest.fixture
            def rerun(self) -> RerunEntry:
                cache = CacheBuild()
                resolved, _ = cache.resolve()
                entry = locate_entry(resolved, "export")
                cache.files.remove_file_if_present(WORK_ROOT / entry / "record.json")
                _, engine = cache.resolve()
                return RerunEntry(
                    [event for event in cache.events if isinstance(event, UnfinishedEntryRemoved)],
                    [run.entry_module for run in engine.runs],
                )

            def test_should_say_it_removed_the_unfinished_entry(self, rerun: RerunEntry) -> None:
                assert [event.name for event in rerun.removed] == ["export"]

            def test_should_empty_it_through_the_cleaning_container_then_run_its_row(
                self, rerun: RerunEntry
            ) -> None:
                assert rerun.entry_modules == [CLEAN_MODULE, EXPORT_MODULE]

        class TestWhenNothingMayRunAndARowIsMissing:
            @pytest.fixture
            def resolve_act(self) -> Callable[[], object]:
                return partial(CacheBuild().resolve, running_steps=frozenset())

            def test_should_refuse_it_and_name_the_build(
                self, resolve_act: Callable[[], object]
            ) -> None:
                with pytest.raises(
                    CacheEntryError,
                    match=r"fleurs-check has no entry at its key [0-9a-f]{12}: make it with "
                    + re.escape("./model-build.sh --work <dir> --recipe <name> build"),
                ):
                    resolve_act()

        class TestWhenAReusedReportIsOneThatTodaysParserRefuses:
            @pytest.fixture
            def resolve_act(self) -> Callable[[], object]:
                cache = CacheBuild()
                resolved, _ = cache.resolve()
                entry = locate_entry(resolved, "fleurs-check")
                french_only = FleursCheckReport(
                    (LanguageCheck("fr_fr", (RecordingDuration("1.wav", 16000),)),)
                )
                rewrite_output(
                    cache.files,
                    entry,
                    FLEURS_CHECK_REPORT_FILE_NAME,
                    render_fleurs_check_report(french_only),
                )
                rewrite_record(cache.files, entry, "output")
                return cache.resolve

            def test_should_refuse_its_entry_as_its_parser_would_right_after_its_run(
                self, resolve_act: Callable[[], object]
            ) -> None:
                with pytest.raises(
                    CacheEntryError,
                    match=re.escape("of fleurs-check is refused: the report checks the languages")
                    + ".*"
                    + re.escape("clean fleurs-check"),
                ):
                    resolve_act()

        class TestWhenAReusedCandidateBeginsWithARefusedSignature:
            @pytest.fixture
            def resolve_act(self) -> Callable[[], object]:
                cache = CacheBuild()
                resolved, _ = cache.resolve()
                entry = locate_entry(resolved, "candidate")
                archive = b"PK\x03\x04" + bytes(len(VOCABULARY) - 4)
                path = WORK_ROOT / entry / "files" / VOCABULARY_NAME
                cache.files.contents[path] = archive
                rewrite_record(cache.files, entry, "files")
                return cache.resolve

            def test_should_refuse_its_entry_through_the_checks_of_its_copy(
                self, resolve_act: Callable[[], object]
            ) -> None:
                with pytest.raises(
                    CacheEntryError,
                    match=re.escape(
                        f"of candidate is refused: the candidate file {VOCABULARY_NAME} begins as"
                    )
                    + ".*"
                    + re.escape("clean inspect"),
                ):
                    resolve_act()

        class TestWhenAReusedCandidateLacksACopy:
            @pytest.fixture
            def resolve_act(self) -> Callable[[], object]:
                cache = CacheBuild()
                resolved, _ = cache.resolve()
                entry = locate_entry(resolved, "candidate")
                cache.files.remove_file_if_present(WORK_ROOT / entry / "files" / VOCABULARY_NAME)
                return cache.resolve

            def test_should_refuse_its_entry_and_name_its_cleaning_rather_than_its_step(
                self, resolve_act: Callable[[], object]
            ) -> None:
                with pytest.raises(
                    CacheEntryError,
                    match=re.escape(
                        f"of candidate is refused: the candidate file {VOCABULARY_NAME} is missing"
                    )
                    + ".*"
                    + re.escape("clean inspect"),
                ):
                    resolve_act()

        class TestWhenNoGpuIsReadableForARunOnTheGpu:
            @pytest.fixture
            def resolve_act(self) -> Callable[[], object]:
                return partial(CacheBuild().resolve, hardware=HARDWARE_WITHOUT_GPU)

            def test_should_refuse_its_key_and_say_why(
                self, resolve_act: Callable[[], object]
            ) -> None:
                with pytest.raises(
                    OutputKeyError,
                    match=re.escape(
                        "transcribe-float32-gpu-fr_fr-0: /proc/driver/nvidia holds no GPU"
                    ),
                ):
                    resolve_act()
