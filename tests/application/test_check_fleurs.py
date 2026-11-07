import json
import re
from collections.abc import Callable
from dataclasses import dataclass, replace
from functools import partial
from pathlib import PurePosixPath

import pytest
from run_step_fakes import (
    GROUP_ID,
    USER_ID,
    FakeContainerEngine,
    FakeStepFiles,
    build_file_facts,
    build_step_output,
)

from mostt_model_builder.application.check_fleurs import (
    FleursCheckOutcome,
    FleursCheckRun,
    check_fleurs_bench,
)
from mostt_model_builder.application.run_step import HostIdentity, ImageFacts
from mostt_model_builder.domain.bench.bench_languages import BENCH_LANGUAGES
from mostt_model_builder.domain.bench.fleurs_check_report import (
    FleursCheckReport,
    LanguageCheck,
    RecordingDuration,
    render_fleurs_check_report,
)
from mostt_model_builder.domain.bench.fleurs_index import locate_bench_files
from mostt_model_builder.domain.containers.account_ranges import HostAccounts
from mostt_model_builder.domain.containers.container_steps import FLEURS_CHECK_ROW, ContainerRun
from mostt_model_builder.domain.containers.step_peaks import (
    PEAKS_FILE_NAME,
    StepPeaks,
    render_step_peaks,
)
from mostt_model_builder.domain.containers.work_directory import WorkDirectoryError, place_entry
from mostt_model_builder.domain.sources.source_pins import RepositoryType, SourcePin

WORK_ROOT = PurePosixPath("/data/builds/work")
KEY = "7b1e04c9a2f3" + "0" * 52
HEAD_TREE = "3f1d1ffe79bc3336eb583331cfbf9b189fd45f01"
REVISION = "70bb2e84b976b7e960aa89f1c648e09c59f894dd"
IMAGE_ID = "sha256:" + "f" * 64
BENCH_PATHS = tuple(path for language in BENCH_LANGUAGES for path in locate_bench_files(language))
ENGLISH_AUDIO_PATH = "data/en_us/audio/test.tar.gz"
PINS = tuple(
    SourcePin(RepositoryType.DATASET, "google/fleurs", REVISION, path, 1, "0" * 64)
    for path in BENCH_PATHS
)
SOURCES = PurePosixPath(f"sources/google/fleurs/{REVISION}")
HOST = HostIdentity(USER_ID, GROUP_ID, 20, HostAccounts(False, (), ()))
REPORT = FleursCheckReport(
    tuple(
        LanguageCheck(language, (RecordingDuration("7105.wav", 150_720),))
        for language in BENCH_LANGUAGES
    )
)
PEAKS = StepPeaks(memory_peak_bytes=21 * 2**20, process_count_peak=6)
RECIPE_NAME = "parakeet-tdt"
PLACEMENT = place_entry(RECIPE_NAME, FLEURS_CHECK_ROW.name, KEY)
RUN = FleursCheckRun(PINS, PLACEMENT, WORK_ROOT, HEAD_TREE, HOST)
REQUEST_PATH = WORK_ROOT / f"cache/{RECIPE_NAME}/fleurs-check/{KEY}/request/request.json"


@dataclass(frozen=True)
class CheckRecord:
    outcome: FleursCheckOutcome
    runs: list[ContainerRun]
    files: FakeStepFiles


def build_files() -> FakeStepFiles:
    files = FakeStepFiles()
    files.add_directory(WORK_ROOT)
    for path in BENCH_PATHS:
        files.add_file(WORK_ROOT / SOURCES / path, build_file_facts(1, 0o444), b"x")
    return files


def build_engine(files: FakeStepFiles) -> FakeContainerEngine:
    return FakeContainerEngine(
        files=files,
        images={FLEURS_CHECK_ROW.image.tag: ImageFacts(IMAGE_ID, HEAD_TREE)},
        step_outputs={
            "fleurs-check.json": build_step_output(render_fleurs_check_report(REPORT)),
            PEAKS_FILE_NAME: build_step_output(render_step_peaks(PEAKS)),
        },
    )


class TestTheCheckFleurs:
    class TestTheCheckFleursBenchFunction:
        class TestWhenTheBenchIsFetched:
            @pytest.fixture
            def check_record(self) -> CheckRecord:
                files = build_files()
                engine = build_engine(files)
                return CheckRecord(check_fleurs_bench(RUN, engine, files), engine.runs, files)

            def test_should_ask_for_each_languages_index_and_audio_where_they_are_bound(
                self, check_record: CheckRecord
            ) -> None:
                assert json.loads(check_record.files.contents[REQUEST_PATH]) == {
                    "languages": [
                        {
                            "language": language,
                            "index": f"/{SOURCES}/data/{language}/test.tsv",
                            "archive": f"/{SOURCES}/data/{language}/audio/test.tar.gz",
                        }
                        for language in BENCH_LANGUAGES
                    ]
                }

            def test_should_bind_every_file_of_the_bench_each_by_itself(
                self, check_record: CheckRecord
            ) -> None:
                assert [
                    mount.host_path
                    for run in check_record.runs
                    for mount in run.mounts
                    if mount.is_file
                ] == [SOURCES / path for path in BENCH_PATHS]

            def test_should_give_the_report_it_judged(self, check_record: CheckRecord) -> None:
                assert check_record.outcome == FleursCheckOutcome(REPORT, PEAKS)

        class TestWhenAPinOfTheBenchIsMissing:
            @pytest.fixture
            def check_act(self) -> Callable[[], FleursCheckOutcome]:
                files = build_files()
                pins_without_the_english_audio = tuple(
                    pin for pin in PINS if pin.path != ENGLISH_AUDIO_PATH
                )
                return partial(
                    check_fleurs_bench,
                    replace(RUN, pins=pins_without_the_english_audio),
                    build_engine(files),
                    files,
                )

            def test_should_refuse_the_step(
                self, check_act: Callable[[], FleursCheckOutcome]
            ) -> None:
                with pytest.raises(
                    WorkDirectoryError,
                    match=re.escape(f"no pin gives {ENGLISH_AUDIO_PATH} of google/fleurs"),
                ):
                    check_act()
