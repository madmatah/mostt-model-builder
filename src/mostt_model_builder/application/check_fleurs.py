from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import PurePosixPath

from mostt_model_builder.application.run_step import (
    ContainerEngine,
    HostIdentity,
    StepFiles,
    StepOutputs,
    StepRunRequest,
    run_step,
)
from mostt_model_builder.domain.bench.bench_languages import BENCH_LANGUAGES
from mostt_model_builder.domain.bench.fleurs_check_files import render_fleurs_check_files
from mostt_model_builder.domain.bench.fleurs_check_report import (
    FLEURS_CHECK_REPORT_FILE_NAME,
    FleursCheckReport,
    parse_fleurs_check_report,
)
from mostt_model_builder.domain.containers.container_steps import FLEURS_CHECK_ROW, StepPlacement
from mostt_model_builder.domain.containers.step_peaks import StepPeaks
from mostt_model_builder.domain.containers.work_directory import locate_referenced_sources
from mostt_model_builder.domain.sources.source_pins import SourcePin

FLEURS_CHECK_PARSED_NAMES = frozenset({FLEURS_CHECK_REPORT_FILE_NAME})


@dataclass(frozen=True, slots=True)
class FleursCheckRun:
    pins: Sequence[SourcePin]
    placement: StepPlacement
    work_root: PurePosixPath
    head_tree: str
    host: HostIdentity


@dataclass(frozen=True, slots=True)
class FleursCheckOutcome:
    report: FleursCheckReport
    peaks: StepPeaks


def check_fleurs_bench(
    run: FleursCheckRun, engine: ContainerEngine, files: StepFiles
) -> FleursCheckOutcome:
    row = FLEURS_CHECK_ROW
    outputs = run_step(
        StepRunRequest(
            row=row,
            placement=run.placement,
            request_files=render_fleurs_check_files(row, BENCH_LANGUAGES, run.pins),
            source_paths=locate_referenced_sources(row.sources, run.pins),
            work_root=run.work_root,
            head_tree=run.head_tree,
            host=run.host,
            parsed_output_names=FLEURS_CHECK_PARSED_NAMES,
        ),
        engine,
        files,
    )
    return read_fleurs_check(outputs)


def read_fleurs_check(outputs: StepOutputs) -> FleursCheckOutcome:
    report = parse_fleurs_check_report(
        outputs.contents[FLEURS_CHECK_REPORT_FILE_NAME], BENCH_LANGUAGES
    )
    return FleursCheckOutcome(report, outputs.peaks)
