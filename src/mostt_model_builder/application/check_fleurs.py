from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import PurePosixPath

from mostt_model_builder.application.run_step import (
    ContainerEngine,
    HostIdentity,
    StepFiles,
    StepRunRequest,
    run_step,
)
from mostt_model_builder.domain.bench.fleurs_check_report import (
    FLEURS_CHECK_REPORT_FILE_NAME,
    FleursCheckReport,
    parse_fleurs_check_report,
)
from mostt_model_builder.domain.bench.fleurs_check_request import (
    FLEURS_CHECK_REQUEST_FILE_NAME,
    BenchFiles,
    FleursCheckRequest,
    render_fleurs_check_request,
)
from mostt_model_builder.domain.bench.fleurs_index import (
    BENCH_LANGUAGES,
    FLEURS_REPOSITORY,
    locate_bench_files,
)
from mostt_model_builder.domain.containers.container_paths import CONTAINER_ROOT
from mostt_model_builder.domain.containers.container_steps import FLEURS_CHECK_ROW, SourceReference
from mostt_model_builder.domain.containers.step_peaks import StepPeaks
from mostt_model_builder.domain.containers.work_directory import (
    locate_referenced_sources,
    place_step,
)
from mostt_model_builder.domain.sources.source_pins import SourcePin


@dataclass(frozen=True, slots=True)
class FleursCheckRun:
    pins: Sequence[SourcePin]
    recipe_name: str
    commit: str
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
    source_paths = locate_referenced_sources(row.sources, run.pins)
    bench_files = []
    for language in BENCH_LANGUAGES:
        index_path, archive_path = locate_bench_files(language)
        bench_files.append(
            BenchFiles(
                language=language,
                index=CONTAINER_ROOT / source_paths[SourceReference(FLEURS_REPOSITORY, index_path)],
                archive=CONTAINER_ROOT
                / source_paths[SourceReference(FLEURS_REPOSITORY, archive_path)],
            )
        )
    request_content = render_fleurs_check_request(FleursCheckRequest(tuple(bench_files)))
    outputs = run_step(
        StepRunRequest(
            row=row,
            placement=place_step(run.recipe_name, run.commit, row.name),
            request_files={FLEURS_CHECK_REQUEST_FILE_NAME: request_content},
            source_paths=source_paths,
            work_root=run.work_root,
            head_tree=run.head_tree,
            host=run.host,
            parsed_output_names=frozenset({FLEURS_CHECK_REPORT_FILE_NAME}),
        ),
        engine,
        files,
    )
    report = parse_fleurs_check_report(
        outputs.contents[FLEURS_CHECK_REPORT_FILE_NAME], BENCH_LANGUAGES
    )
    return FleursCheckOutcome(report, outputs.peaks)
