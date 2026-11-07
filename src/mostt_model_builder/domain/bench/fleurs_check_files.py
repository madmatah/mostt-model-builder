from collections.abc import Sequence

from mostt_model_builder.domain.bench.fleurs_check_request import (
    FLEURS_CHECK_REQUEST_FILE_NAME,
    BenchFiles,
    FleursCheckRequest,
    render_fleurs_check_request,
)
from mostt_model_builder.domain.bench.fleurs_index import FLEURS_REPOSITORY, locate_bench_files
from mostt_model_builder.domain.containers.container_paths import CONTAINER_ROOT
from mostt_model_builder.domain.containers.container_steps import SourceReference, StepRow
from mostt_model_builder.domain.containers.work_directory import locate_referenced_sources
from mostt_model_builder.domain.sources.source_pins import SourcePin


def render_fleurs_check_files(
    row: StepRow, languages: Sequence[str], pins: Sequence[SourcePin]
) -> dict[str, bytes]:
    source_paths = locate_referenced_sources(row.sources, pins)
    bench_files = []
    for language in languages:
        index_path, archive_path = locate_bench_files(language)
        bench_files.append(
            BenchFiles(
                language=language,
                index=CONTAINER_ROOT / source_paths[SourceReference(FLEURS_REPOSITORY, index_path)],
                archive=CONTAINER_ROOT
                / source_paths[SourceReference(FLEURS_REPOSITORY, archive_path)],
            )
        )
    return {
        FLEURS_CHECK_REQUEST_FILE_NAME: render_fleurs_check_request(
            FleursCheckRequest(tuple(bench_files))
        )
    }
