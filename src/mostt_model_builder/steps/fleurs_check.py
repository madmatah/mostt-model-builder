from pathlib import Path, PurePosixPath

from mostt_model_builder.domain.bench.fleurs_check_report import (
    FLEURS_CHECK_REPORT_FILE_NAME,
    FleursCheckReport,
    LanguageCheck,
    RecordingDuration,
    render_fleurs_check_report,
)
from mostt_model_builder.domain.bench.fleurs_check_request import (
    FLEURS_CHECK_REQUEST_FILE_NAME,
    BenchFiles,
    parse_fleurs_check_request,
)
from mostt_model_builder.domain.bench.fleurs_index import FleursUtterance, parse_fleurs_index
from mostt_model_builder.domain.bench.wav_format import parse_wav_layout
from mostt_model_builder.domain.containers.container_paths import (
    CONTAINER_ROOT,
    OUTPUT_IN_THE_CONTAINER,
    REQUEST_IN_THE_CONTAINER,
)
from mostt_model_builder.steps.fleurs_audio import read_recordings
from mostt_model_builder.steps.output_modes import finish_outputs


class FleursCheckError(Exception):
    pass


def measure_recording(utterance: FleursUtterance, content: bytes) -> int:
    sample_count = parse_wav_layout(content).sample_count
    if sample_count != utterance.sample_count:
        raise FleursCheckError(
            f"the recording {utterance.recording_name} holds {sample_count} samples, "
            f"where its index gives {utterance.sample_count}"
        )
    return sample_count


def locate_in_the_container(container_root: Path, container_path: PurePosixPath) -> Path:
    return container_root / container_path.relative_to(CONTAINER_ROOT)


def check_language(bench_files: BenchFiles, container_root: Path) -> LanguageCheck:
    index_path = locate_in_the_container(container_root, bench_files.index)
    archive_path = locate_in_the_container(container_root, bench_files.archive)
    utterances = {
        utterance.recording_name: utterance
        for utterance in parse_fleurs_index(index_path.read_text(encoding="utf-8"))
    }
    durations = {
        recording_name: measure_recording(utterances[recording_name], content)
        for recording_name, content in read_recordings(archive_path, utterances)
    }
    return LanguageCheck(
        language=bench_files.language,
        recordings=tuple(
            RecordingDuration(recording_name, durations[recording_name])
            for recording_name in utterances
        ),
    )


def check_fleurs(
    request_directory: Path, output_directory: Path, container_root: Path = Path(CONTAINER_ROOT)
) -> None:
    request = parse_fleurs_check_request(
        (request_directory / FLEURS_CHECK_REQUEST_FILE_NAME).read_bytes()
    )
    report = FleursCheckReport(
        tuple(check_language(files, container_root) for files in request.languages)
    )
    (output_directory / FLEURS_CHECK_REPORT_FILE_NAME).write_bytes(
        render_fleurs_check_report(report)
    )


def main() -> None:
    output_directory = Path(OUTPUT_IN_THE_CONTAINER)
    check_fleurs(Path(REQUEST_IN_THE_CONTAINER), output_directory)
    finish_outputs(output_directory)


if __name__ == "__main__":
    main()
