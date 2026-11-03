from pathlib import Path

from mostt_model_builder.domain.containers.step_peaks import StepPeaks, render_step_peaks

CONTROL_GROUP_DIRECTORY = Path("/sys/fs/cgroup")
MEMORY_PEAK_FILE_NAME = "memory.peak"
PROCESS_COUNT_PEAK_FILE_NAME = "pids.peak"


def read_control_group_peaks(control_group_directory: Path) -> StepPeaks:
    return StepPeaks(
        memory_peak_bytes=int(
            (control_group_directory / MEMORY_PEAK_FILE_NAME).read_text(encoding="ascii")
        ),
        process_count_peak=int(
            (control_group_directory / PROCESS_COUNT_PEAK_FILE_NAME).read_text(encoding="ascii")
        ),
    )


def write_control_group_peaks(control_group_directory: Path, peaks_path: Path) -> None:
    peaks_path.write_bytes(render_step_peaks(read_control_group_peaks(control_group_directory)))
