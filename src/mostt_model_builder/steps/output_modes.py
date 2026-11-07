import os
import stat
from pathlib import Path

from mostt_model_builder.domain.containers.step_peaks import PEAKS_FILE_NAME
from mostt_model_builder.domain.containers.written_files import WRITTEN_FILE_MODE
from mostt_model_builder.steps.control_group_peaks import (
    CONTROL_GROUP_DIRECTORY,
    write_control_group_peaks,
)


def set_output_modes(output_directory: Path) -> None:
    for entry in output_directory.iterdir():
        descriptor = os.open(entry, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC)
        try:
            if stat.S_ISREG(os.fstat(descriptor).st_mode):
                os.fchmod(descriptor, WRITTEN_FILE_MODE)
        finally:
            os.close(descriptor)


def finish_outputs(
    output_directory: Path, control_group_directory: Path = CONTROL_GROUP_DIRECTORY
) -> None:
    write_control_group_peaks(control_group_directory, output_directory / PEAKS_FILE_NAME)
    set_output_modes(output_directory)
