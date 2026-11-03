import json
import os
import stat
from pathlib import Path

import pytest

from mostt_model_builder.steps.output_modes import finish_outputs


def write_control_group(directory: Path) -> Path:
    directory.mkdir()
    (directory / "memory.peak").write_text("22020096\n")
    (directory / "pids.peak").write_text("6\n")
    return directory


class TestTheOutputModes:
    class TestTheFinishOutputsFunction:
        class TestWhenALibraryLeftAnOutputReadableByItsOwnerAlone:
            @pytest.fixture
            def finished_outputs(self, tmp_path: Path) -> dict[str, int]:
                output = tmp_path / "output"
                output.mkdir()
                (output / "encoder-model.onnx.data").write_bytes(b"weights")
                (output / "encoder-model.onnx.data").chmod(0o600)
                finish_outputs(output, write_control_group(tmp_path / "cgroup"))
                return {
                    name: stat.S_IMODE(os.stat(output / name).st_mode)
                    for name in os.listdir(output)
                }

            def test_should_leave_every_output_in_mode_0640_the_peaks_included(
                self, finished_outputs: dict[str, int]
            ) -> None:
                assert finished_outputs == {
                    "encoder-model.onnx.data": 0o640,
                    "peaks.json": 0o640,
                }

        class TestWhenTheControlGroupGivesItsPeaks:
            @pytest.fixture
            def peaks(self, tmp_path: Path) -> object:
                output = tmp_path / "output"
                output.mkdir()
                finish_outputs(output, write_control_group(tmp_path / "cgroup"))
                return json.loads((output / "peaks.json").read_text())

            def test_should_write_them_into_the_peaks(self, peaks: object) -> None:
                assert peaks == {"memory_peak_bytes": 22_020_096, "process_count_peak": 6}
