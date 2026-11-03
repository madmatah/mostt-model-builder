import re
from collections.abc import Callable, Mapping, Sequence
from functools import partial

import pytest

from mostt_model_builder.domain.formats.external_data import (
    ExternalDataError,
    judge_external_data,
)
from mostt_model_builder.domain.onnx.inspection_report import ExternalTensor

DATA_NAME = "encoder.onnx.data"
ONNX_NAME = "encoder.onnx"
QUANTIZED_ONNX_NAME = "encoder-int8.onnx"
DECODER_ONNX_NAME = "decoder.onnx"
FIRST_TENSOR = ExternalTensor(DATA_NAME, 0, 4096)
SECOND_TENSOR = ExternalTensor(DATA_NAME, 4096, 1024)
THIRD_TENSOR = ExternalTensor(DATA_NAME, 5120, 2048)
DATA_SIZE_BYTES = 7168
DECODER_TENSOR = ExternalTensor("decoder.onnx.data", 0, 512)
WHERE = "the external data encoder.onnx.data"

TensorsByOnnxFile = Mapping[str, Sequence[ExternalTensor]]

ACCEPTED_LAYOUTS: dict[str, tuple[int, TensorsByOnnxFile]] = {
    "tensors in the order of the file": (
        DATA_SIZE_BYTES,
        {ONNX_NAME: (FIRST_TENSOR, SECOND_TENSOR, THIRD_TENSOR)},
    ),
    "tensors listed in another order": (
        DATA_SIZE_BYTES,
        {ONNX_NAME: (THIRD_TENSOR, FIRST_TENSOR, SECOND_TENSOR)},
    ),
    "a single tensor that fills the file": (4096, {ONNX_NAME: (FIRST_TENSOR,)}),
    "another onnx file that reads only its own data": (
        DATA_SIZE_BYTES,
        {
            ONNX_NAME: (FIRST_TENSOR, SECOND_TENSOR, THIRD_TENSOR),
            DECODER_ONNX_NAME: (DECODER_TENSOR,),
        },
    ),
    "its onnx file reading another data file too": (
        4096,
        {ONNX_NAME: (FIRST_TENSOR, ExternalTensor("other.onnx.data", 4096, 1024))},
    ),
}
REFUSED_LAYOUTS: dict[str, tuple[int, TensorsByOnnxFile, str]] = {
    "a tensor beyond the end of the file": (
        7000,
        {ONNX_NAME: (FIRST_TENSOR, SECOND_TENSOR, THIRD_TENSOR)},
        f"{WHERE} holds 7000 bytes, and a tensor reads up to byte 7168",
    ),
    "two tensors that overlap": (
        5024,
        {ONNX_NAME: (FIRST_TENSOR, ExternalTensor(DATA_NAME, 4000, 1024))},
        f"{WHERE} has two tensors that overlap, at byte 4000",
    ),
    "a tensor listed twice": (
        4096,
        {ONNX_NAME: (FIRST_TENSOR, FIRST_TENSOR)},
        f"{WHERE} has two tensors that overlap, at byte 0",
    ),
    "a gap between two tensors": (
        5124,
        {ONNX_NAME: (FIRST_TENSOR, ExternalTensor(DATA_NAME, 4100, 1024))},
        f"{WHERE} has a gap of 4 bytes at byte 4096",
    ),
    "a gap before the first tensor": (
        4112,
        {ONNX_NAME: (ExternalTensor(DATA_NAME, 16, 4096),)},
        f"{WHERE} has a gap of 16 bytes at byte 0",
    ),
    "padding after the last tensor": (
        DATA_SIZE_BYTES + 32,
        {ONNX_NAME: (FIRST_TENSOR, SECOND_TENSOR, THIRD_TENSOR)},
        f"{WHERE} holds 32 bytes after its last tensor",
    ),
    "a file that no tensor of its onnx file names": (
        DATA_SIZE_BYTES,
        {ONNX_NAME: (DECODER_TENSOR,), DECODER_ONNX_NAME: (DECODER_TENSOR,)},
        f"{WHERE} holds no tensor of encoder.onnx",
    ),
    "a file whose onnx file is not inspected": (
        DATA_SIZE_BYTES,
        {DECODER_ONNX_NAME: (DECODER_TENSOR,)},
        f"{WHERE} holds no tensor of encoder.onnx",
    ),
    "a file that the onnx file of another variant names": (
        DATA_SIZE_BYTES,
        {
            ONNX_NAME: (FIRST_TENSOR, SECOND_TENSOR, THIRD_TENSOR),
            QUANTIZED_ONNX_NAME: (FIRST_TENSOR,),
        },
        f"{WHERE} is named by encoder-int8.onnx, not only by encoder.onnx, the onnx file it "
        "belongs to",
    ),
}


class TestTheExternalData:
    class TestTheJudgeExternalDataFunction:
        @pytest.mark.parametrize(
            ("size_bytes", "tensors_by_onnx_file"),
            list(ACCEPTED_LAYOUTS.values()),
            ids=list(ACCEPTED_LAYOUTS.keys()),
        )
        class TestWhenItsTensorsLieEndToEndFromItsStartToItsEnd:
            @pytest.fixture
            def judge_act(
                self, size_bytes: int, tensors_by_onnx_file: TensorsByOnnxFile
            ) -> Callable[[], object]:
                return partial(
                    judge_external_data, DATA_NAME, size_bytes, ONNX_NAME, tensors_by_onnx_file
                )

            def test_should_accept_it(self, judge_act: Callable[[], object]) -> None:
                assert judge_act() is None

        @pytest.mark.parametrize(
            ("size_bytes", "tensors_by_onnx_file", "reason"),
            list(REFUSED_LAYOUTS.values()),
            ids=list(REFUSED_LAYOUTS.keys()),
        )
        class TestWhenItHoldsBytesThatNoTensorReadsOrThatAnotherFileReads:
            @pytest.fixture
            def judge_act(
                self, size_bytes: int, tensors_by_onnx_file: TensorsByOnnxFile
            ) -> Callable[[], object]:
                return partial(
                    judge_external_data, DATA_NAME, size_bytes, ONNX_NAME, tensors_by_onnx_file
                )

            def test_should_refuse_it_since_the_exporter_lays_tensors_end_to_end_and_say_why(
                self, judge_act: Callable[[], object], reason: str
            ) -> None:
                with pytest.raises(ExternalDataError, match=re.escape(reason)):
                    judge_act()
