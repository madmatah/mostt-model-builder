import json
import re
from collections.abc import Callable
from functools import partial

import pytest

from mostt_recipe_parakeet_tdt.domain.variants.candidate_layout import (
    CONFIGURATION_FILE_NAME,
    DECODER_JOINT_FILE_NAME,
    ENCODER_DATA_FILE_NAME,
    ENCODER_FILE_NAME,
    ONNX_ASR_CONFIGURATION,
    ONNX_ASR_MODEL_TYPE,
    VOCABULARY_FILE_NAME,
    CandidateLayoutError,
    read_quantization,
)
from mostt_recipe_parakeet_tdt.domain.variants.quantizations import (
    QUANTIZATION_BY_VARIANT,
    find_quantization,
    name_decoder_joint_file,
    name_encoder_file,
)

REFUSED_FILE_LISTS = {
    "no encoder": (
        ("decoder_joint-model.onnx", "vocab.txt"),
        "the files ['decoder_joint-model.onnx', 'vocab.txt'] hold 0 encoders, where onnx-asr "
        "loads one",
    ),
    "two encoders": (
        ("encoder-model.onnx", "encoder-model.int8.onnx"),
        "hold 2 encoders, where onnx-asr loads one",
    ),
    "an encoder whose quantization holds a dot": (
        ("encoder-model.int8.extra.onnx",),
        "hold 0 encoders",
    ),
}
EXPECTED_CONFIGURATION = (
    b'{"model_type": "nemo-conformer-tdt", "features_size": 128, "subsampling_factor": 8, '
    b'"max_tokens_per_step": 10}'
)
EXPECTED_CONFIGURATION_ENTRIES = {
    "model_type": "nemo-conformer-tdt",
    "features_size": 128,
    "subsampling_factor": 8,
    "max_tokens_per_step": 10,
}


class TestTheCandidateLayout:
    class TestTheReadQuantizationFunction:
        @pytest.mark.parametrize(
            "variant", list(QUANTIZATION_BY_VARIANT), ids=list(QUANTIZATION_BY_VARIANT)
        )
        class TestWhenGivenTheFilesOfAVariant:
            @pytest.fixture
            def quantization(self, variant: str) -> str | None:
                return read_quantization(
                    (
                        name_encoder_file(variant),
                        name_decoder_joint_file(variant),
                        VOCABULARY_FILE_NAME,
                        CONFIGURATION_FILE_NAME,
                    )
                )

            def test_should_read_the_quantization_that_names_its_encoder(
                self, quantization: str | None, variant: str
            ) -> None:
                assert quantization == find_quantization(variant)

        class TestWhenGivenTheExportedEncoderWithItsExternalData:
            @pytest.fixture
            def quantization(self) -> str | None:
                return read_quantization(
                    (ENCODER_FILE_NAME, ENCODER_DATA_FILE_NAME, DECODER_JOINT_FILE_NAME)
                )

            def test_should_read_no_quantization(self, quantization: str | None) -> None:
                assert quantization is None

        @pytest.mark.parametrize(
            ("file_names", "reason"),
            list(REFUSED_FILE_LISTS.values()),
            ids=list(REFUSED_FILE_LISTS.keys()),
        )
        class TestWhenTheFilesHoldOtherThanOneEncoder:
            @pytest.fixture
            def read_act(self, file_names: tuple[str, ...]) -> Callable[[], str | None]:
                return partial(read_quantization, file_names)

            def test_should_refuse_them_since_onnx_asr_loads_one(
                self, read_act: Callable[[], str | None], reason: str
            ) -> None:
                with pytest.raises(CandidateLayoutError, match=re.escape(reason)):
                    read_act()

    class TestTheOnnxAsrConfigurationConstant:
        def test_should_hold_exactly_the_bytes_onnx_asr_reads(self) -> None:
            assert ONNX_ASR_CONFIGURATION == EXPECTED_CONFIGURATION

        def test_should_set_the_features_subsampling_and_tokens_per_step_of_the_model(
            self,
        ) -> None:
            assert json.loads(ONNX_ASR_CONFIGURATION) == EXPECTED_CONFIGURATION_ENTRIES

        def test_should_name_the_model_type_of_the_layout(self) -> None:
            assert json.loads(ONNX_ASR_CONFIGURATION)["model_type"] == ONNX_ASR_MODEL_TYPE

    class TestTheFileNameConstants:
        def test_should_name_the_exported_files_as_onnx_asr_loads_them_in_float32(self) -> None:
            assert (ENCODER_FILE_NAME, ENCODER_DATA_FILE_NAME, DECODER_JOINT_FILE_NAME) == (
                "encoder-model.onnx",
                "encoder-model.onnx.data",
                "decoder_joint-model.onnx",
            )

        def test_should_name_the_vocabulary_as_onnx_asr_reads_it(self) -> None:
            assert VOCABULARY_FILE_NAME == "vocab.txt"

        def test_should_name_the_configuration_as_onnx_asr_reads_it(self) -> None:
            assert CONFIGURATION_FILE_NAME == "config.json"
