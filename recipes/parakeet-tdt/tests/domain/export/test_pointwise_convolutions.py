from pathlib import Path

import pytest

from mostt_recipe_parakeet_tdt.domain.export.pointwise_convolutions import POINTWISE_CONVOLUTIONS
from mostt_recipe_parakeet_tdt.domain.export.tensor_fixtures import parse_name_list

FIXTURES_DIRECTORY = Path(__file__).resolve().parents[2] / "fixtures"
NEMO_TENSORS = parse_name_list((FIXTURES_DIRECTORY / "nemo-state-dict").read_bytes())
WEIGHT_SUFFIX = ".weight"
POINTWISE_CONVOLUTION_COUNT = 48


class TestThePointwiseConvolutions:
    class TestThePointwiseConvolutionsConstant:
        class TestWhenComparedWithTheWeightsOfNemosModel:
            @pytest.fixture
            def kernel_size_one_convolutions(self) -> list[str]:
                return [
                    tensor.name.removesuffix(WEIGHT_SUFFIX)
                    for tensor in NEMO_TENSORS
                    if tensor.name.endswith(WEIGHT_SUFFIX)
                    and len(tensor.shape) == 3
                    and tensor.shape[-1] == 1
                ]

            def test_should_name_every_convolution_of_kernel_size_one_and_nothing_else(
                self, kernel_size_one_convolutions: list[str]
            ) -> None:
                assert sorted(kernel_size_one_convolutions) == sorted(POINTWISE_CONVOLUTIONS)

        def test_should_name_two_per_layer_of_the_encoder(self) -> None:
            assert len(POINTWISE_CONVOLUTIONS) == POINTWISE_CONVOLUTION_COUNT
