import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import pytest

from mostt_recipe_parakeet_tdt.domain.export.tensor_fixtures import parse_name_list
from mostt_recipe_parakeet_tdt.domain.export.weight_names import (
    COMPUTED_BUFFERS,
    WeightMappingError,
    is_loaded_weight,
    map_weight_name,
    map_weight_names,
)

FIXTURES_DIRECTORY = Path(__file__).resolve().parents[2] / "fixtures"
ULTRA_SHAPES = {
    tensor.name: tensor.shape
    for tensor in parse_name_list((FIXTURES_DIRECTORY / "moondream-tensors").read_bytes())
}
NEMO_SHAPES = {
    tensor.name: tensor.shape
    for tensor in parse_name_list((FIXTURES_DIRECTORY / "nemo-state-dict").read_bytes())
}
# Moondream's file holds 729 tensors, of which its voice activity head holds 6.
LOADED_WEIGHT_COUNT = 723
VOICE_ACTIVITY_HEAD_COUNT = 6
NAME_TABLE = {
    "the subsampling's convolutions": (
        "encoder.subsampling.layers.0.weight",
        "encoder.pre_encode.conv.0.weight",
    ),
    "the subsampling's projection": (
        "encoder.subsampling.linear.bias",
        "encoder.pre_encode.out.bias",
    ),
    "the positional encoding": ("encoder.encode_positions.pe", "encoder.pos_enc.pe"),
    "a convolution module's norm": (
        "encoder.layers.3.conv.norm.running_mean",
        "encoder.layers.3.conv.batch_norm.running_mean",
    ),
    "the relative positions' projection, before the key's": (
        "encoder.layers.0.self_attn.relative_k_proj.weight",
        "encoder.layers.0.self_attn.linear_pos.weight",
    ),
    "the key's projection": (
        "encoder.layers.0.self_attn.k_proj.weight",
        "encoder.layers.0.self_attn.linear_k.weight",
    ),
    "a positional bias": (
        "encoder.layers.0.self_attn.bias_u",
        "encoder.layers.0.self_attn.pos_bias_u",
    ),
    "the prediction network's embedding": (
        "decoder.embedding.weight",
        "decoder.prediction.embed.weight",
    ),
    "the prediction network's LSTM": (
        "decoder.lstm.weight_ih_l0",
        "decoder.prediction.dec_rnn.lstm.weight_ih_l0",
    ),
    "the joint's projection of the encoder": (
        "encoder_projector.weight",
        "joint.enc.weight",
    ),
    "the joint's head": ("joint.head.weight", "joint.joint_net.2.weight"),
}
SMALL_ULTRA_SHAPES = {"joint.head.weight": (4, 2), "vad_head.weight": (1, 2)}
SMALL_NEMO_SHAPES = {
    "joint.joint_net.2.weight": (4, 2),
    "preprocessor.featurizer.fb": (1, 2, 3),
    "preprocessor.featurizer.window": (4,),
}
REFUSED_MAPPINGS: dict[
    str, tuple[Mapping[str, Sequence[int]], Mapping[str, Sequence[int]], str]
] = {
    "a weight that NeMo's model does not hold": (
        {**SMALL_ULTRA_SHAPES, "joint.tail.weight": (4, 2)},
        SMALL_NEMO_SHAPES,
        "Moondream's joint.tail.weight maps to joint.tail.weight, which NeMo's model does not hold",
    ),
    "a weight of another shape": (
        {"joint.head.weight": (2, 4)},
        SMALL_NEMO_SHAPES,
        "Moondream's joint.head.weight has the shape [2, 4], and joint.joint_net.2.weight the "
        "shape [4, 2]",
    ),
    "two weights onto one": (
        {**SMALL_ULTRA_SHAPES, "joint.joint_net.2.weight": (4, 2)},
        SMALL_NEMO_SHAPES,
        "two of Moondream's weights map to joint.joint_net.2.weight",
    ),
    "a parameter of NeMo's model that no weight covers": (
        SMALL_ULTRA_SHAPES,
        {**SMALL_NEMO_SHAPES, "joint.pred.weight": (4, 4)},
        "Moondream's weights leave ['joint.pred.weight', 'preprocessor.featurizer.fb', "
        "'preprocessor.featurizer.window'] of NeMo's model to its construction",
    ),
    "a computed buffer that a weight replaces": (
        {**SMALL_ULTRA_SHAPES, "preprocessor.featurizer.window": (4,)},
        SMALL_NEMO_SHAPES,
        "Moondream's weights leave ['preprocessor.featurizer.fb'] of NeMo's model",
    ),
}


@dataclass(frozen=True)
class MappingCheck:
    checked_count: int
    mismatched_names: list[str]


class TestTheWeightNames:
    class TestTheMapWeightNameFunction:
        class TestWhenEveryLoadedWeightOfMoondreamIsMapped:
            @pytest.fixture
            def mapping_check(self) -> MappingCheck:
                loaded_shapes = {
                    ultra_name: ultra_shape
                    for ultra_name, ultra_shape in ULTRA_SHAPES.items()
                    if is_loaded_weight(ultra_name)
                }
                return MappingCheck(
                    len(loaded_shapes),
                    [
                        ultra_name
                        for ultra_name, ultra_shape in loaded_shapes.items()
                        if NEMO_SHAPES.get(map_weight_name(ultra_name)) != ultra_shape
                    ],
                )

            def test_should_give_each_a_nemo_name_of_the_same_shape(
                self, mapping_check: MappingCheck
            ) -> None:
                assert mapping_check.mismatched_names == []
                assert mapping_check.checked_count == LOADED_WEIGHT_COUNT

        class TestWhenNemosNamesAreCompared:
            @pytest.fixture
            def left_names(self) -> set[str]:
                return set(NEMO_SHAPES) - {
                    map_weight_name(ultra_name)
                    for ultra_name in ULTRA_SHAPES
                    if is_loaded_weight(ultra_name)
                }

            def test_should_leave_exactly_the_two_buffers_that_nemo_computes(
                self, left_names: set[str]
            ) -> None:
                assert left_names == COMPUTED_BUFFERS

        @pytest.mark.parametrize(
            ("ultra_name", "nemo_name"), list(NAME_TABLE.values()), ids=list(NAME_TABLE.keys())
        )
        class TestWhenGivenANameOfEachRule:
            @pytest.fixture
            def mapped_name(self, ultra_name: str) -> str:
                return map_weight_name(ultra_name)

            def test_should_give_its_name_in_nemo(self, mapped_name: str, nemo_name: str) -> None:
                assert mapped_name == nemo_name

    class TestTheIsLoadedWeightFunction:
        class TestWhenEveryWeightOfMoondreamIsJudged:
            @pytest.fixture
            def dropped_names(self) -> list[str]:
                return [name for name in ULTRA_SHAPES if not is_loaded_weight(name)]

            def test_should_drop_the_voice_activity_head_alone(
                self, dropped_names: list[str]
            ) -> None:
                assert len(dropped_names) == VOICE_ACTIVITY_HEAD_COUNT
                assert [name for name in dropped_names if not name.startswith("vad_head.")] == []

    class TestTheMapWeightNamesFunction:
        class TestWhenGivenTheNamesThatDescribeListed:
            @pytest.fixture
            def nemo_names(self) -> dict[str, str]:
                return map_weight_names(ULTRA_SHAPES, NEMO_SHAPES)

            def test_should_load_every_weight_but_the_voice_activity_heads(
                self, nemo_names: dict[str, str]
            ) -> None:
                assert len(nemo_names) == LOADED_WEIGHT_COUNT

        @pytest.mark.parametrize(
            ("ultra_shapes", "nemo_shapes", "reason"),
            list(REFUSED_MAPPINGS.values()),
            ids=list(REFUSED_MAPPINGS.keys()),
        )
        class TestWhenTheMappingBreaksARule:
            @pytest.fixture
            def map_act(
                self,
                ultra_shapes: Mapping[str, Sequence[int]],
                nemo_shapes: Mapping[str, Sequence[int]],
            ) -> Callable[[], dict[str, str]]:
                return partial(map_weight_names, ultra_shapes, nemo_shapes)

            def test_should_refuse_it_and_say_why(
                self, map_act: Callable[[], dict[str, str]], reason: str
            ) -> None:
                with pytest.raises(WeightMappingError, match=re.escape(reason)):
                    map_act()
