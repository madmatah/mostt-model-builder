import re
from collections.abc import Mapping, Sequence

# Moondream publishes Parakeet Ultra under the names of Transformers' Parakeet; these rules,
# ported from parakeet-rs's export script, give each its NeMo name, in this order, since
# relative_k_proj holds k_proj.
WEIGHT_NAME_RULES = tuple(
    (re.compile(pattern), replacement)
    for pattern, replacement in (
        (r"^encoder\.subsampling\.layers\.", "encoder.pre_encode.conv."),
        (r"^encoder\.subsampling\.linear\.", "encoder.pre_encode.out."),
        (r"^encoder\.encode_positions\.", "encoder.pos_enc."),
        (r"^(encoder\.layers\.\d+\.conv)\.norm\.", r"\1.batch_norm."),
        (r"\.relative_k_proj\.", ".linear_pos."),
        (r"\.q_proj\.", ".linear_q."),
        (r"\.k_proj\.", ".linear_k."),
        (r"\.v_proj\.", ".linear_v."),
        (r"\.o_proj\.", ".linear_out."),
        (r"\.bias_([uv])$", r".pos_bias_\1"),
        (r"^decoder\.embedding\.", "decoder.prediction.embed."),
        (r"^decoder\.lstm\.", "decoder.prediction.dec_rnn.lstm."),
        (r"^encoder_projector\.", "joint.enc."),
        (r"^decoder\.decoder_projector\.", "joint.pred."),
        (r"^joint\.head\.", "joint.joint_net.2."),
    )
)
# Moondream's voice activity head serves its own runtime alone, and NeMo's model has no place for
# it.
DROPPED_WEIGHT_PREFIX = "vad_head."
# NeMo computes the filter bank and the window of its preprocessor when it builds the model, and
# Moondream's file, from Transformers, holds neither.
COMPUTED_BUFFERS = frozenset({"preprocessor.featurizer.fb", "preprocessor.featurizer.window"})

type TensorShapes = Mapping[str, Sequence[int]]


class WeightMappingError(ValueError):
    pass


def map_weight_name(ultra_name: str) -> str:
    nemo_name = ultra_name
    for pattern, replacement in WEIGHT_NAME_RULES:
        nemo_name = pattern.sub(replacement, nemo_name)
    return nemo_name


def is_loaded_weight(ultra_name: str) -> bool:
    return not ultra_name.startswith(DROPPED_WEIGHT_PREFIX)


def map_weight_names(ultra_shapes: TensorShapes, nemo_shapes: TensorShapes) -> dict[str, str]:
    nemo_names: dict[str, str] = {}
    loaded_names: set[str] = set()
    for ultra_name, ultra_shape in ultra_shapes.items():
        if not is_loaded_weight(ultra_name):
            continue
        nemo_name = map_weight_name(ultra_name)
        nemo_shape = nemo_shapes.get(nemo_name)
        if nemo_shape is None:
            raise WeightMappingError(
                f"Moondream's {ultra_name} maps to {nemo_name}, which NeMo's model does not hold"
            )
        if tuple(nemo_shape) != tuple(ultra_shape):
            raise WeightMappingError(
                f"Moondream's {ultra_name} has the shape {list(ultra_shape)}, and {nemo_name} "
                f"the shape {list(nemo_shape)}"
            )
        if nemo_name in loaded_names:
            raise WeightMappingError(f"two of Moondream's weights map to {nemo_name}")
        loaded_names.add(nemo_name)
        nemo_names[ultra_name] = nemo_name
    left_names = set(nemo_shapes) - loaded_names
    if left_names != COMPUTED_BUFFERS:
        raise WeightMappingError(
            f"Moondream's weights leave {sorted(left_names)} of NeMo's model to its "
            f"construction, not exactly the buffers it computes, {sorted(COMPUTED_BUFFERS)}"
        )
    return nemo_names
