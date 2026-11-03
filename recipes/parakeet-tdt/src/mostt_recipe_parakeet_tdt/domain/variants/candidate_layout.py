import json
from collections.abc import Mapping
from types import MappingProxyType

VOCABULARY_FILE_NAME = "vocab.txt"
CONFIGURATION_FILE_NAME = "config.json"
ONNX_ASR_MODEL_TYPE = "nemo-conformer-tdt"
# onnx-asr's defaults for this model type, 80 features, would pick the preprocessor of another
# model, so the configuration names all three; 10 tokens per step match NeMo's `max_symbols=10`
# in the reference transcription.
ONNX_ASR_CONFIGURATION = json.dumps(
    {
        "model_type": ONNX_ASR_MODEL_TYPE,
        "features_size": 128,
        "subsampling_factor": 8,
        "max_tokens_per_step": 10,
    }
).encode("utf-8")
QUANTIZATION_BY_VARIANT: Mapping[str, str | None] = MappingProxyType(
    {
        "float32": None,
        "float16": "fp16",
        "bfloat16": "bf16",
        "eight-bit-blocks": "int8",
    }
)
ENCODER_FILE_STEM = "encoder-model"
DECODER_JOINT_FILE_STEM = "decoder_joint-model"
ONNX_FILE_EXTENSION = ".onnx"


class CandidateLayoutError(ValueError):
    pass


def find_quantization(variant: str) -> str | None:
    if variant not in QUANTIZATION_BY_VARIANT:
        raise CandidateLayoutError(
            f"the variant {variant!a} is none of {', '.join(QUANTIZATION_BY_VARIANT)}"
        )
    return QUANTIZATION_BY_VARIANT[variant]


def name_encoder_file(variant: str) -> str:
    return name_onnx_file(ENCODER_FILE_STEM, variant)


def name_decoder_joint_file(variant: str) -> str:
    return name_onnx_file(DECODER_JOINT_FILE_STEM, variant)


def name_onnx_file(stem: str, variant: str) -> str:
    quantization = find_quantization(variant)
    if quantization is None:
        return f"{stem}{ONNX_FILE_EXTENSION}"
    return f"{stem}.{quantization}{ONNX_FILE_EXTENSION}"
