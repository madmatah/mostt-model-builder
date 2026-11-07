from collections.abc import Mapping
from types import MappingProxyType

from mostt_recipe_parakeet_tdt.domain.variants.candidate_layout import (
    DECODER_JOINT_FILE_STEM,
    ENCODER_FILE_STEM,
    ONNX_FILE_EXTENSION,
    CandidateLayoutError,
)

QUANTIZATION_BY_VARIANT: Mapping[str, str | None] = MappingProxyType(
    {
        "float32": None,
        "float16": "fp16",
        "bfloat16": "bf16",
        "eight-bit-blocks": "int8",
    }
)


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
