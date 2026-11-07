import json
import re
from collections.abc import Sequence

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
ENCODER_FILE_STEM = "encoder-model"
DECODER_JOINT_FILE_STEM = "decoder_joint-model"
ONNX_FILE_EXTENSION = ".onnx"
ENCODER_FILE_NAME = f"{ENCODER_FILE_STEM}{ONNX_FILE_EXTENSION}"
ENCODER_DATA_FILE_NAME = f"{ENCODER_FILE_NAME}.data"
DECODER_JOINT_FILE_NAME = f"{DECODER_JOINT_FILE_STEM}{ONNX_FILE_EXTENSION}"
# onnx-asr names a quantized file by its quantization between its stem and its extension.
ENCODER_FILE_GRAMMAR = re.compile(
    re.escape(ENCODER_FILE_STEM)
    + r"(\.(?P<quantization>[a-z0-9]{1,16}))?"
    + re.escape(ONNX_FILE_EXTENSION)
)


class CandidateLayoutError(ValueError):
    pass


def read_quantization(file_names: Sequence[str]) -> str | None:
    matches = [
        match
        for match in (ENCODER_FILE_GRAMMAR.fullmatch(name) for name in file_names)
        if match is not None
    ]
    if len(matches) != 1:
        raise CandidateLayoutError(
            f"the files {list(file_names)!a} hold {len(matches)} encoders, where onnx-asr loads one"
        )
    return matches[0].group("quantization")
