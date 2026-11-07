from typing import Any

import torch
from omegaconf import open_dict

from mostt_model_builder.steps.bound_sources import find_bound_source
from mostt_recipe_parakeet_tdt.domain.export.pinned_sources import (
    NEMO_ARCHIVE,
    NEMO_REPOSITORY,
    ULTRA_REPOSITORY,
    ULTRA_TOKENIZER,
    ULTRA_WEIGHTS,
)
from mostt_recipe_parakeet_tdt.steps.fleurs_recordings import Recording
from mostt_recipe_parakeet_tdt.steps.nemo_model import (
    NemoModel,
    build_nemo_model,
    check_ultra_tokenizer,
    load_ultra_weights,
)

# torch and NeMo are installed in the export image alone, so mypy on the host knows nothing of
# their types.
type TorchTensor = Any

# onnx-asr stops after 10 tokens on one frame, its max_tokens_per_step, so the reference decodes
# under the same bound.
MAX_SYMBOLS_PER_FRAME = 10
GREEDY_STRATEGY = "greedy"


def prepare_reference_model() -> NemoModel:
    model = build_nemo_model(find_bound_source(NEMO_REPOSITORY, NEMO_ARCHIVE))
    load_ultra_weights(model, find_bound_source(ULTRA_REPOSITORY, ULTRA_WEIGHTS))
    check_ultra_tokenizer(model, find_bound_source(ULTRA_REPOSITORY, ULTRA_TOKENIZER))
    decoding = model.cfg.decoding
    with open_dict(decoding):
        decoding.strategy = GREEDY_STRATEGY
        decoding.greedy.max_symbols = MAX_SYMBOLS_PER_FRAME
    model.change_decoding_strategy(decoding)
    return model


def encode_recording(
    model: NemoModel, recording: Recording
) -> tuple[TorchTensor, TorchTensor, TorchTensor, TorchTensor]:
    samples = torch.from_numpy(recording.samples.copy())
    features, feature_lengths = model.preprocessor(
        input_signal=samples[None], length=torch.tensor([len(samples)])
    )
    encoded, encoded_lengths = model.encoder(audio_signal=features, length=feature_lengths)
    return features, feature_lengths, encoded, encoded_lengths
