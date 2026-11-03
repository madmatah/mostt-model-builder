import json
import tarfile
from pathlib import Path, PurePosixPath
from typing import Any

import numpy
import torch
from nemo.collections.asr.models import ASRModel, EncDecRNNTBPEModel
from safetensors import safe_open

from mostt_model_builder.domain.containers.container_steps import (
    TEMPORARY_DIRECTORY_IN_THE_CONTAINER,
)
from mostt_recipe_parakeet_tdt.domain.export.nemo_configuration import (
    list_dataset_keys,
    prepare_tokenizer,
)
from mostt_recipe_parakeet_tdt.domain.export.weight_names import map_weight_names

# NeMo, torch, NumPy and the OmegaConf configuration NeMo builds from are installed in the export
# image alone, so mypy on the host knows nothing of their types.
type NemoConfiguration = Any
type NemoModel = Any
type NumpyArray = Any

TOKENIZER_DIRECTORY = Path(TEMPORARY_DIRECTORY_IN_THE_CONTAINER) / "nemo-tokenizer"
SAFETENSORS_FRAMEWORK = "np"


class NemoModelError(Exception):
    pass


def prepare_configuration(configuration: NemoConfiguration, archive_path: Path) -> None:
    with tarfile.open(archive_path, "r:") as archive:
        preparation = prepare_tokenizer(
            configuration.tokenizer, archive.getnames(), PurePosixPath(TOKENIZER_DIRECTORY)
        )
        TOKENIZER_DIRECTORY.mkdir()
        archive.extractall(
            TOKENIZER_DIRECTORY,
            members=[archive.getmember(name) for name in preparation.archive_members],
            filter="data",
        )
    for key in list_dataset_keys(configuration.keys()):
        configuration[key] = None
    for key, setting in preparation.tokenizer_settings.items():
        configuration.tokenizer[key] = setting


def build_nemo_model(archive_path: Path) -> NemoModel:
    configuration = ASRModel.restore_from(
        str(archive_path), return_config=True, map_location=torch.device("cpu")
    )
    prepare_configuration(configuration, archive_path)
    model = EncDecRNNTBPEModel(cfg=configuration, trainer=None)
    model.eval()
    return model


def widen_to_float32(tensor: NumpyArray) -> NumpyArray:
    if tensor.dtype == numpy.float16:
        return numpy.array(tensor, dtype=numpy.float32)
    return numpy.array(tensor)


# Moondream stores most of its weights in float16; they are widened to the float32 of NeMo's model,
# as parakeet-rs's export widens them, so that the export and its reference read the same values.
# Each is copied into the tensor of the model that its state shares, since a state of new tensors
# would hold a second float32 copy of every weight until it loads.
def load_ultra_weights(model: NemoModel, weights_path: Path) -> None:
    state = model.state_dict()
    nemo_shapes = {name: tuple(tensor.shape) for name, tensor in state.items()}
    with safe_open(str(weights_path), framework=SAFETENSORS_FRAMEWORK) as weights:
        ultra_names = list(weights.keys())
        ultra_shapes = {name: tuple(weights.get_slice(name).get_shape()) for name in ultra_names}
        for ultra_name, nemo_name in map_weight_names(ultra_shapes, nemo_shapes).items():
            state[nemo_name].copy_(
                torch.from_numpy(widen_to_float32(weights.get_tensor(ultra_name)))
            )


def list_tokenizer_pieces(tokenizer_path: Path) -> list[str]:
    vocabulary = json.loads(tokenizer_path.read_text(encoding="utf-8"))["model"]["vocab"]
    if isinstance(vocabulary, list):
        return [str(entry[0]) for entry in vocabulary]
    return sorted(vocabulary, key=vocabulary.get)


# The vocabulary that the export writes comes from NeMo's tokenizer, built from v3's archive, so
# Moondream's tokenizer must begin with the same pieces, as parakeet-rs's export requires.
def check_ultra_tokenizer(model: NemoModel, tokenizer_path: Path) -> None:
    nemo_pieces = list(model.tokenizer.vocab)
    if list_tokenizer_pieces(tokenizer_path)[: len(nemo_pieces)] != nemo_pieces:
        raise NemoModelError(
            f"{tokenizer_path.name} does not begin with the {len(nemo_pieces)} pieces of the "
            "vocabulary of NeMo's model"
        )
