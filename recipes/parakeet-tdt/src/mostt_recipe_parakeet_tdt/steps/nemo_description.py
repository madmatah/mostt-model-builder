import json
import tarfile
from collections.abc import Iterable, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

import torch
from nemo.collections.asr.models import ASRModel, EncDecRNNTBPEModel

from mostt_model_builder.domain.containers.container_steps import (
    OUTPUT_IN_THE_CONTAINER,
    TEMPORARY_DIRECTORY_IN_THE_CONTAINER,
)
from mostt_model_builder.steps.bound_sources import find_bound_source
from mostt_model_builder.steps.output_modes import finish_outputs
from mostt_recipe_parakeet_tdt.domain.export.nemo_configuration import (
    list_dataset_keys,
    prepare_tokenizer,
)
from mostt_recipe_parakeet_tdt.recipe import (
    NEMO_ARCHIVE,
    NEMO_REPOSITORY,
    NEMO_STATE_DICT_FILE_NAME,
    ULTRA_REPOSITORY,
    ULTRA_TENSORS_FILE_NAME,
    ULTRA_WEIGHTS,
)

# NeMo, and the OmegaConf configuration it builds from, are installed in the export image alone,
# so mypy on the host knows nothing of their types.
type NemoConfiguration = Any

TOKENIZER_DIRECTORY = Path(TEMPORARY_DIRECTORY_IN_THE_CONTAINER) / "nemo-tokenizer"
TORCH_DTYPE_PREFIX = "torch."
SAFETENSORS_HEADER_LENGTH_BYTES = 8
SAFETENSORS_METADATA_KEY = "__metadata__"
# The format bounds its header at 100 MB.
MAX_SAFETENSORS_HEADER_SIZE_BYTES = 100_000_000


class NemoDescriptionError(Exception):
    pass


def render_name_list(tensors: Iterable[tuple[str, str, Sequence[int]]]) -> bytes:
    return "".join(
        f"{name}\t{dtype}\t[{','.join(str(dimension) for dimension in shape)}]\n"
        for name, dtype, shape in tensors
    ).encode("utf-8")


def read_safetensors_tensors(weights_path: Path) -> list[tuple[str, str, list[int]]]:
    with weights_path.open("rb") as stream:
        header_size_bytes = int.from_bytes(stream.read(SAFETENSORS_HEADER_LENGTH_BYTES), "little")
        if header_size_bytes > MAX_SAFETENSORS_HEADER_SIZE_BYTES:
            raise NemoDescriptionError(f"{weights_path} has a header of {header_size_bytes} bytes")
        header = json.loads(stream.read(header_size_bytes))
    return [
        (name, entry["dtype"], entry["shape"])
        for name, entry in header.items()
        if name != SAFETENSORS_METADATA_KEY
    ]


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


def describe_nemo_state_dict(archive_path: Path) -> list[tuple[str, str, list[int]]]:
    configuration = ASRModel.restore_from(
        str(archive_path), return_config=True, map_location=torch.device("cpu")
    )
    prepare_configuration(configuration, archive_path)
    model = EncDecRNNTBPEModel(cfg=configuration, trainer=None)
    return [
        (name, str(tensor.dtype).removeprefix(TORCH_DTYPE_PREFIX), list(tensor.shape))
        for name, tensor in model.state_dict().items()
    ]


def main() -> None:
    output_directory = Path(OUTPUT_IN_THE_CONTAINER)
    ultra_tensors = read_safetensors_tensors(find_bound_source(ULTRA_REPOSITORY, ULTRA_WEIGHTS))
    (output_directory / ULTRA_TENSORS_FILE_NAME).write_bytes(render_name_list(ultra_tensors))
    nemo_tensors = describe_nemo_state_dict(find_bound_source(NEMO_REPOSITORY, NEMO_ARCHIVE))
    (output_directory / NEMO_STATE_DICT_FILE_NAME).write_bytes(render_name_list(nemo_tensors))
    finish_outputs(output_directory)


if __name__ == "__main__":
    main()
