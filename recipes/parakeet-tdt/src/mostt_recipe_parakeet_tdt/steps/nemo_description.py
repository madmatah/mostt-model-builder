import json
from pathlib import Path

from mostt_model_builder.domain.containers.container_steps import OUTPUT_IN_THE_CONTAINER
from mostt_model_builder.steps.bound_sources import find_bound_source
from mostt_model_builder.steps.output_modes import finish_outputs
from mostt_recipe_parakeet_tdt.domain.export.tensor_fixtures import (
    TensorDescription,
    render_name_list,
)
from mostt_recipe_parakeet_tdt.recipe import (
    NEMO_ARCHIVE,
    NEMO_REPOSITORY,
    NEMO_STATE_DICT_FILE_NAME,
    ULTRA_REPOSITORY,
    ULTRA_TENSORS_FILE_NAME,
    ULTRA_WEIGHTS,
)
from mostt_recipe_parakeet_tdt.steps.nemo_model import build_nemo_model

TORCH_DTYPE_PREFIX = "torch."
SAFETENSORS_HEADER_LENGTH_BYTES = 8
SAFETENSORS_METADATA_KEY = "__metadata__"
# The format bounds its header at 100 MB.
MAX_SAFETENSORS_HEADER_SIZE_BYTES = 100_000_000


class NemoDescriptionError(Exception):
    pass


def read_safetensors_tensors(weights_path: Path) -> list[TensorDescription]:
    with weights_path.open("rb") as stream:
        header_size_bytes = int.from_bytes(stream.read(SAFETENSORS_HEADER_LENGTH_BYTES), "little")
        if header_size_bytes > MAX_SAFETENSORS_HEADER_SIZE_BYTES:
            raise NemoDescriptionError(f"{weights_path} has a header of {header_size_bytes} bytes")
        header = json.loads(stream.read(header_size_bytes))
    return [
        TensorDescription(name, entry["dtype"], tuple(entry["shape"]))
        for name, entry in header.items()
        if name != SAFETENSORS_METADATA_KEY
    ]


def describe_nemo_state_dict(archive_path: Path) -> list[TensorDescription]:
    model = build_nemo_model(archive_path)
    return [
        TensorDescription(
            name, str(tensor.dtype).removeprefix(TORCH_DTYPE_PREFIX), tuple(tensor.shape)
        )
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
