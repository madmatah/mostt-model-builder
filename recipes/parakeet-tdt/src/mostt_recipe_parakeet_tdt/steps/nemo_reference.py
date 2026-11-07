import json
import time
from pathlib import Path
from typing import Any

import torch

from mostt_model_builder.domain.bench.run_lines import RUN_LINES_FILE_NAME
from mostt_model_builder.domain.containers.container_paths import (
    OUTPUT_IN_THE_CONTAINER,
    REQUEST_IN_THE_CONTAINER,
)
from mostt_model_builder.domain.contract.shard_request import (
    SHARD_REQUEST_FILE_NAME,
    parse_shard_request,
)
from mostt_model_builder.steps.output_modes import finish_outputs
from mostt_recipe_parakeet_tdt.domain.cpu_counts import REFERENCE_CPU_COUNT
from mostt_recipe_parakeet_tdt.steps.fleurs_recordings import Recording, read_shard_recordings
from mostt_recipe_parakeet_tdt.steps.nemo_model import NemoModel
from mostt_recipe_parakeet_tdt.steps.nemo_reference_model import (
    encode_recording,
    prepare_reference_model,
)

# NeMo is installed in the export image alone, so mypy on the host knows nothing of its types.
type NemoHypotheses = Any

MILLISECONDS_PER_SECOND = 1000.0


def list_token_ids(hypotheses: NemoHypotheses) -> list[int]:
    if isinstance(hypotheses, tuple):
        hypotheses = hypotheses[0]
    sequence = hypotheses[0].y_sequence
    token_ids = sequence.tolist() if hasattr(sequence, "tolist") else sequence
    return [int(token_id) for token_id in token_ids]


def transcribe_recording(model: NemoModel, recording: Recording) -> dict[str, object]:
    start_s = time.perf_counter()
    _, _, encoded, encoded_lengths = encode_recording(model, recording)
    hypotheses = model.decoding.rnnt_decoder_predictions_tensor(
        encoder_output=encoded, encoded_lengths=encoded_lengths, return_hypotheses=True
    )
    token_ids = list_token_ids(hypotheses)
    decoding_time_ms = (time.perf_counter() - start_s) * MILLISECONDS_PER_SECOND
    return {
        "language": recording.language,
        "utterance_id": recording.utterance_id,
        "token_ids": token_ids,
        "decoding_time_ms": decoding_time_ms,
    }


def run_reference(request_directory: Path, output_directory: Path) -> None:
    request = parse_shard_request((request_directory / SHARD_REQUEST_FILE_NAME).read_bytes())
    model = prepare_reference_model()
    recordings = read_shard_recordings(request.languages, request.shard, request.shard_count)
    with (output_directory / RUN_LINES_FILE_NAME).open("w", encoding="ascii") as lines:
        for recording in recordings:
            lines.write(json.dumps(transcribe_recording(model, recording)) + "\n")


def main() -> None:
    torch.set_num_threads(REFERENCE_CPU_COUNT)
    torch.set_grad_enabled(False)
    output_directory = Path(OUTPUT_IN_THE_CONTAINER)
    run_reference(Path(REQUEST_IN_THE_CONTAINER), output_directory)
    finish_outputs(output_directory)


if __name__ == "__main__":
    main()
