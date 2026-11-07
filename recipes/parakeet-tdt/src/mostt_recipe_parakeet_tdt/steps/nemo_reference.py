import json
import time
from pathlib import Path
from typing import Any

import torch
from omegaconf import open_dict
from safetensors.torch import save_file

from mostt_model_builder.domain.bench.run_lines import RUN_LINES_FILE_NAME
from mostt_model_builder.domain.containers.container_paths import (
    OUTPUT_IN_THE_CONTAINER,
    REQUEST_IN_THE_CONTAINER,
)
from mostt_model_builder.domain.contract.shard_request import (
    SHARD_REQUEST_FILE_NAME,
    parse_shard_request,
)
from mostt_model_builder.steps.bound_sources import find_bound_source
from mostt_model_builder.steps.output_modes import finish_outputs
from mostt_recipe_parakeet_tdt.domain.cpu_counts import REFERENCE_CPU_COUNT
from mostt_recipe_parakeet_tdt.domain.export.parity_checks import PARITY_LANGUAGE
from mostt_recipe_parakeet_tdt.domain.export.pinned_sources import (
    NEMO_ARCHIVE,
    NEMO_REPOSITORY,
    ULTRA_REPOSITORY,
    ULTRA_TOKENIZER,
    ULTRA_WEIGHTS,
)
from mostt_recipe_parakeet_tdt.domain.export.reference_dump import (
    REFERENCE_STEPS_FILE_NAME,
    REFERENCE_TENSORS_FILE_NAME,
    DumpedUtterance,
    JointStep,
    ReferenceSteps,
    name_encoder_outputs,
    name_features,
    name_joint_logits,
    render_reference_steps,
)
from mostt_recipe_parakeet_tdt.steps.fleurs_recordings import Recording, read_shard_recordings
from mostt_recipe_parakeet_tdt.steps.nemo_model import (
    NemoModel,
    build_nemo_model,
    check_ultra_tokenizer,
    load_ultra_weights,
)

# torch and NeMo are installed in the export image alone, so mypy on the host knows nothing of
# their types.
type TorchTensor = Any
type NemoHypotheses = Any

# onnx-asr stops after 10 tokens on one frame, its max_tokens_per_step, so the reference decodes
# under the same bound.
MAX_SYMBOLS_PER_FRAME = 10
GREEDY_STRATEGY = "greedy"
# The Transformers check compares the features, the encoder's outputs and the joint's first greedy
# steps of the first utterances of FLEURS fr with NeMo's.
DUMPED_UTTERANCE_COUNT = 5
JOINT_STEP_COUNT = 5
MILLISECONDS_PER_SECOND = 1000.0


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


def list_token_ids(hypotheses: NemoHypotheses) -> list[int]:
    if isinstance(hypotheses, tuple):
        hypotheses = hypotheses[0]
    sequence = hypotheses[0].y_sequence
    token_ids = sequence.tolist() if hasattr(sequence, "tolist") else sequence
    return [int(token_id) for token_id in token_ids]


def trace_joint_steps(
    model: NemoModel, encoded: TorchTensor, durations: list[int]
) -> list[tuple[int, list[int], TorchTensor]]:
    blank_id = model.decoder.blank_idx
    frames = encoded.transpose(1, 2)
    prediction, state = model.decoder.predict(None, None, add_sos=False, batch_size=1)
    steps: list[tuple[int, list[int], TorchTensor]] = []
    emitted: list[int] = []
    frame = 0
    symbols_on_frame = 0
    while len(steps) < JOINT_STEP_COUNT and frame < frames.shape[1]:
        logits = model.joint.joint(frames[:, frame : frame + 1, :], prediction)[0, 0, 0]
        steps.append((frame, list(emitted), logits.clone()))
        token = int(logits[: blank_id + 1].argmax())
        duration = durations[int(logits[blank_id + 1 :].argmax())]
        if token != blank_id:
            emitted.append(token)
            prediction, state = model.decoder.predict(
                torch.tensor([[token]]), state, add_sos=False, batch_size=1
            )
            symbols_on_frame += 1
        if duration > 0:
            frame += duration
            symbols_on_frame = 0
        elif token == blank_id or symbols_on_frame >= MAX_SYMBOLS_PER_FRAME:
            frame += 1
            symbols_on_frame = 0
    return steps


class ReferenceDump:
    def __init__(self, durations: list[int]) -> None:
        self.durations = durations
        self.tensors: dict[str, TorchTensor] = {}
        self.utterances: list[DumpedUtterance] = []

    def is_full(self) -> bool:
        return len(self.utterances) >= DUMPED_UTTERANCE_COUNT

    def add(
        self,
        model: NemoModel,
        recording: Recording,
        features: TorchTensor,
        encoded: TorchTensor,
        lengths: tuple[int, int],
    ) -> None:
        position = len(self.utterances)
        self.tensors[name_features(position)] = features[0].contiguous()
        self.tensors[name_encoder_outputs(position)] = encoded[0].contiguous()
        steps = trace_joint_steps(model, encoded, self.durations)
        for step_position, (_, _, logits) in enumerate(steps):
            self.tensors[name_joint_logits(position, step_position)] = logits.contiguous()
        self.utterances.append(
            DumpedUtterance(
                utterance_id=recording.utterance_id,
                feature_frames=lengths[0],
                encoded_frames=lengths[1],
                steps=tuple(JointStep(frame, tuple(prefix)) for frame, prefix, _ in steps),
            )
        )

    def write(self, model: NemoModel, output_directory: Path) -> None:
        save_file(self.tensors, str(output_directory / REFERENCE_TENSORS_FILE_NAME))
        reference_steps = ReferenceSteps(
            utterances=tuple(self.utterances),
            durations=tuple(self.durations),
            blank_id=int(model.decoder.blank_idx),
        )
        (output_directory / REFERENCE_STEPS_FILE_NAME).write_bytes(
            render_reference_steps(reference_steps)
        )


def transcribe_recording(
    model: NemoModel, recording: Recording
) -> tuple[dict[str, object], tuple[TorchTensor, TorchTensor, int, int]]:
    samples = torch.from_numpy(recording.samples.copy())
    start_s = time.perf_counter()
    features, feature_lengths = model.preprocessor(
        input_signal=samples[None], length=torch.tensor([len(samples)])
    )
    encoded, encoded_lengths = model.encoder(audio_signal=features, length=feature_lengths)
    hypotheses = model.decoding.rnnt_decoder_predictions_tensor(
        encoder_output=encoded, encoded_lengths=encoded_lengths, return_hypotheses=True
    )
    token_ids = list_token_ids(hypotheses)
    decoding_time_ms = (time.perf_counter() - start_s) * MILLISECONDS_PER_SECOND
    line = {
        "language": recording.language,
        "utterance_id": recording.utterance_id,
        "token_ids": token_ids,
        "decoding_time_ms": decoding_time_ms,
    }
    return line, (features, encoded, int(feature_lengths[0]), int(encoded_lengths[0]))


def run_reference(request_directory: Path, output_directory: Path) -> None:
    request = parse_shard_request((request_directory / SHARD_REQUEST_FILE_NAME).read_bytes())
    model = prepare_reference_model()
    durations = [int(duration) for duration in model.cfg.model_defaults.tdt_durations]
    dump = ReferenceDump(durations)
    recordings = read_shard_recordings(request.languages, request.shard, request.shard_count)
    with (output_directory / RUN_LINES_FILE_NAME).open("w", encoding="ascii") as lines:
        for recording in recordings:
            line, (features, encoded, feature_frames, encoded_frames) = transcribe_recording(
                model, recording
            )
            lines.write(json.dumps(line) + "\n")
            if recording.language == PARITY_LANGUAGE and not dump.is_full():
                dump.add(model, recording, features, encoded, (feature_frames, encoded_frames))
    dump.write(model, output_directory)


def main() -> None:
    torch.set_num_threads(REFERENCE_CPU_COUNT)
    torch.set_grad_enabled(False)
    output_directory = Path(OUTPUT_IN_THE_CONTAINER)
    run_reference(Path(REQUEST_IN_THE_CONTAINER), output_directory)
    finish_outputs(output_directory)


if __name__ == "__main__":
    main()
