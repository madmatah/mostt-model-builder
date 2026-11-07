from pathlib import Path
from typing import Any

import torch
from safetensors.torch import save_file

from mostt_model_builder.domain.containers.container_paths import OUTPUT_IN_THE_CONTAINER
from mostt_model_builder.steps.output_modes import finish_outputs
from mostt_recipe_parakeet_tdt.domain.cpu_counts import REFERENCE_CPU_COUNT
from mostt_recipe_parakeet_tdt.domain.export.parity_checks import PARITY_LANGUAGE
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
from mostt_recipe_parakeet_tdt.steps.fleurs_recordings import (
    Recording,
    list_utterance_ids,
    read_language_recordings,
)
from mostt_recipe_parakeet_tdt.steps.nemo_model import NemoModel
from mostt_recipe_parakeet_tdt.steps.nemo_reference_model import (
    MAX_SYMBOLS_PER_FRAME,
    encode_recording,
    prepare_reference_model,
)

# torch is installed in the export image alone, so mypy on the host knows nothing of its types.
type TorchTensor = Any

# The Transformers check compares the features, the encoder's outputs and the joint's first greedy
# steps of the first utterances of FLEURS fr with NeMo's.
DUMPED_UTTERANCE_COUNT = 5
JOINT_STEP_COUNT = 5


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

    def add(self, model: NemoModel, recording: Recording) -> None:
        features, feature_lengths, encoded, encoded_lengths = encode_recording(model, recording)
        position = len(self.utterances)
        self.tensors[name_features(position)] = features[0].contiguous()
        self.tensors[name_encoder_outputs(position)] = encoded[0].contiguous()
        steps = trace_joint_steps(model, encoded, self.durations)
        for step_position, (_, _, logits) in enumerate(steps):
            self.tensors[name_joint_logits(position, step_position)] = logits.contiguous()
        self.utterances.append(
            DumpedUtterance(
                utterance_id=recording.utterance_id,
                feature_frames=int(feature_lengths[0]),
                encoded_frames=int(encoded_lengths[0]),
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


def dump_reference(output_directory: Path) -> None:
    model = prepare_reference_model()
    dump = ReferenceDump([int(duration) for duration in model.cfg.model_defaults.tdt_durations])
    utterance_ids = list_utterance_ids(PARITY_LANGUAGE)[:DUMPED_UTTERANCE_COUNT]
    for recording in read_language_recordings(PARITY_LANGUAGE, utterance_ids):
        dump.add(model, recording)
    dump.write(model, output_directory)


def main() -> None:
    torch.set_num_threads(REFERENCE_CPU_COUNT)
    torch.set_grad_enabled(False)
    output_directory = Path(OUTPUT_IN_THE_CONTAINER)
    dump_reference(output_directory)
    finish_outputs(output_directory)


if __name__ == "__main__":
    main()
