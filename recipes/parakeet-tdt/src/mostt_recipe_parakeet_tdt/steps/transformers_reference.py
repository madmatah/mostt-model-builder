import json
import time
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file
from transformers import AutoProcessor, ParakeetForTDT

from mostt_model_builder.domain.acceptance.parity_reports import (
    ParityReport,
    render_parity_report,
    summarize_differences,
)
from mostt_model_builder.domain.bench.hypotheses import RUN_LINES_FILE_NAME
from mostt_model_builder.domain.bench.wav_format import SAMPLE_RATE_HZ
from mostt_model_builder.domain.containers.container_steps import (
    INPUTS_IN_THE_CONTAINER,
    OUTPUT_IN_THE_CONTAINER,
    REQUEST_IN_THE_CONTAINER,
    TEMPORARY_DIRECTORY_IN_THE_CONTAINER,
)
from mostt_model_builder.domain.contract.derived_rows import (
    SHARD_REQUEST_FILE_NAME,
    parse_shard_request,
)
from mostt_model_builder.steps.bound_sources import find_bound_source
from mostt_model_builder.steps.output_modes import finish_outputs
from mostt_recipe_parakeet_tdt.domain.export.reference_dump import (
    DumpedUtterance,
    JointStep,
    ReferenceSteps,
    list_tensor_names,
    name_encoder_outputs,
    name_features,
    name_joint_logits,
    parse_reference_steps,
)
from mostt_recipe_parakeet_tdt.domain.transcripts.generated_sequences import (
    read_generated_token_ids,
)
from mostt_recipe_parakeet_tdt.recipe import (
    PROCESSOR_MOUNTS,
    REFERENCE_STEP_NAME,
    REFERENCE_STEPS_FILE_NAME,
    REFERENCE_TENSORS_FILE_NAME,
    TRANSFORMERS_ENCODER_PARITY_FILE_NAME,
    TRANSFORMERS_ENCODER_PARITY_NAME,
    TRANSFORMERS_JOINT_PARITY_FILE_NAME,
    TRANSFORMERS_JOINT_PARITY_NAME,
    TRANSFORMERS_LIMITS,
    ULTRA_MOUNTS,
)
from mostt_recipe_parakeet_tdt.steps.fleurs_recordings import Recording, read_shard_recordings

# torch and Transformers are installed in the Transformers image alone, so mypy on the host knows
# nothing of their types.
type TorchTensor = Any
type TransformersModel = Any
type TransformersProcessor = Any

# Transformers loads a model and its processor from one directory, where Moondream publishes the
# weights and v3 the processor: the directory links each bound file under its own name.
MODEL_DIRECTORY = Path(TEMPORARY_DIRECTORY_IN_THE_CONTAINER) / "transformers-model"
REFERENCE_DIRECTORY = Path(INPUTS_IN_THE_CONTAINER) / REFERENCE_STEP_NAME
MILLISECONDS_PER_SECOND = 1000.0


class TransformersReferenceError(Exception):
    pass


def link_model_files(model_directory: Path) -> None:
    model_directory.mkdir()
    for mount in (*ULTRA_MOUNTS, *PROCESSOR_MOUNTS):
        bound_path = find_bound_source(mount.repository, mount.path)
        (model_directory / bound_path.name).symlink_to(bound_path)


def load_model(model_directory: Path) -> tuple[TransformersModel, TransformersProcessor]:
    model = ParakeetForTDT.from_pretrained(
        str(model_directory), dtype=torch.float32, local_files_only=True
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(str(model_directory), local_files_only=True)
    return model, processor


def read_reference() -> tuple[ReferenceSteps, dict[str, TorchTensor]]:
    reference_steps = parse_reference_steps(
        (REFERENCE_DIRECTORY / REFERENCE_STEPS_FILE_NAME).read_bytes()
    )
    tensors = load_file(str(REFERENCE_DIRECTORY / REFERENCE_TENSORS_FILE_NAME))
    expected_names = set(list_tensor_names(reference_steps))
    if set(tensors) != expected_names:
        raise TransformersReferenceError(
            f"the reference holds the tensors {sorted(tensors)}, where its steps name "
            f"{sorted(expected_names)}"
        )
    return reference_steps, tensors


# NeMo log-softmaxes the tokens and the durations of its joint together, which Transformers does
# not, so their raw logits differ by a constant per head: each head is compared once it is
# log-softmaxed on its own.
def normalize_joint_heads(logits: TorchTensor, token_count: int) -> TorchTensor:
    return torch.cat([logits[:token_count].log_softmax(-1), logits[token_count:].log_softmax(-1)])


def measure_encoder_difference(
    model: TransformersModel,
    utterance: DumpedUtterance,
    features: TorchTensor,
    nemo_encoded: TorchTensor,
) -> float:
    frame_count = features.shape[1]
    attention_mask = (torch.arange(frame_count) < utterance.feature_frames)[None].long()
    outputs = model.get_audio_features(
        input_features=features.transpose(0, 1)[None], attention_mask=attention_mask
    )
    encoded = outputs.last_hidden_state[0, : utterance.encoded_frames]
    return float((encoded - nemo_encoded[: utterance.encoded_frames]).abs().max())


def measure_joint_difference(
    model: TransformersModel,
    projected_encoded: TorchTensor,
    step: JointStep,
    nemo_logits: TorchTensor,
    blank_id: int,
) -> float:
    decoder_input = torch.tensor([[blank_id, *step.prefix]])
    decoder_output = model.decoder(decoder_input)[:, -1:, :]
    logits = model.joint(
        decoder_hidden_states=decoder_output[:, None, :, :],
        encoder_hidden_states=projected_encoded[:, step.frame : step.frame + 1, None, :],
    )[0, 0, 0]
    token_count = blank_id + 1
    difference = normalize_joint_heads(logits, token_count) - normalize_joint_heads(
        nemo_logits, token_count
    )
    return float(difference.abs().max())


def measure_parities(
    model: TransformersModel,
    reference_steps: ReferenceSteps,
    tensors: dict[str, TorchTensor],
) -> tuple[ParityReport, ParityReport]:
    encoder_differences: list[float] = []
    joint_differences: list[float] = []
    for position, utterance in enumerate(reference_steps.utterances):
        nemo_encoded = tensors[name_encoder_outputs(position)].transpose(0, 1)
        encoder_differences.append(
            measure_encoder_difference(
                model, utterance, tensors[name_features(position)], nemo_encoded
            )
        )
        projected_encoded = model.encoder_projector(nemo_encoded[None])
        for step_position, step in enumerate(utterance.steps):
            joint_differences.append(
                measure_joint_difference(
                    model,
                    projected_encoded,
                    step,
                    tensors[name_joint_logits(position, step_position)],
                    reference_steps.blank_id,
                )
            )
    return (
        summarize_differences(TRANSFORMERS_ENCODER_PARITY_NAME, encoder_differences),
        summarize_differences(TRANSFORMERS_JOINT_PARITY_NAME, joint_differences),
    )


# Transformers documents the sequences of generate() as holding the blank tokens, led by the
# decoder's start token, the blank:
# https://github.com/huggingface/transformers/blob/v5.22.1/src/transformers/models/parakeet/generation_parakeet.py
def transcribe_recording(
    model: TransformersModel,
    processor: TransformersProcessor,
    recording: Recording,
    blank_id: int,
) -> dict[str, object]:
    start_s = time.perf_counter()
    inputs = processor(recording.samples, sampling_rate=SAMPLE_RATE_HZ, return_tensors="pt")
    generated = model.generate(**inputs)
    token_ids = read_generated_token_ids(
        [int(token_id) for token_id in generated.sequences[0].tolist()], blank_id
    )
    decoding_time_ms = (time.perf_counter() - start_s) * MILLISECONDS_PER_SECOND
    return {
        "language": recording.language,
        "utterance_id": recording.utterance_id,
        "token_ids": list(token_ids),
        "decoding_time_ms": decoding_time_ms,
    }


def run_transformers_reference(request_directory: Path, output_directory: Path) -> None:
    request = parse_shard_request((request_directory / SHARD_REQUEST_FILE_NAME).read_bytes())
    link_model_files(MODEL_DIRECTORY)
    model, processor = load_model(MODEL_DIRECTORY)
    reference_steps, tensors = read_reference()
    encoder_parity, joint_parity = measure_parities(model, reference_steps, tensors)
    (output_directory / TRANSFORMERS_ENCODER_PARITY_FILE_NAME).write_bytes(
        render_parity_report(encoder_parity)
    )
    (output_directory / TRANSFORMERS_JOINT_PARITY_FILE_NAME).write_bytes(
        render_parity_report(joint_parity)
    )
    recordings = read_shard_recordings(request.languages, request.shard, request.shard_count)
    with (output_directory / RUN_LINES_FILE_NAME).open("w", encoding="ascii") as lines:
        for recording in recordings:
            line = transcribe_recording(model, processor, recording, reference_steps.blank_id)
            lines.write(json.dumps(line) + "\n")


def main() -> None:
    torch.set_num_threads(TRANSFORMERS_LIMITS.cpu_count)
    torch.set_grad_enabled(False)
    output_directory = Path(OUTPUT_IN_THE_CONTAINER)
    run_transformers_reference(Path(REQUEST_IN_THE_CONTAINER), output_directory)
    finish_outputs(output_directory)


if __name__ == "__main__":
    main()
