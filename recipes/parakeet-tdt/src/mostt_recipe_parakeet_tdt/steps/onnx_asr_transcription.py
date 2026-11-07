import json
import time
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import onnx_asr
import onnxruntime

from mostt_model_builder.domain.bench.run_lines import RUN_LINES_FILE_NAME
from mostt_model_builder.domain.bench.wav_format import SAMPLE_RATE_HZ
from mostt_model_builder.domain.containers.container_paths import (
    CANDIDATE_IN_THE_CONTAINER,
    OUTPUT_IN_THE_CONTAINER,
    REQUEST_IN_THE_CONTAINER,
    TEMPORARY_DIRECTORY_IN_THE_CONTAINER,
)
from mostt_model_builder.domain.containers.devices import Device
from mostt_model_builder.domain.contract.shard_request import (
    SHARD_REQUEST_FILE_NAME,
    ShardRequest,
    parse_shard_request,
)
from mostt_model_builder.domain.onnx.node_placement import (
    PLACEMENT_FILE_NAME,
    PlacedNodes,
    PlacementSummary,
    render_placement_summary,
)
from mostt_model_builder.steps.output_modes import finish_outputs
from mostt_recipe_parakeet_tdt.domain.cpu_counts import TRANSCRIPTION_CPU_COUNTS
from mostt_recipe_parakeet_tdt.domain.variants.candidate_layout import (
    ONNX_ASR_MODEL_TYPE,
    read_quantization,
)
from mostt_recipe_parakeet_tdt.steps.fleurs_recordings import Recording, read_shard_recordings

# onnx-asr and ONNX Runtime are installed in the transcription images alone, so mypy on the host
# knows nothing of their types.
type OnnxAsrModel = Any

# The first utterances pay for the allocation of the sessions' buffers, so three utterances run
# before any is timed.
WARM_UP_COUNT = 3
# The trial build profiled 5 utterances, after the warm-ups, to find where ONNX Runtime places each
# node.
PROFILED_UTTERANCE_COUNT = 5
CPU_PROVIDER = "CPUExecutionProvider"
CUDA_PROVIDER = "CUDAExecutionProvider"
PROFILE_PREFIX = Path(TEMPORARY_DIRECTORY_IN_THE_CONTAINER) / "onnx-runtime-profile"
PROFILE_NODE_CATEGORY = "Node"
KERNEL_TIME_SUFFIX = "_kernel_time"
TYPE_SHAPE_FIELDS = ("input_type_shape", "output_type_shape")
MILLISECONDS_PER_SECOND = 1000.0


def load_model(request: ShardRequest, profile_prefix: Path | None = None) -> OnnxAsrModel:
    if request.device is Device.GPU:
        onnxruntime.preload_dlls()
    options = onnxruntime.SessionOptions()
    options.intra_op_num_threads = TRANSCRIPTION_CPU_COUNTS[request.device]
    options.inter_op_num_threads = 1
    if profile_prefix is not None:
        options.enable_profiling = True
        options.profile_file_prefix = str(profile_prefix)
    providers: list[object] = [CPU_PROVIDER]
    if request.device is Device.GPU:
        providers = [(CUDA_PROVIDER, {}), CPU_PROVIDER]
    return onnx_asr.load_model(
        ONNX_ASR_MODEL_TYPE,
        str(CANDIDATE_IN_THE_CONTAINER),
        quantization=read_quantization(request.files),
        sess_options=options,
        providers=providers,
    ).with_timestamps()


def transcribe(model: OnnxAsrModel, recordings: Sequence[Recording], lines_path: Path) -> None:
    for recording in recordings[:WARM_UP_COUNT]:
        model.recognize(recording.samples, sample_rate=SAMPLE_RATE_HZ)
    with lines_path.open("w", encoding="ascii") as lines:
        for recording in recordings:
            start_s = time.perf_counter()
            recognition = model.recognize(recording.samples, sample_rate=SAMPLE_RATE_HZ)
            decoding_time_ms = (time.perf_counter() - start_s) * MILLISECONDS_PER_SECOND
            line = {
                "language": recording.language,
                "utterance_id": recording.utterance_id,
                "pieces": [str(piece) for piece in recognition.tokens],
                "decoding_time_ms": decoding_time_ms,
            }
            lines.write(json.dumps(line) + "\n")


def summarize_profiles(profile_paths: Sequence[Path]) -> PlacementSummary:
    element_types_by_node: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    for profile_path in profile_paths:
        events = json.loads(profile_path.read_text(encoding="utf-8"))
        for event in events:
            arguments = event.get("args", {})
            if event.get("cat") != PROFILE_NODE_CATEGORY or "provider" not in arguments:
                continue
            node_name = str(event.get("name", "")).removesuffix(KERNEL_TIME_SUFFIX)
            key = (
                f"{profile_path.name}/{node_name}",
                str(arguments["provider"]),
                str(arguments.get("op_name", "")),
            )
            element_types = element_types_by_node[key]
            for field in TYPE_SHAPE_FIELDS:
                for entry in arguments.get(field, []):
                    element_types.update(str(element_type) for element_type in entry)
    counts: dict[tuple[str, str, tuple[str, ...]], int] = defaultdict(int)
    for (_, provider, operator), element_types in element_types_by_node.items():
        counts[provider, operator, tuple(sorted(element_types))] += 1
    return PlacementSummary(
        tuple(
            PlacedNodes(provider, operator, element_types, count)
            for (provider, operator, element_types), count in sorted(counts.items())
        )
    )


def profile_placement(
    request: ShardRequest, recordings: Sequence[Recording], placement_path: Path
) -> None:
    model = load_model(request, PROFILE_PREFIX)
    for recording in recordings[: WARM_UP_COUNT + PROFILED_UTTERANCE_COUNT]:
        model.recognize(recording.samples, sample_rate=SAMPLE_RATE_HZ)
    del model
    profile_paths = sorted(PROFILE_PREFIX.parent.glob(f"{PROFILE_PREFIX.name}*.json"))
    placement_path.write_bytes(render_placement_summary(summarize_profiles(profile_paths)))
    for profile_path in profile_paths:
        profile_path.unlink()


def run_transcription(request_directory: Path, output_directory: Path) -> None:
    request = parse_shard_request((request_directory / SHARD_REQUEST_FILE_NAME).read_bytes())
    recordings = read_shard_recordings(request.languages, request.shard, request.shard_count)
    transcribe(load_model(request), recordings, output_directory / RUN_LINES_FILE_NAME)
    if request.device is Device.GPU:
        profile_placement(request, recordings, output_directory / PLACEMENT_FILE_NAME)


def main() -> None:
    output_directory = Path(OUTPUT_IN_THE_CONTAINER)
    run_transcription(Path(REQUEST_IN_THE_CONTAINER), output_directory)
    finish_outputs(output_directory)


if __name__ == "__main__":
    main()
