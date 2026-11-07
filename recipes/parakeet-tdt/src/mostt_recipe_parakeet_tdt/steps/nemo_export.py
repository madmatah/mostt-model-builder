import gc
from pathlib import Path
from typing import Any

import numpy
import onnx
import onnxruntime
import torch

from mostt_model_builder.domain.acceptance.parity_reports import (
    ParityReport,
    render_parity_report,
    summarize_differences,
)
from mostt_model_builder.domain.containers.container_paths import (
    OUTPUT_IN_THE_CONTAINER,
    SCRATCH_IN_THE_CONTAINER,
)
from mostt_model_builder.steps.bound_sources import find_bound_source
from mostt_model_builder.steps.output_modes import finish_outputs
from mostt_recipe_parakeet_tdt.domain.cpu_counts import EXPORT_CPU_COUNT
from mostt_recipe_parakeet_tdt.domain.export.parity_checks import (
    ENCODER_PARITY_FILE_NAME,
    ENCODER_PARITY_NAME,
    PARITY_LANGUAGE,
)
from mostt_recipe_parakeet_tdt.domain.export.pinned_sources import (
    NEMO_ARCHIVE,
    NEMO_REPOSITORY,
    ULTRA_REPOSITORY,
    ULTRA_TOKENIZER,
    ULTRA_WEIGHTS,
)
from mostt_recipe_parakeet_tdt.domain.variants.candidate_layout import (
    DECODER_JOINT_FILE_NAME,
    ENCODER_DATA_FILE_NAME,
    ENCODER_FILE_NAME,
    VOCABULARY_FILE_NAME,
)
from mostt_recipe_parakeet_tdt.steps.fleurs_recordings import (
    list_utterance_ids,
    read_language_recordings,
)
from mostt_recipe_parakeet_tdt.steps.nemo_model import (
    NemoModel,
    build_nemo_model,
    check_ultra_tokenizer,
    load_ultra_weights,
)
from mostt_recipe_parakeet_tdt.steps.onnx_export import (
    export_decoder_joint,
    export_module,
    save_model,
    use_exact_position_slices,
    use_scaled_dot_product_attention,
)
from mostt_recipe_parakeet_tdt.steps.pointwise_forward import run_pointwise_as_matrix_products

# torch and NumPy are installed in the export image alone, so mypy on the host knows nothing of
# their types.
type TorchModule = Any
type TorchTensor = Any
type NumpyArray = Any
type EncoderFeatures = tuple[TorchTensor, TorchTensor]

# The trial build measured the parity on the first 20 utterances of FLEURS fr's index.
PARITY_UTTERANCE_COUNT = 20
# onnx-asr reads the encoder's inputs and outputs under the names parakeet-rs's export gives them.
FEATURES_INPUT = "audio_signal"
FEATURE_LENGTHS_INPUT = "length"
ENCODED_OUTPUT = "outputs"
ENCODED_LENGTHS_OUTPUT = "encoded_lengths"
ENCODER_DYNAMIC_AXES = {
    FEATURES_INPUT: {0: "batch", 2: "time"},
    FEATURE_LENGTHS_INPUT: {0: "batch"},
}
RAW_ENCODER_PATH = Path(SCRATCH_IN_THE_CONTAINER) / "encoder-raw.onnx"
BLANK_PIECE = "<blk>"
CPU_PROVIDER = "CPUExecutionProvider"
THREAD_COUNT = EXPORT_CPU_COUNT


# NeMo's encoder takes its inputs by name, which the exporter cannot pass, so a module of its own
# passes them on.
class EncoderModule(torch.nn.Module):  # type: ignore[misc]  # torch is untyped on the host
    def __init__(self, encoder: TorchModule) -> None:
        super().__init__()
        self.encoder = encoder

    def forward(self, audio_signal: TorchTensor, length: TorchTensor) -> TorchTensor:
        return self.encoder(audio_signal=audio_signal, length=length)


def read_parity_recordings() -> list[NumpyArray]:
    parity_ids = list_utterance_ids(PARITY_LANGUAGE)[:PARITY_UTTERANCE_COUNT]
    return [
        recording.samples for recording in read_language_recordings(PARITY_LANGUAGE, parity_ids)
    ]


def compute_features(model: NemoModel, samples: NumpyArray) -> EncoderFeatures:
    signal = torch.from_numpy(samples.copy())
    features, feature_lengths = model.preprocessor(
        input_signal=signal[None], length=torch.tensor([len(samples)])
    )
    return features, feature_lengths


def export_raw_encoder(encoder: TorchModule, example: EncoderFeatures) -> None:
    export_module(
        encoder,
        example,
        RAW_ENCODER_PATH,
        input_names=(FEATURES_INPUT, FEATURE_LENGTHS_INPUT),
        output_names=(ENCODED_OUTPUT, ENCODED_LENGTHS_OUTPUT),
        dynamic_axes=ENCODER_DYNAMIC_AXES,
    )


def save_encoder(output_directory: Path) -> None:
    save_model(
        onnx.load(str(RAW_ENCODER_PATH)),
        output_directory / ENCODER_FILE_NAME,
        ENCODER_DATA_FILE_NAME,
    )
    for raw_file in RAW_ENCODER_PATH.parent.glob(f"{RAW_ENCODER_PATH.name}*"):
        raw_file.unlink()


def write_vocabulary(model: NemoModel, path: Path) -> None:
    pieces = [*model.tokenizer.vocab, BLANK_PIECE]
    path.write_text(
        "".join(f"{piece} {piece_index}\n" for piece_index, piece in enumerate(pieces)),
        encoding="utf-8",
    )


def measure_encoder_parity(
    encoder_path: Path, features: list[EncoderFeatures], reference_outputs: list[NumpyArray]
) -> ParityReport:
    session_options = onnxruntime.SessionOptions()
    session_options.intra_op_num_threads = THREAD_COUNT
    session = onnxruntime.InferenceSession(
        str(encoder_path), session_options, providers=[CPU_PROVIDER]
    )
    differences = [
        float(
            numpy.abs(
                session.run(
                    [ENCODED_OUTPUT],
                    {FEATURES_INPUT: signal.numpy(), FEATURE_LENGTHS_INPUT: length.numpy()},
                )[0]
                - reference_output
            ).max()
        )
        for (signal, length), reference_output in zip(features, reference_outputs, strict=True)
    ]
    return summarize_differences(ENCODER_PARITY_NAME, differences)


def export_model(output_directory: Path) -> tuple[list[EncoderFeatures], list[NumpyArray]]:
    model = build_nemo_model(find_bound_source(NEMO_REPOSITORY, NEMO_ARCHIVE))
    load_ultra_weights(model, find_bound_source(ULTRA_REPOSITORY, ULTRA_WEIGHTS))
    check_ultra_tokenizer(model, find_bound_source(ULTRA_REPOSITORY, ULTRA_TOKENIZER))
    features = [compute_features(model, samples) for samples in read_parity_recordings()]
    encoder = EncoderModule(model.encoder).eval()
    reference_outputs = [encoder(signal, length)[0].numpy().copy() for signal, length in features]
    run_pointwise_as_matrix_products(model)
    use_scaled_dot_product_attention(model.encoder)
    use_exact_position_slices(model.encoder)
    export_raw_encoder(encoder, features[0])
    export_decoder_joint(model, output_directory / DECODER_JOINT_FILE_NAME)
    write_vocabulary(model, output_directory / VOCABULARY_FILE_NAME)
    return features, reference_outputs


def main() -> None:
    torch.set_num_threads(THREAD_COUNT)
    torch.set_grad_enabled(False)
    output_directory = Path(OUTPUT_IN_THE_CONTAINER)
    features, reference_outputs = export_model(output_directory)
    # NeMo's model goes with export_model's frame, and is freed before the raw encoder loads, so
    # that the two never hold their weights at the same time.
    gc.collect()
    save_encoder(output_directory)
    parity_report = measure_encoder_parity(
        output_directory / ENCODER_FILE_NAME, features, reference_outputs
    )
    (output_directory / ENCODER_PARITY_FILE_NAME).write_bytes(render_parity_report(parity_report))
    finish_outputs(output_directory)


if __name__ == "__main__":
    main()
