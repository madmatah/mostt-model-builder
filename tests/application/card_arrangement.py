from dataclasses import dataclass, replace
from pathlib import PurePosixPath

from accepted_build import (
    RUN_DIRECTORY,
    SAME_TOKENS_EVERYWHERE,
    WORK_ROOT,
    build_request,
)

from mostt_model_builder.application.prepare_publication import CardRequest
from mostt_model_builder.domain.contract.model_description import (
    ModelDescription,
    VariantDescription,
)
from mostt_model_builder.domain.lock.declared_requirements import DeclaredRequirement
from mostt_model_builder.domain.sources.source_pins import RepositoryType, SourcePin

CARD_PATH = WORK_ROOT / RUN_DIRECTORY / "README.md"
WEIGHTS_PIN = SourcePin(
    RepositoryType.MODEL,
    "maker/weights",
    "89abcdef0123456789abcdef0123456789abcdef",
    "model.safetensors",
    7,
    "2" * 64,
)
MODEL = ModelDescription(
    title="Weights in ONNX",
    template_path=PurePosixPath("model-card.md"),
    measuring_library_names=("onnx-asr",),
    variants=(VariantDescription("float32", "float32"),),
)
TEMPLATE = "# $title\n\n$variant_table\n\nMeasured $measured_with.\n"
REQUIREMENTS = (
    DeclaredRequirement("family-transcribe-cpu", "numpy", (), "2.5.4"),
    DeclaredRequirement("family-transcribe-cpu", "onnx-asr", (), "0.12.0"),
    DeclaredRequirement("family-transcribe-gpu", "onnx-asr", (), "0.12.0"),
    DeclaredRequirement("family-export", "onnx-asr", (), "0.11.0"),
)


@dataclass
class FakeClone:
    changed_paths: tuple[str, ...] = ()

    def list_changes(self) -> tuple[str, ...]:
        return self.changed_paths


def build_card_request(
    model: ModelDescription | None = MODEL, is_rejected: bool = False
) -> CardRequest:
    verification = build_request(SAME_TOKENS_EVERYWHERE) if is_rejected else build_request()
    return CardRequest(
        verification=replace(verification, pins=(*verification.pins, WEIGHTS_PIN)),
        model=model,
        template=None if model is None else TEMPLATE,
        conversions=(),
        requirements=REQUIREMENTS,
    )
