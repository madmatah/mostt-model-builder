from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import PurePosixPath
from typing import Protocol

from mostt_model_builder.application.verify_build import (
    VerificationFiles,
    VerificationRequest,
    VerifiedBuild,
    refuse_rejected_build,
    verify_build,
)
from mostt_model_builder.domain.containers.container_files import FileKind
from mostt_model_builder.domain.containers.work_directory import locate_model_card
from mostt_model_builder.domain.contract.model_description import ModelDescription
from mostt_model_builder.domain.contract.recipe_contract import VariantConversion
from mostt_model_builder.domain.lock.declared_requirements import DeclaredRequirement
from mostt_model_builder.domain.publication.measuring_libraries import select_measuring_libraries
from mostt_model_builder.domain.publication.model_card import (
    MODEL_CARD_FILE_MODE,
    MODEL_CARD_MAX_SIZE_BYTES,
    CardInputs,
    refuse_card_of_changed_clone,
    render_model_card,
)

CLEAN_COMMAND = "./model-build.sh --work <dir> --recipe <name> clean"
CARD_COMMAND_LINE = "./model-build.sh --work <dir> --recipe <name> card"


class PublicationError(Exception):
    pass


class CloneChanges(Protocol):
    def list_changes(self) -> tuple[str, ...]: ...


@dataclass(frozen=True, slots=True)
class CardRequest:
    verification: VerificationRequest
    model: ModelDescription | None
    template: str | None
    conversions: Sequence[VariantConversion]
    requirements: Sequence[DeclaredRequirement]


@dataclass(frozen=True, slots=True)
class PreparedPublication:
    verified: VerifiedBuild
    card: bytes
    card_path: PurePosixPath


class CardOutcome(Enum):
    WRITTEN = auto()
    UNCHANGED = auto()


def prepare_publication(
    request: CardRequest, clone: CloneChanges, files: VerificationFiles
) -> PreparedPublication:
    refuse_card_of_changed_clone(clone.list_changes())
    candidate = request.verification.candidate
    if request.model is None or request.template is None:
        raise PublicationError(
            f"the recipe {candidate.recipe_name} describes no model: its contract gives no "
            "model, which the card is written from"
        )
    verified = verify_build(request.verification, files)
    refuse_rejected_build(verified)
    card = render_model_card(
        CardInputs(
            template=request.template,
            description=request.model,
            pins=tuple(request.verification.pins),
            verification=verified.verification,
            runs=tuple(request.verification.runs),
            conversions=tuple(request.conversions),
            measuring_libraries=select_measuring_libraries(
                request.model.measuring_library_names,
                request.requirements,
                request.verification.derived_rows,
            ),
            build_commit=candidate.commit,
        )
    )
    return PreparedPublication(
        verified, card, locate_model_card(candidate.recipe_name, candidate.commit)
    )


def read_reviewed_card(
    prepared: PreparedPublication, work_root: PurePosixPath, files: VerificationFiles
) -> bytes | None:
    card_path = prepared.card_path
    reviewed = files.read_host_file(
        work_root / card_path.parent, card_path.name, MODEL_CARD_MAX_SIZE_BYTES
    )
    if reviewed.facts.kind is FileKind.MISSING:
        return None
    if reviewed.content is None:
        raise PublicationError(
            f"{card_path} is not a card the host wrote: remove it with {CLEAN_COMMAND}"
        )
    return reviewed.content


def write_model_card(
    prepared: PreparedPublication, work_root: PurePosixPath, files: VerificationFiles
) -> CardOutcome:
    reviewed_card = read_reviewed_card(prepared, work_root, files)
    if reviewed_card is None:
        files.write_new_file(work_root / prepared.card_path, prepared.card, MODEL_CARD_FILE_MODE)
        return CardOutcome.WRITTEN
    if reviewed_card != prepared.card:
        raise PublicationError(
            f"{prepared.card_path} holds another card than the build's records give: remove it "
            f"with {CLEAN_COMMAND}, then write it again"
        )
    return CardOutcome.UNCHANGED
