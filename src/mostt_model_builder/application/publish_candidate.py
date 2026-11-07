import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Protocol

from mostt_model_builder.application.assemble_candidate import judge_inspected_copies
from mostt_model_builder.application.prepare_publication import (
    CARD_COMMAND_LINE,
    CardRequest,
    CloneChanges,
    PreparedPublication,
    PublicationError,
    prepare_publication,
    read_reviewed_card,
)
from mostt_model_builder.application.run_step import ContainerEngine, StepRunRequest, run_step
from mostt_model_builder.application.verify_build import VerificationFiles
from mostt_model_builder.domain.containers.container_steps import PUBLISH_ROW, StepPlacement
from mostt_model_builder.domain.containers.work_directory import (
    find_next_run_number,
    locate_publications,
    place_publication,
)
from mostt_model_builder.domain.publication.hub_token import HubToken
from mostt_model_builder.domain.publication.publication_report import (
    PUBLICATION_REPORT_FILE_NAME,
    PublicationReport,
    judge_publication_report,
    parse_publication_report,
)
from mostt_model_builder.domain.publication.publication_request import (
    MODEL_CARD_FILE_NAME,
    PUBLICATION_REQUEST_FILE_NAME,
    PublicationRequest,
    PublishedFile,
    describe_commit_message,
    render_publication_request,
)
from mostt_model_builder.domain.publication.repository_files import (
    judge_repository_confirmation,
    judge_repository_name,
)


class PublicationPrompt(Protocol):
    def announce(self, lines: Sequence[str]) -> None: ...

    def ask_repository(self) -> str: ...

    def ask_token(self) -> HubToken: ...


@dataclass(frozen=True, slots=True)
class CandidatePublication:
    card: CardRequest
    repository: str


@dataclass(frozen=True, slots=True)
class PublishedCandidate:
    report: PublicationReport
    placement: StepPlacement


def publish_candidate(
    publication: CandidatePublication,
    clone: CloneChanges,
    prompt: PublicationPrompt,
    engine: ContainerEngine,
    files: VerificationFiles,
) -> PublishedCandidate:
    judge_repository_name(publication.repository)
    candidate = publication.card.verification.candidate
    prepared = prepare_publication(publication.card, clone, files)
    refuse_unreviewed_card(prepared, candidate.work_root, files)
    request = build_publication_request(publication, prepared)
    prompt.announce(describe_upload(request))
    judge_repository_confirmation(publication.repository, prompt.ask_repository())
    token = prompt.ask_token()
    publications_directory = locate_publications(candidate.recipe_name, candidate.commit)
    files.make_directories(candidate.work_root, publications_directory)
    placement = place_publication(
        candidate.recipe_name,
        candidate.commit,
        find_next_run_number(files.list_entry_names(candidate.work_root / publications_directory)),
    )
    judge_inspected_copies(candidate, files)
    prompt.announce([f"uploading; its log is {placement.log_path}"])
    outputs = run_step(
        StepRunRequest(
            row=PUBLISH_ROW,
            placement=placement,
            request_files={
                PUBLICATION_REQUEST_FILE_NAME: render_publication_request(request),
                MODEL_CARD_FILE_NAME: prepared.card,
            },
            source_paths={},
            work_root=candidate.work_root,
            head_tree=candidate.head_tree,
            host=candidate.host,
            parsed_output_names=frozenset({PUBLICATION_REPORT_FILE_NAME}),
        ),
        engine,
        files,
        token,
    )
    report = parse_publication_report(outputs.contents[PUBLICATION_REPORT_FILE_NAME])
    judge_publication_report(report, request)
    return PublishedCandidate(report, placement)


def refuse_unreviewed_card(
    prepared: PreparedPublication, work_root: PurePosixPath, files: VerificationFiles
) -> None:
    reviewed_card = read_reviewed_card(prepared, work_root, files)
    if reviewed_card is None:
        raise PublicationError(
            f"{prepared.card_path} is missing: write it with {CARD_COMMAND_LINE}, and read it "
            "before publishing"
        )
    if reviewed_card != prepared.card:
        raise PublicationError(
            f"{prepared.card_path} differs from the card the build's records give now: write it "
            f"again with {CARD_COMMAND_LINE}, and read it before publishing"
        )


def build_publication_request(
    publication: CandidatePublication, prepared: PreparedPublication
) -> PublicationRequest:
    model = publication.card.model
    if model is None:
        raise PublicationError("the recipe describes no model")
    return PublicationRequest(
        repository=publication.repository,
        commit_message=describe_commit_message(model.title),
        candidate=tuple(
            PublishedFile(digest.name, digest.size_bytes, digest.sha256)
            for digest in prepared.verified.verification.candidate
        ),
        card=PublishedFile(
            MODEL_CARD_FILE_NAME, len(prepared.card), hashlib.sha256(prepared.card).hexdigest()
        ),
    )


def describe_upload(request: PublicationRequest) -> list[str]:
    return [
        f"publish into {request.repository}:",
        *(
            f"  {published.name}, {published.size_bytes} bytes, sha256 {published.sha256}"
            for published in request.files
        ),
    ]
