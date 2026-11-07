import hashlib
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Protocol

from mostt_model_builder.application.assemble_candidate import (
    CandidateRequest,
    judge_candidate_file,
    judge_inspected_copies,
    locate_candidate_directory,
)
from mostt_model_builder.application.check_published import RefusedListingError, TreeListing
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
from mostt_model_builder.domain.publication.git_blobs import compute_git_blob_id
from mostt_model_builder.domain.publication.hub_files import FileDecision, HubFile, decide_file
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
from mostt_model_builder.domain.publication.published_tree import (
    FILE_ENTRY_TYPE,
    PUBLISHED_LISTING_MAX_SIZE_BYTES,
    ListedEntry,
    parse_published_listing,
    refuse_listed_directories,
)
from mostt_model_builder.domain.publication.repository_files import (
    judge_repository_confirmation,
    judge_repository_files,
    judge_repository_name,
)
from mostt_model_builder.domain.sources.source_urls import build_tree_url

# The container commits to the repository's main branch, so the summary reads that branch's head.
HEAD_BRANCH = "main"


class PublicationPrompt(Protocol):
    def announce(self, lines: Sequence[str]) -> None: ...

    def ask_repository(self) -> str: ...

    def ask_token(self) -> HubToken: ...


@dataclass(frozen=True, slots=True)
class CandidatePublication:
    card: CardRequest
    repository: str
    is_dry_run: bool = False


@dataclass(frozen=True, slots=True)
class SummarizedFile:
    name: str
    size_bytes: int
    is_kept: bool


@dataclass(frozen=True, slots=True)
class PublicationSummary:
    repository: str
    files: tuple[SummarizedFile, ...]


@dataclass(frozen=True, slots=True)
class SummaryUnavailable:
    repository: str


@dataclass(frozen=True, slots=True)
class DryRunEnded:
    repository: str


type PublicationEvent = PublicationSummary | SummaryUnavailable | DryRunEnded


@dataclass(frozen=True, slots=True)
class PublishedCandidate:
    report: PublicationReport
    placement: StepPlacement


def publish_candidate(
    publication: CandidatePublication,
    clone: CloneChanges,
    prompt: PublicationPrompt,
    listing: TreeListing,
    engine: ContainerEngine,
    files: VerificationFiles,
    report: Callable[[PublicationEvent], None],
) -> PublishedCandidate | None:
    judge_repository_name(publication.repository)
    candidate = publication.card.verification.candidate
    prepared = prepare_publication(publication.card, clone, files)
    refuse_unreviewed_card(prepared, candidate.work_root, files)
    request = build_publication_request(publication, prepared)
    prompt.announce(describe_upload(request))
    report(summarize_publication(request, prepared.card, candidate, listing, files))
    if publication.is_dry_run:
        report(DryRunEnded(publication.repository))
        return None
    judge_repository_confirmation(publication.repository, prompt.ask_repository())
    token = prompt.ask_token()
    recipe_name = publication.card.recipe_name
    publications_directory = locate_publications(recipe_name)
    files.make_directories(candidate.work_root, publications_directory)
    placement = place_publication(
        recipe_name,
        find_next_run_number(files.list_entry_names(candidate.work_root / publications_directory)),
        candidate.candidate_directory,
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
    publication_report = parse_publication_report(outputs.contents[PUBLICATION_REPORT_FILE_NAME])
    judge_publication_report(publication_report, request)
    return PublishedCandidate(publication_report, placement)


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


# The summary only advises the maintainer before the token: the container decides again at commit
# time, by SHA-256, since the Hub can change in between, and check-published verifies every file.
def summarize_publication(
    request: PublicationRequest,
    card: bytes,
    candidate: CandidateRequest,
    listing: TreeListing,
    files: VerificationFiles,
) -> PublicationSummary | SummaryUnavailable:
    try:
        content = listing.read_listing(
            build_tree_url(request.repository, HEAD_BRANCH), PUBLISHED_LISTING_MAX_SIZE_BYTES
        )
    except RefusedListingError:
        return SummaryUnavailable(request.repository)
    entries = parse_published_listing(content)
    refuse_listed_directories(entries)
    judge_repository_files(
        [entry.path for entry in entries], [published.name for published in request.files]
    )
    listed = {entry.path: entry for entry in entries if entry.entry_type == FILE_ENTRY_TYPE}
    return PublicationSummary(
        request.repository,
        tuple(
            SummarizedFile(
                published.name,
                published.size_bytes,
                is_listed_alike(published, listed.get(published.name), card, candidate, files),
            )
            for published in request.files
        ),
    )


def is_listed_alike(
    published: PublishedFile,
    entry: ListedEntry | None,
    card: bytes,
    candidate: CandidateRequest,
    files: VerificationFiles,
) -> bool:
    hub_file = (
        None
        if entry is None
        else HubFile(entry.path, entry.size_bytes, entry.large_file_storage_sha256)
    )
    match decide_file(published, hub_file):
        case FileDecision.KEPT:
            return True
        case FileDecision.COMPARE_CONTENT if entry is not None:
            content = read_published_bytes(published, card, candidate, files)
            return compute_git_blob_id(content) == entry.blob_id
        case _:
            return False


def read_published_bytes(
    published: PublishedFile, card: bytes, candidate: CandidateRequest, files: VerificationFiles
) -> bytes:
    if published.name == MODEL_CARD_FILE_NAME:
        return card
    candidate_directory = locate_candidate_directory(candidate, files)
    written = files.read_host_file(
        candidate.work_root / candidate_directory, published.name, published.size_bytes
    )
    judge_candidate_file(
        published.name, written.facts, published.size_bytes, candidate.host.user_id
    )
    if written.content is None:
        raise PublicationError(f"the candidate file {published.name} was not read")
    return written.content


def describe_upload(request: PublicationRequest) -> list[str]:
    return [
        f"publish into {request.repository}:",
        *(
            f"  {published.name}, {published.size_bytes} bytes, sha256 {published.sha256}"
            for published in request.files
        ),
    ]
