import hashlib
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Protocol

from mostt_model_builder.application.assemble_candidate import (
    CandidateFiles,
    CandidateRequest,
    DigestFiles,
    digest_candidate,
    judge_inspected_copies,
)
from mostt_model_builder.application.fetch_sources import (
    SourceDownloader,
    SourceFiles,
    fetch_source,
)
from mostt_model_builder.application.prepare_publication import CARD_COMMAND_LINE
from mostt_model_builder.domain.containers.work_directory import (
    WorkDirectory,
    locate_model_card,
    locate_source,
)
from mostt_model_builder.domain.contract.model_description import ModelDescription
from mostt_model_builder.domain.publication.model_card import (
    MODEL_CARD_MAX_SIZE_BYTES,
    is_written_card,
)
from mostt_model_builder.domain.publication.published_tree import (
    PUBLISHED_LISTING_MAX_SIZE_BYTES,
    ExpectedFile,
    judge_published_listing,
    judge_published_revision,
    parse_published_listing,
    pin_published_file,
)
from mostt_model_builder.domain.sources.source_urls import build_tree_url


class PublishedCheckError(Exception):
    pass


class TreeListing(Protocol):
    def read_listing(self, url: str, max_size_bytes: int) -> bytes: ...


class PublishedCheckFiles(CandidateFiles, DigestFiles, Protocol):
    pass


@dataclass(frozen=True, slots=True)
class PublishedCheckRequest:
    repository: str
    revision: str
    candidate: CandidateRequest
    model: ModelDescription | None


@dataclass(frozen=True, slots=True)
class CheckedFile:
    name: str
    size_bytes: int
    sha256: str
    source_path: PurePosixPath


def check_published(
    request: PublishedCheckRequest,
    listing: TreeListing,
    downloader: SourceDownloader,
    files: PublishedCheckFiles,
    source_files: SourceFiles,
) -> tuple[CheckedFile, ...]:
    judge_published_revision(request.repository, request.revision)
    candidate = request.candidate
    if request.model is None:
        raise PublishedCheckError(
            f"the recipe {candidate.recipe_name} describes no model, so it publishes none"
        )
    expected_files = (*collect_candidate_files(candidate, files), read_card_file(candidate, files))
    entries = parse_published_listing(
        listing.read_listing(
            build_tree_url(request.repository, request.revision),
            PUBLISHED_LISTING_MAX_SIZE_BYTES,
        )
    )
    judge_published_listing(entries, expected_files)
    work_directory = WorkDirectory(str(candidate.work_root))
    checked = []
    for expected in expected_files:
        pin = pin_published_file(request.repository, request.revision, expected)
        fetch_source(pin, work_directory, downloader, source_files)
        checked.append(
            CheckedFile(expected.name, expected.size_bytes, expected.sha256, locate_source(pin))
        )
    return tuple(checked)


def collect_candidate_files(
    candidate: CandidateRequest, files: PublishedCheckFiles
) -> tuple[ExpectedFile, ...]:
    judge_inspected_copies(candidate, files)
    return tuple(
        ExpectedFile(digest.name, digest.size_bytes, digest.sha256)
        for digest in digest_candidate(candidate, files)
    )


def read_card_file(candidate: CandidateRequest, files: PublishedCheckFiles) -> ExpectedFile:
    card_path = locate_model_card(candidate.recipe_name, candidate.commit)
    card = files.read_host_file(
        candidate.work_root / card_path.parent, card_path.name, MODEL_CARD_MAX_SIZE_BYTES
    )
    if card.content is None or not is_written_card(card.facts, candidate.host.user_id):
        raise PublishedCheckError(
            f"{card_path} is missing, or is not the card that card writes: write it with "
            f"{CARD_COMMAND_LINE}"
        )
    return ExpectedFile(card_path.name, len(card.content), hashlib.sha256(card.content).hexdigest())
