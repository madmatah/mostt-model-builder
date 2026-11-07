from dataclasses import dataclass
from enum import Enum, auto

from mostt_model_builder.domain.publication.publication_request import PublishedFile

# Uploading a file again costs no more than downloading it to compare, and the publication's /tmp
# stays bounded, so a larger file without the Hub's SHA-256 is sent again.
COMPARED_CONTENT_MAX_SIZE_BYTES = 16 * 2**20


class FileDecision(Enum):
    KEPT = auto()
    UPLOAD = auto()
    COMPARE_CONTENT = auto()


@dataclass(frozen=True, slots=True)
class HubFile:
    name: str
    size_bytes: int
    large_file_storage_sha256: str | None


def decide_file(requested: PublishedFile, listed: HubFile | None) -> FileDecision:
    if listed is None or listed.size_bytes != requested.size_bytes:
        return FileDecision.UPLOAD
    if listed.large_file_storage_sha256 is not None:
        if listed.large_file_storage_sha256 == requested.sha256:
            return FileDecision.KEPT
        return FileDecision.UPLOAD
    if requested.size_bytes <= COMPARED_CONTENT_MAX_SIZE_BYTES:
        return FileDecision.COMPARE_CONTENT
    return FileDecision.UPLOAD
