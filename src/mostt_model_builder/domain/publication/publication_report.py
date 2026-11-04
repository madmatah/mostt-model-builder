import json
from dataclasses import dataclass

from mostt_model_builder.domain.containers.container_steps import (
    PUBLICATION_REPORT_MAX_SIZE_BYTES,
)
from mostt_model_builder.domain.publication.publication_request import (
    PublicationRequest,
    PublicationRequestError,
    PublishedFile,
    decode_document,
    parse_published_file,
    parse_repository,
    render_published_file,
    require_object,
)
from mostt_model_builder.domain.publication.requested_files import (
    RequestedFilesError,
    judge_requested_files,
)
from mostt_model_builder.domain.sources.source_pins import REVISION_GRAMMAR

REPORT_ENCODING = "ascii"
REPORT_KEYS = frozenset({"repository", "is_private", "parent_revision", "revision", "files"})
MAX_REPORTED_FILE_COUNT = 256


class PublicationReportError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class PublicationReport:
    repository: str
    is_private: bool
    parent_revision: str
    revision: str
    files: tuple[PublishedFile, ...]


def render_publication_report(report: PublicationReport) -> bytes:
    document = {
        "repository": report.repository,
        "is_private": report.is_private,
        "parent_revision": report.parent_revision,
        "revision": report.revision,
        "files": [render_published_file(published) for published in report.files],
    }
    return (json.dumps(document, indent=2, ensure_ascii=True) + "\n").encode(REPORT_ENCODING)


def parse_publication_report(content: bytes) -> PublicationReport:
    try:
        document = decode_document(content, PUBLICATION_REPORT_MAX_SIZE_BYTES, "the report")
        fields = require_object(document, REPORT_KEYS, "the report")
        repository = parse_repository(fields["repository"], "the report")
        entries = fields["files"]
        if not isinstance(entries, list) or len(entries) > MAX_REPORTED_FILE_COUNT:
            raise PublicationReportError(
                f"the report's files are not a list of at most {MAX_REPORTED_FILE_COUNT}"
            )
        files = tuple(parse_published_file(entry, "the report") for entry in entries)
    except PublicationRequestError as error:
        raise PublicationReportError(str(error)) from error
    is_private = fields["is_private"]
    if not isinstance(is_private, bool):
        raise PublicationReportError(f"the report says the repository is private: {is_private!a}")
    return PublicationReport(
        repository=repository,
        is_private=is_private,
        parent_revision=parse_revision(fields["parent_revision"], "parent revision"),
        revision=parse_revision(fields["revision"], "revision"),
        files=files,
    )


def parse_revision(field: object, what: str) -> str:
    if not isinstance(field, str) or not REVISION_GRAMMAR.fullmatch(field):
        raise PublicationReportError(f"the report's {what} is {field!a}, not 40 hexadecimal digits")
    return field


# The report shows what the container read and committed; what the Hub stored, only
# check-published shows.
def judge_publication_report(report: PublicationReport, request: PublicationRequest) -> None:
    if report.repository != request.repository:
        raise PublicationReportError(
            f"the report names the repository {report.repository}, not {request.repository}"
        )
    if report.revision == report.parent_revision:
        raise PublicationReportError("the report's revision is its parent: nothing was committed")
    try:
        judge_requested_files(report.files, request, "the report")
    except RequestedFilesError as error:
        raise PublicationReportError(str(error)) from error
