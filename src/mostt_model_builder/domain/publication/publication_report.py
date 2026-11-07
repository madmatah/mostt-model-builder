import json
from dataclasses import dataclass

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

PUBLICATION_REPORT_FILE_NAME = "publication.json"
PUBLICATION_REPORT_MAX_SIZE_BYTES = 64 * 1024
REPORT_ENCODING = "ascii"
REPORT_KEYS = frozenset({"repository", "is_private", "parent_revision", "revision", "files"})
UPLOADED_KEY = "uploaded"
MAX_REPORTED_FILE_COUNT = 256


class PublicationReportError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class ReportedFile:
    name: str
    size_bytes: int
    sha256: str
    is_uploaded: bool

    @property
    def published(self) -> PublishedFile:
        return PublishedFile(self.name, self.size_bytes, self.sha256)


@dataclass(frozen=True, slots=True)
class PublicationReport:
    repository: str
    is_private: bool
    parent_revision: str
    revision: str
    files: tuple[ReportedFile, ...]

    @property
    def has_uploaded(self) -> bool:
        return any(reported.is_uploaded for reported in self.files)


def render_publication_report(report: PublicationReport) -> bytes:
    document = {
        "repository": report.repository,
        "is_private": report.is_private,
        "parent_revision": report.parent_revision,
        "revision": report.revision,
        "files": [
            {**render_published_file(reported.published), UPLOADED_KEY: reported.is_uploaded}
            for reported in report.files
        ],
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
        files = tuple(parse_reported_file(entry) for entry in entries)
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


def parse_reported_file(entry: object) -> ReportedFile:
    if not isinstance(entry, dict) or not isinstance(entry.get(UPLOADED_KEY), bool):
        raise PublicationReportError(
            f"a file of the report is not an object that says whether it was {UPLOADED_KEY}"
        )
    published = parse_published_file(
        {key: field for key, field in entry.items() if key != UPLOADED_KEY}, "the report"
    )
    return ReportedFile(published.name, published.size_bytes, published.sha256, entry[UPLOADED_KEY])


def parse_revision(field: object, what: str) -> str:
    if not isinstance(field, str) or not REVISION_GRAMMAR.fullmatch(field):
        raise PublicationReportError(f"the report's {what} is {field!a}, not 40 hexadecimal digits")
    return field


# The report shows what the container read and committed, and for a file it left in place what the
# Hub listed; what the Hub stored, only check-published shows.
def judge_publication_report(report: PublicationReport, request: PublicationRequest) -> None:
    if report.repository != request.repository:
        raise PublicationReportError(
            f"the report names the repository {report.repository}, not {request.repository}"
        )
    if report.has_uploaded and report.revision == report.parent_revision:
        raise PublicationReportError(
            "the report's revision is its parent, though it uploaded files: nothing was committed"
        )
    if not report.has_uploaded and report.revision != report.parent_revision:
        raise PublicationReportError("the report gives a new revision, though it uploaded no file")
    try:
        judge_requested_files(
            tuple(reported.published for reported in report.files), request, "the report"
        )
    except RequestedFilesError as error:
        raise PublicationReportError(str(error)) from error
