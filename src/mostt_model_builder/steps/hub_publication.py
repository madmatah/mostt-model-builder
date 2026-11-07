import hashlib
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import BinaryIO, Protocol

from mostt_model_builder.domain.containers.container_paths import (
    CANDIDATE_IN_THE_CONTAINER,
    OUTPUT_IN_THE_CONTAINER,
    REQUEST_IN_THE_CONTAINER,
)
from mostt_model_builder.domain.publication.hub_files import (
    FileDecision,
    HubFile,
    decide_file,
)
from mostt_model_builder.domain.publication.hub_token import (
    MAX_HUB_TOKEN_BYTES,
    TOKEN_ENCODING,
    HubToken,
    HubTokenError,
    parse_hub_token,
)
from mostt_model_builder.domain.publication.publication_report import (
    PUBLICATION_REPORT_FILE_NAME,
    PublicationReport,
    ReportedFile,
    render_publication_report,
)
from mostt_model_builder.domain.publication.publication_request import (
    MODEL_CARD_FILE_NAME,
    PUBLICATION_REQUEST_FILE_NAME,
    PublicationRequest,
    PublishedFile,
    parse_publication_request,
)
from mostt_model_builder.domain.publication.repository_files import judge_repository_files
from mostt_model_builder.domain.publication.requested_files import (
    RequestedFilesError,
    judge_requested_files,
)
from mostt_model_builder.steps.output_modes import finish_outputs

LINE_FEED = b"\n"
DOWNLOAD_CHUNK_BYTES = 2**20


class PublicationStepError(Exception):
    pass


@dataclass(frozen=True, slots=True)
class RepositoryHead:
    revision: str
    is_private: bool


@dataclass(frozen=True, slots=True)
class StagedFile:
    name: str
    size_bytes: int
    sha256: str
    operation: object


class Hub(Protocol):
    def find_head(self, repository: str) -> RepositoryHead: ...

    def list_tree(self, repository: str, revision: str) -> list[HubFile]: ...

    def download_file(self, repository: str, revision: str, name: str) -> Path: ...

    def stage_file(self, local_path: Path, name: str) -> StagedFile: ...

    def commit(
        self,
        repository: str,
        staged_files: Sequence[StagedFile],
        commit_message: str,
        parent_revision: str,
    ) -> str: ...


# huggingface_hub lives in the hub image alone, untyped on the host: this class turns its
# objects into the plain values the publication reads, and is the one place that calls it.
class HuggingFaceHub:
    def __init__(self, library: ModuleType, token: HubToken) -> None:
        self.library = library
        self.token = token
        self.api = library.HfApi(token=token.secret)

    def find_head(self, repository: str) -> RepositoryHead:
        model_info = self.api.model_info(repository)
        return RepositoryHead(str(model_info.sha), bool(model_info.private))

    # A listed file's large-file-storage SHA-256 is the Hub's `lfs` field; a folder holds no file
    # of its own, and the recursive listing gives the files inside it.
    def list_tree(self, repository: str, revision: str) -> list[HubFile]:
        return [
            HubFile(
                name=str(entry.path),
                size_bytes=int(entry.size),
                large_file_storage_sha256=None if entry.lfs is None else str(entry.lfs.sha256),
            )
            for entry in self.api.list_repo_tree(repository, revision=revision, recursive=True)
            if isinstance(entry, self.library.RepoFile)
        ]

    def download_file(self, repository: str, revision: str, name: str) -> Path:
        return Path(
            self.library.hf_hub_download(
                repo_id=repository, filename=name, revision=revision, token=self.token.secret
            )
        )

    def stage_file(self, local_path: Path, name: str) -> StagedFile:
        operation = self.library.CommitOperationAdd(
            path_in_repo=name, path_or_fileobj=str(local_path)
        )
        upload_info = operation.upload_info
        return StagedFile(name, int(upload_info.size), bytes(upload_info.sha256).hex(), operation)

    def commit(
        self,
        repository: str,
        staged_files: Sequence[StagedFile],
        commit_message: str,
        parent_revision: str,
    ) -> str:
        commit_info = self.api.create_commit(
            repository,
            [staged.operation for staged in staged_files],
            commit_message=commit_message,
            parent_commit=parent_revision,
        )
        return str(commit_info.oid)


# A refusal never quotes the line, which may be a token typed with a mistake.
def read_token_line(stream: BinaryIO) -> HubToken:
    line = stream.readline(MAX_HUB_TOKEN_BYTES + 1)
    if len(line) > MAX_HUB_TOKEN_BYTES or not line.endswith(LINE_FEED):
        raise HubTokenError(
            f"the standard input holds no line of at most {MAX_HUB_TOKEN_BYTES} bytes"
        )
    try:
        text = line.removesuffix(LINE_FEED).decode(TOKEN_ENCODING)
    except UnicodeDecodeError:
        raise HubTokenError("the token holds a byte outside ASCII") from None
    return parse_hub_token(text)


# hf_hub_download writes the whole file into the container's /tmp before it is read, so only that
# tmpfs bounds it, and the rule downloads only a file that the Hub lists at 16 MiB at most. The
# reading stops past the listed size, since a file of another size differs whatever its bytes.
def digest_listed_file(
    hub: Hub, repository: str, revision: str, published: PublishedFile
) -> str | None:
    path = hub.download_file(repository, revision, published.name)
    digest = hashlib.sha256()
    read_bytes = 0
    with path.open("rb") as stream:
        while chunk := stream.read(DOWNLOAD_CHUNK_BYTES):
            read_bytes += len(chunk)
            if read_bytes > published.size_bytes:
                return None
            digest.update(chunk)
    return digest.hexdigest() if read_bytes == published.size_bytes else None


def find_kept_sha256(
    hub: Hub, repository: str, revision: str, published: PublishedFile, listed: HubFile | None
) -> str | None:
    match decide_file(published, listed):
        case FileDecision.KEPT if listed is not None:
            return listed.large_file_storage_sha256
        case FileDecision.COMPARE_CONTENT:
            sha256 = digest_listed_file(hub, repository, revision, published)
            return sha256 if sha256 == published.sha256 else None
        case _:
            return None


def locate_published_file(
    published: PublishedFile, candidate_directory: Path, request_directory: Path
) -> Path:
    if published.name == MODEL_CARD_FILE_NAME:
        return request_directory / MODEL_CARD_FILE_NAME
    return candidate_directory / published.name


def publish(
    request: PublicationRequest, hub: Hub, candidate_directory: Path, request_directory: Path
) -> PublicationReport:
    head = hub.find_head(request.repository)
    listed = hub.list_tree(request.repository, head.revision)
    judge_repository_files(
        [listed_file.name for listed_file in listed],
        [published.name for published in request.files],
    )
    listed_by_name = {listed_file.name: listed_file for listed_file in listed}
    reported: dict[str, ReportedFile] = {}
    staged_files = []
    for published in request.files:
        listed_file = listed_by_name.get(published.name)
        kept_sha256 = find_kept_sha256(
            hub, request.repository, head.revision, published, listed_file
        )
        if kept_sha256 is not None and listed_file is not None:
            reported[published.name] = ReportedFile(
                published.name, listed_file.size_bytes, kept_sha256, is_uploaded=False
            )
            continue
        staged = hub.stage_file(
            locate_published_file(published, candidate_directory, request_directory),
            published.name,
        )
        staged_files.append(staged)
        reported[published.name] = ReportedFile(
            staged.name, staged.size_bytes, staged.sha256, is_uploaded=True
        )
    files = tuple(reported[published.name] for published in request.files)
    try:
        judge_requested_files(
            tuple(reported_file.published for reported_file in files), request, "the upload"
        )
    except RequestedFilesError as error:
        raise PublicationStepError(f"{error}: nothing was committed") from error
    revision = head.revision
    if staged_files:
        revision = hub.commit(
            request.repository, staged_files, request.commit_message, head.revision
        )
    return PublicationReport(
        repository=request.repository,
        is_private=head.is_private,
        parent_revision=head.revision,
        revision=revision,
        files=files,
    )


def main() -> None:
    token = read_token_line(sys.stdin.buffer)
    request_directory = Path(REQUEST_IN_THE_CONTAINER)
    output_directory = Path(OUTPUT_IN_THE_CONTAINER)
    request = parse_publication_request(
        (request_directory / PUBLICATION_REQUEST_FILE_NAME).read_bytes()
    )
    # huggingface_hub is imported here, in the hub image alone, so that the host's tests import
    # this module and run its orchestration on a fake hub.
    import huggingface_hub

    hub = HuggingFaceHub(huggingface_hub, token)
    report = publish(request, hub, Path(CANDIDATE_IN_THE_CONTAINER), request_directory)
    (output_directory / PUBLICATION_REPORT_FILE_NAME).write_bytes(render_publication_report(report))
    finish_outputs(output_directory)


if __name__ == "__main__":
    main()
