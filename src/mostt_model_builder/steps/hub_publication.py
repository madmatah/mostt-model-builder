import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import BinaryIO, Protocol

from mostt_model_builder.domain.containers.container_steps import (
    CANDIDATE_IN_THE_CONTAINER,
    OUTPUT_IN_THE_CONTAINER,
    PUBLICATION_REPORT_FILE_NAME,
    REQUEST_IN_THE_CONTAINER,
)
from mostt_model_builder.domain.containers.work_directory import MODEL_CARD_FILE_NAME
from mostt_model_builder.domain.publication.hub_token import (
    MAX_HUB_TOKEN_BYTES,
    TOKEN_ENCODING,
    HubToken,
    HubTokenError,
    parse_hub_token,
)
from mostt_model_builder.domain.publication.publication_report import (
    PublicationReport,
    render_publication_report,
)
from mostt_model_builder.domain.publication.publication_request import (
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

    def list_files(self, repository: str, revision: str) -> list[str]: ...

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
        self.api = library.HfApi(token=token.secret)

    def find_head(self, repository: str) -> RepositoryHead:
        model_info = self.api.model_info(repository)
        return RepositoryHead(str(model_info.sha), bool(model_info.private))

    def list_files(self, repository: str, revision: str) -> list[str]:
        return [str(name) for name in self.api.list_repo_files(repository, revision=revision)]

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


def stage_files(
    request: PublicationRequest, hub: Hub, candidate_directory: Path, request_directory: Path
) -> list[StagedFile]:
    staged = [
        hub.stage_file(candidate_directory / published.name, published.name)
        for published in request.candidate
    ]
    staged.append(hub.stage_file(request_directory / MODEL_CARD_FILE_NAME, MODEL_CARD_FILE_NAME))
    return staged


def publish(
    request: PublicationRequest, hub: Hub, candidate_directory: Path, request_directory: Path
) -> PublicationReport:
    head = hub.find_head(request.repository)
    judge_repository_files(
        hub.list_files(request.repository, head.revision),
        [published.name for published in request.files],
    )
    staged_files = stage_files(request, hub, candidate_directory, request_directory)
    read_files = tuple(
        PublishedFile(staged.name, staged.size_bytes, staged.sha256) for staged in staged_files
    )
    try:
        judge_requested_files(read_files, request, "the upload")
    except RequestedFilesError as error:
        raise PublicationStepError(f"{error}: nothing was committed") from error
    revision = hub.commit(request.repository, staged_files, request.commit_message, head.revision)
    return PublicationReport(
        repository=request.repository,
        is_private=head.is_private,
        parent_revision=head.revision,
        revision=revision,
        files=read_files,
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
