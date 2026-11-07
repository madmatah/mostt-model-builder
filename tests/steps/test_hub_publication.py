import contextlib
import hashlib
import io
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from mostt_model_builder.domain.publication.hub_files import (
    COMPARED_CONTENT_MAX_SIZE_BYTES,
    HubFile,
)
from mostt_model_builder.domain.publication.hub_token import (
    HubToken,
    HubTokenError,
    parse_hub_token,
)
from mostt_model_builder.domain.publication.publication_report import (
    PublicationReport,
    ReportedFile,
)
from mostt_model_builder.domain.publication.publication_request import (
    PublicationRequest,
    PublishedFile,
)
from mostt_model_builder.domain.publication.repository_files import RepositoryFilesError
from mostt_model_builder.steps.hub_publication import (
    HuggingFaceHub,
    PublicationStepError,
    RepositoryHead,
    StagedFile,
    publish,
    read_token_line,
)

REPOSITORY = "example-user/parakeet-ultra-onnx"
PARENT_REVISION = "0123456789abcdef0123456789abcdef01234567"
REVISION = "89abcdef0123456789abcdef0123456789abcdef"
PLACEHOLDER_TOKEN = "hf_placeholder0123456789"
ENCODER = b"\x08\x07\x3a\x00"
TOKEN_GRAMMAR_REFUSAL = (
    "the token is not hf_ followed by 8 to 256 ASCII letters and digits, as a Hugging Face token is"
)
CARD = b'---\nlicense: "cc-by-4.0"\n---\n'


def describe(name: str, content: bytes) -> PublishedFile:
    return PublishedFile(name, len(content), hashlib.sha256(content).hexdigest())


REQUEST = PublicationRequest(
    REPOSITORY,
    "Publish Weights in ONNX",
    (describe("encoder-model.onnx", ENCODER),),
    describe("README.md", CARD),
)


@dataclass(frozen=True)
class Commit:
    repository: str
    names: list[str]
    commit_message: str
    parent_revision: str


@dataclass
class FakeHub:
    listed_files: list[HubFile]
    download_directory: Path | None = None
    stored: dict[str, bytes] = field(default_factory=dict)
    commits: list[Commit] = field(default_factory=list)
    calls: list[str] = field(default_factory=list)

    def find_head(self, repository: str) -> RepositoryHead:
        self.calls.append(f"find_head {repository}")
        return RepositoryHead(PARENT_REVISION, is_private=True)

    def list_tree(self, repository: str, revision: str) -> list[HubFile]:
        self.calls.append(f"list_tree {repository} {revision}")
        return self.listed_files

    def download_file(self, repository: str, revision: str, name: str) -> Path:
        self.calls.append(f"download_file {repository} {revision} {name}")
        if self.download_directory is None:
            raise LookupError("the fake hub was given no directory to download into")
        path = self.download_directory / name
        path.write_bytes(self.stored[name])
        return path

    def stage_file(self, local_path: Path, name: str) -> StagedFile:
        content = local_path.read_bytes()
        return StagedFile(name, len(content), hashlib.sha256(content).hexdigest(), local_path)

    def commit(
        self,
        repository: str,
        staged_files: Sequence[StagedFile],
        commit_message: str,
        parent_revision: str,
    ) -> str:
        self.commits.append(
            Commit(
                repository,
                [staged.name for staged in staged_files],
                commit_message,
                parent_revision,
            )
        )
        self.calls.append(f"commit {repository} {parent_revision}")
        return REVISION


@dataclass(frozen=True)
class Publication:
    report: PublicationReport
    hub: FakeHub


HUB_ATTRIBUTES = HubFile(".gitattributes", 1519, None)


def list_on_the_hub(name: str, content: bytes, is_large: bool) -> HubFile:
    return HubFile(name, len(content), hashlib.sha256(content).hexdigest() if is_large else None)


def report_kept(name: str, content: bytes) -> ReportedFile:
    published = describe(name, content)
    return ReportedFile(published.name, published.size_bytes, published.sha256, is_uploaded=False)


def report_uploaded(name: str, content: bytes) -> ReportedFile:
    published = describe(name, content)
    return ReportedFile(published.name, published.size_bytes, published.sha256, is_uploaded=True)


def lay_out(directory: Path, encoder: bytes = ENCODER) -> tuple[Path, Path]:
    candidate_directory = directory / "candidate"
    request_directory = directory / "request"
    candidate_directory.mkdir()
    request_directory.mkdir()
    (candidate_directory / "encoder-model.onnx").write_bytes(encoder)
    (request_directory / "README.md").write_bytes(CARD)
    return candidate_directory, request_directory


ENCODER_SHA256 = hashlib.sha256(ENCODER).digest()


@dataclass(frozen=True)
class LibraryCommit:
    repository: str
    operations: list[object]
    commit_message: str
    parent_commit: str


class StubOperation:
    def __init__(self, path_in_repo: str, path_or_fileobj: str) -> None:
        self.path_in_repo = path_in_repo
        self.path_or_fileobj = path_or_fileobj
        self.upload_info = SimpleNamespace(size=len(ENCODER), sha256=ENCODER_SHA256)


@dataclass(frozen=True)
class StubLargeFileStorage:
    sha256: str


@dataclass(frozen=True)
class StubRepoFile:
    path: str
    size: int
    lfs: StubLargeFileStorage | None


@dataclass(frozen=True)
class StubRepoFolder:
    path: str


LISTED_TREE: list[StubRepoFile | StubRepoFolder] = [
    StubRepoFile(".gitattributes", 1519, None),
    StubRepoFile("encoder-model.onnx.data", 2_480_000_000, StubLargeFileStorage("a" * 64)),
    StubRepoFolder("extra"),
]
DOWNLOADED_PATH = "/tmp/huggingface/hub/models--example-user--parakeet-ultra-onnx/README.md"


def download_from_the_stub(*, repo_id: str, filename: str, revision: str, token: str) -> str:
    del repo_id, filename, revision, token
    return DOWNLOADED_PATH


class StubApi:
    def __init__(self, token: str) -> None:
        self.token = token
        self.read_repositories: list[str] = []
        self.commits: list[LibraryCommit] = []

    def model_info(self, repository: str) -> SimpleNamespace:
        self.read_repositories.append(repository)
        return SimpleNamespace(sha=PARENT_REVISION, private=True)

    def list_repo_tree(
        self, repository: str, *, revision: str, recursive: bool
    ) -> list[StubRepoFile | StubRepoFolder]:
        del repository, revision, recursive
        return LISTED_TREE

    def create_commit(
        self,
        repository: str,
        operations: list[object],
        *,
        commit_message: str,
        parent_commit: str,
    ) -> SimpleNamespace:
        self.commits.append(LibraryCommit(repository, operations, commit_message, parent_commit))
        return SimpleNamespace(oid=REVISION)


# The hub image's huggingface_hub, stood in for by the names the adapter calls, so that its
# translation of the library's objects runs on the host.
class StubLibrary(ModuleType):
    HfApi = StubApi
    CommitOperationAdd = StubOperation
    RepoFile = StubRepoFile
    RepoFolder = StubRepoFolder
    hf_hub_download = staticmethod(download_from_the_stub)


def build_library_hub() -> HuggingFaceHub:
    return HuggingFaceHub(StubLibrary("huggingface_hub"), parse_hub_token(PLACEHOLDER_TOKEN))


@dataclass(frozen=True)
class DifferentialPublication:
    report: PublicationReport
    hub: FakeHub


def publish_beside(
    tmp_path: Path, listed: list[HubFile], stored: dict[str, bytes], encoder: bytes = ENCODER
) -> DifferentialPublication:
    candidate_directory, request_directory = lay_out(tmp_path, encoder)
    download_directory = tmp_path / "downloads"
    download_directory.mkdir()
    hub = FakeHub([HUB_ATTRIBUTES, *listed], download_directory, stored)
    request = PublicationRequest(
        REPOSITORY,
        "Publish Weights in ONNX",
        (describe("encoder-model.onnx", encoder),),
        describe("README.md", CARD),
    )
    return DifferentialPublication(
        publish(request, hub, candidate_directory, request_directory), hub
    )


def list_downloads(hub: FakeHub) -> list[str]:
    return [call.split()[-1] for call in hub.calls if call.startswith("download_file")]


def list_committed(hub: FakeHub) -> list[list[str]]:
    return [commit.names for commit in hub.commits]


LARGE_ENCODER = ENCODER + bytes(COMPARED_CONTENT_MAX_SIZE_BYTES + 1 - len(ENCODER))


class TestTheHubPublication:
    class TestTheHuggingFaceHubFindHeadMethod:
        class TestWhenTheLibraryGivesTheModelsInformation:
            @pytest.fixture
            def repository_head(self) -> RepositoryHead:
                return build_library_hub().find_head(REPOSITORY)

            def test_should_read_its_revision_and_its_privacy(
                self, repository_head: RepositoryHead
            ) -> None:
                assert repository_head == RepositoryHead(PARENT_REVISION, is_private=True)

    class TestTheHuggingFaceHubListTreeMethod:
        class TestWhenTheLibraryListsFilesAndAFolder:
            @pytest.fixture
            def listed(self) -> list[HubFile]:
                return build_library_hub().list_tree(REPOSITORY, PARENT_REVISION)

            def test_should_read_each_files_size_and_large_file_storage_sha256_and_skip_folders(
                self, listed: list[HubFile]
            ) -> None:
                assert listed == [
                    HubFile(".gitattributes", 1519, None),
                    HubFile("encoder-model.onnx.data", 2_480_000_000, "a" * 64),
                ]

    class TestTheHuggingFaceHubDownloadFileMethod:
        class TestWhenTheLibraryDownloadsAFile:
            @pytest.fixture
            def downloaded(self) -> Path:
                return build_library_hub().download_file(REPOSITORY, PARENT_REVISION, "README.md")

            def test_should_give_the_path_the_library_wrote(self, downloaded: Path) -> None:
                assert downloaded == Path(DOWNLOADED_PATH)

    class TestTheHuggingFaceHubStageFileMethod:
        class TestWhenTheLibraryReadsAFile:
            @pytest.fixture
            def staged_file(self, tmp_path: Path) -> StagedFile:
                return build_library_hub().stage_file(tmp_path / "encoder.onnx", "encoder.onnx")

            def test_should_read_the_size_and_the_sha256_in_hexadecimal_that_the_library_read(
                self, staged_file: StagedFile
            ) -> None:
                assert (staged_file.name, staged_file.size_bytes, staged_file.sha256) == (
                    "encoder.onnx",
                    len(ENCODER),
                    ENCODER_SHA256.hex(),
                )

    class TestTheHuggingFaceHubCommitMethod:
        class TestWhenTheLibraryCommitsAStagedFile:
            @pytest.fixture
            def committed(self, tmp_path: Path) -> tuple[str, StagedFile, StubApi]:
                hub = build_library_hub()
                staged_file = hub.stage_file(tmp_path / "encoder.onnx", "encoder.onnx")
                revision = hub.commit(REPOSITORY, [staged_file], "Publish it", PARENT_REVISION)
                api: StubApi = hub.api
                return revision, staged_file, api

            def test_should_give_the_revision_of_the_new_commit(
                self, committed: tuple[str, StagedFile, StubApi]
            ) -> None:
                assert committed[0] == REVISION

            def test_should_commit_the_staged_operations_on_the_parent(
                self, committed: tuple[str, StagedFile, StubApi]
            ) -> None:
                _, staged_file, api = committed
                assert api.commits == [
                    LibraryCommit(
                        REPOSITORY, [staged_file.operation], "Publish it", PARENT_REVISION
                    )
                ]

    class TestTheReadTokenLineFunction:
        class TestWhenTheStandardInputHoldsOneLine:
            @pytest.fixture
            def hub_token(self) -> HubToken:
                return read_token_line(io.BytesIO(PLACEHOLDER_TOKEN.encode() + b"\nrest\n"))

            def test_should_read_the_token_of_the_first_line_alone(
                self, hub_token: HubToken
            ) -> None:
                assert hub_token.secret == PLACEHOLDER_TOKEN

        class TestWhenTheLineHasNoEnd:
            @pytest.fixture
            def read_act(self) -> Callable[[], HubToken]:
                return partial(read_token_line, io.BytesIO(b"hf_" + b"a" * 600))

            def test_should_refuse_it_after_its_bound(
                self, read_act: Callable[[], HubToken]
            ) -> None:
                with pytest.raises(
                    HubTokenError,
                    match=re.escape("the standard input holds no line of at most 512 bytes"),
                ):
                    read_act()

        class TestWhenTheTokenIsMalformed:
            @pytest.fixture
            def read_act(self) -> Callable[[], HubToken]:
                return partial(read_token_line, io.BytesIO(b"hf_short\n"))

            def test_should_refuse_it_with_a_message_that_holds_nothing_but_the_grammar(
                self, read_act: Callable[[], HubToken]
            ) -> None:
                with pytest.raises(HubTokenError, match=f"^{re.escape(TOKEN_GRAMMAR_REFUSAL)}$"):
                    read_act()

    class TestThePublishFunction:
        class TestWhenTheRepositoryHoldsTheHubsFileAlone:
            @pytest.fixture
            def publication(self, tmp_path: Path) -> Publication:
                hub = FakeHub([HUB_ATTRIBUTES])
                candidate_directory, request_directory = lay_out(tmp_path)
                return Publication(
                    publish(REQUEST, hub, candidate_directory, request_directory), hub
                )

            def test_should_commit_the_candidate_and_the_card_on_the_parent_it_read(
                self, publication: Publication
            ) -> None:
                assert publication.hub.commits == [
                    Commit(
                        REPOSITORY,
                        ["encoder-model.onnx", "README.md"],
                        "Publish Weights in ONNX",
                        PARENT_REVISION,
                    )
                ]

            def test_should_list_the_files_of_the_parent_before_it_commits(
                self, publication: Publication
            ) -> None:
                assert publication.hub.calls == [
                    f"find_head {REPOSITORY}",
                    f"list_tree {REPOSITORY} {PARENT_REVISION}",
                    f"commit {REPOSITORY} {PARENT_REVISION}",
                ]

            def test_should_report_what_it_read_and_committed(
                self, publication: Publication
            ) -> None:
                assert publication.report == PublicationReport(
                    REPOSITORY,
                    True,
                    PARENT_REVISION,
                    REVISION,
                    (
                        report_uploaded("encoder-model.onnx", ENCODER),
                        report_uploaded("README.md", CARD),
                    ),
                )

        class TestWhenTheRepositoryHoldsAnUnexpectedFile:
            @pytest.fixture
            def publish_act(self, tmp_path: Path) -> Callable[[], PublicationReport]:
                candidate_directory, request_directory = lay_out(tmp_path)
                return partial(
                    publish,
                    REQUEST,
                    FakeHub([HUB_ATTRIBUTES, HubFile("model.safetensors", 7, None)]),
                    candidate_directory,
                    request_directory,
                )

            def test_should_refuse_it(self, publish_act: Callable[[], PublicationReport]) -> None:
                with pytest.raises(RepositoryFilesError, match=re.escape("'model.safetensors'")):
                    publish_act()

        class TestWhenAFileReadsOtherBytesThanTheVerificationAccepted:
            @pytest.fixture
            def publish_act(self, tmp_path: Path) -> Callable[[], PublicationReport]:
                candidate_directory, request_directory = lay_out(tmp_path, ENCODER + b"\x00")
                return partial(
                    publish,
                    REQUEST,
                    FakeHub([HUB_ATTRIBUTES]),
                    candidate_directory,
                    request_directory,
                )

            def test_should_refuse_it_and_say_that_nothing_was_committed(
                self, publish_act: Callable[[], PublicationReport]
            ) -> None:
                with pytest.raises(
                    PublicationStepError,
                    match=re.escape(
                        f"the upload gives {len(ENCODER) + 1} bytes of SHA-256 "
                        f"{hashlib.sha256(ENCODER + b'\x00').hexdigest()} for encoder-model.onnx, "
                        f"where the verification accepted {len(ENCODER)} bytes of SHA-256 "
                        f"{hashlib.sha256(ENCODER).hexdigest()}: nothing was committed"
                    ),
                ):
                    publish_act()

        class TestWhenAFileReadsOtherBytesAndTheRefusalIsCaught:
            @pytest.fixture
            def hub(self, tmp_path: Path) -> FakeHub:
                hub = FakeHub([HUB_ATTRIBUTES])
                candidate_directory, request_directory = lay_out(tmp_path, ENCODER + b"\x00")
                with contextlib.suppress(PublicationStepError):
                    publish(REQUEST, hub, candidate_directory, request_directory)
                return hub

            def test_should_commit_nothing(self, hub: FakeHub) -> None:
                assert hub.commits == []

        class TestWhenTheHubListsTheEncoderWithTheSameLargeFileStorageSha256:
            @pytest.fixture
            def publication(self, tmp_path: Path) -> DifferentialPublication:
                return publish_beside(
                    tmp_path, [list_on_the_hub("encoder-model.onnx", ENCODER, is_large=True)], {}
                )

            def test_should_commit_the_missing_card_alone(
                self, publication: DifferentialPublication
            ) -> None:
                assert list_committed(publication.hub) == [["README.md"]]

            def test_should_report_the_encoder_left_in_place_with_what_the_hub_listed(
                self, publication: DifferentialPublication
            ) -> None:
                assert publication.report.files == (
                    report_kept("encoder-model.onnx", ENCODER),
                    report_uploaded("README.md", CARD),
                )

            def test_should_download_nothing(self, publication: DifferentialPublication) -> None:
                assert list_downloads(publication.hub) == []

        class TestWhenTheHubListsTheEncoderWithAnotherLargeFileStorageSha256:
            @pytest.fixture
            def publication(self, tmp_path: Path) -> DifferentialPublication:
                listed = HubFile("encoder-model.onnx", len(ENCODER), "f" * 64)
                return publish_beside(tmp_path, [listed], {})

            def test_should_upload_it_again(self, publication: DifferentialPublication) -> None:
                assert list_committed(publication.hub) == [["encoder-model.onnx", "README.md"]]

        class TestWhenASmallFileWithoutAHashOnTheHubHoldsTheSameBytes:
            @pytest.fixture
            def publication(self, tmp_path: Path) -> DifferentialPublication:
                return publish_beside(
                    tmp_path,
                    [
                        list_on_the_hub("encoder-model.onnx", ENCODER, is_large=True),
                        list_on_the_hub("README.md", CARD, is_large=False),
                    ],
                    {"README.md": CARD},
                )

            def test_should_download_it_to_compare_its_sha256(
                self, publication: DifferentialPublication
            ) -> None:
                assert list_downloads(publication.hub) == ["README.md"]

            def test_should_commit_nothing_since_nothing_differs(
                self, publication: DifferentialPublication
            ) -> None:
                assert publication.hub.commits == []

            def test_should_report_the_head_as_the_revision(
                self, publication: DifferentialPublication
            ) -> None:
                assert (publication.report.revision, publication.report.has_uploaded) == (
                    PARENT_REVISION,
                    False,
                )

        class TestWhenASmallFileWithoutAHashOnTheHubHoldsOtherBytesOfTheSameSize:
            @pytest.fixture
            def publication(self, tmp_path: Path) -> DifferentialPublication:
                other_card = CARD.replace(b"4.0", b"4.1")
                return publish_beside(
                    tmp_path,
                    [
                        list_on_the_hub("encoder-model.onnx", ENCODER, is_large=True),
                        list_on_the_hub("README.md", other_card, is_large=False),
                    ],
                    {"README.md": other_card},
                )

            def test_should_upload_it(self, publication: DifferentialPublication) -> None:
                assert list_committed(publication.hub) == [["README.md"]]

        @pytest.mark.parametrize(
            "downloaded_card",
            [
                pytest.param(CARD + b"\n", id="one byte longer"),
                pytest.param(CARD[:-1], id="one byte shorter"),
            ],
        )
        class TestWhenTheDownloadOfASmallFileHoldsAnotherSizeThanTheHubListed:
            @pytest.fixture
            def publication(
                self, tmp_path: Path, downloaded_card: bytes
            ) -> DifferentialPublication:
                return publish_beside(
                    tmp_path,
                    [
                        list_on_the_hub("encoder-model.onnx", ENCODER, is_large=True),
                        list_on_the_hub("README.md", CARD, is_large=False),
                    ],
                    {"README.md": downloaded_card},
                )

            def test_should_upload_it(self, publication: DifferentialPublication) -> None:
                assert list_committed(publication.hub) == [["README.md"]]

        class TestWhenALargeFileHasNoHashOnTheHub:
            @pytest.fixture
            def publication(self, tmp_path: Path) -> DifferentialPublication:
                return publish_beside(
                    tmp_path,
                    [list_on_the_hub("encoder-model.onnx", LARGE_ENCODER, is_large=False)],
                    {},
                    encoder=LARGE_ENCODER,
                )

            def test_should_stage_it_without_downloading_it(
                self, publication: DifferentialPublication
            ) -> None:
                assert (list_downloads(publication.hub), list_committed(publication.hub)) == (
                    [],
                    [["encoder-model.onnx", "README.md"]],
                )
