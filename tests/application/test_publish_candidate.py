import contextlib
import hashlib
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from functools import partial

import pytest
from accepted_build import (
    BUILD_DIRECTORY,
    ENCODER,
    ENCODER_NAME,
    HEAD_TREE,
    PEAKS,
    VOCABULARY,
    WORK_ROOT,
    build_files,
)
from card_arrangement import CARD_PATH, FakeClone, build_card_request
from run_step_fakes import FakeContainerEngine, FakeStepFiles, build_file_facts, build_step_output

from mostt_model_builder.application.prepare_publication import (
    PublicationError,
    prepare_publication,
    write_model_card,
)
from mostt_model_builder.application.publish_candidate import (
    CandidatePublication,
    PublishedCandidate,
    publish_candidate,
)
from mostt_model_builder.application.run_step import ImageFacts
from mostt_model_builder.application.verify_build import BuildRejectedError
from mostt_model_builder.domain.containers.container_steps import HUB_IMAGE
from mostt_model_builder.domain.containers.step_peaks import PEAKS_FILE_NAME
from mostt_model_builder.domain.publication.hub_token import HubToken, parse_hub_token
from mostt_model_builder.domain.publication.publication_report import (
    PublicationReport,
    PublicationReportError,
    ReportedFile,
    render_publication_report,
)
from mostt_model_builder.domain.publication.repository_files import RepositoryFilesError

REPOSITORY = "example-user/parakeet-ultra-onnx"
PARENT_REVISION = "0123456789abcdef0123456789abcdef01234567"
REVISION = "89abcdef0123456789abcdef0123456789abcdef"
TOKEN = parse_hub_token("hf_placeholder0123456789")
PUBLICATIONS_DIRECTORY = WORK_ROOT / BUILD_DIRECTORY / "publications"
IMAGE_ID = "sha256:" + "e" * 64
MISTYPED_REPOSITORY = "example-user/parakeet-ultra-onx"


REPOSITORY_ASKED = "the repository asked"
TOKEN_ASKED = "the token asked"
UPLOAD_HEADER = f"publish into {REPOSITORY}:"
ENCODER_LINE = (
    f"  {ENCODER_NAME}, {len(ENCODER)} bytes, sha256 {hashlib.sha256(ENCODER).hexdigest()}"
)


@dataclass
class FakePrompt:
    typed_repository: str = REPOSITORY
    events: list[str] = field(default_factory=list)

    def announce(self, lines: Sequence[str]) -> None:
        self.events.extend(lines)

    def ask_repository(self) -> str:
        self.events.append(REPOSITORY_ASKED)
        return self.typed_repository

    def ask_token(self) -> HubToken:
        self.events.append(TOKEN_ASKED)
        return TOKEN


@dataclass(frozen=True)
class Publication:
    published: PublishedCandidate
    engine: FakeContainerEngine
    files: FakeStepFiles
    prompt: FakePrompt


def describe(name: str, content: bytes, is_uploaded: bool = True) -> ReportedFile:
    return ReportedFile(name, len(content), hashlib.sha256(content).hexdigest(), is_uploaded)


def build_reviewed_files() -> tuple[FakeStepFiles, bytes]:
    files = build_files()
    prepared = prepare_publication(build_card_request(), FakeClone(), files)
    write_model_card(prepared, WORK_ROOT, files)
    return files, prepared.card


def build_report(
    card: bytes, encoder: bytes = ENCODER, is_uploaded: bool = True, revision: str = REVISION
) -> PublicationReport:
    return PublicationReport(
        REPOSITORY,
        True,
        PARENT_REVISION,
        revision,
        (
            describe(ENCODER_NAME, encoder, is_uploaded),
            describe("vocab.txt", VOCABULARY, is_uploaded),
            describe("README.md", card, is_uploaded),
        ),
    )


def build_engine(files: FakeStepFiles, report: PublicationReport) -> FakeContainerEngine:
    return FakeContainerEngine(
        files=files,
        images={HUB_IMAGE.tag: ImageFacts(IMAGE_ID, HEAD_TREE)},
        step_outputs={
            "publication.json": build_step_output(render_publication_report(report)),
            PEAKS_FILE_NAME: build_step_output(PEAKS),
        },
    )


def publish_reviewed_card(
    prompt: FakePrompt | None = None,
    encoder: bytes = ENCODER,
    is_uploaded: bool = True,
    revision: str = REVISION,
) -> Publication:
    files, card = build_reviewed_files()
    engine = build_engine(files, build_report(card, encoder, is_uploaded, revision))
    prompt = prompt or FakePrompt()
    published = publish_candidate(
        CandidatePublication(build_card_request(), REPOSITORY), FakeClone(), prompt, engine, files
    )
    return Publication(published, engine, files, prompt)


class TestTheCandidatePublication:
    class TestThePublishCandidateFunction:
        class TestWhenTheReviewedCardMatchesAndTheUploadAgrees:
            @pytest.fixture
            def publication(self) -> Publication:
                return publish_reviewed_card()

            def test_should_report_the_published_revision(self, publication: Publication) -> None:
                assert publication.published.report.revision == REVISION

            def test_should_give_the_token_to_the_engine_alone(
                self, publication: Publication
            ) -> None:
                secret = TOKEN.encode_secret()
                assert (
                    publication.engine.tokens,
                    [
                        path
                        for path, content in publication.files.contents.items()
                        if secret in content
                    ],
                    [
                        run
                        for run in publication.engine.runs
                        if any(secret.decode() in setting for _, setting in run.environment)
                    ],
                ) == ([TOKEN], [], [])

            def test_should_write_the_request_and_the_card_alone_into_its_request(
                self, publication: Publication
            ) -> None:
                request_directory = PUBLICATIONS_DIRECTORY / "1" / "request"
                assert publication.files.list_children(request_directory) == (
                    "README.md",
                    "publication-request.json",
                )

            def test_should_send_the_card_that_was_reviewed(self, publication: Publication) -> None:
                assert (
                    publication.files.contents[
                        PUBLICATIONS_DIRECTORY / "1" / "request" / "README.md"
                    ]
                    == publication.files.contents[CARD_PATH]
                )

            def test_should_show_the_files_before_it_asks_for_the_repository_and_the_token(
                self, publication: Publication
            ) -> None:
                ordered_events = (UPLOAD_HEADER, ENCODER_LINE, REPOSITORY_ASKED, TOKEN_ASKED)
                assert [
                    event for event in publication.prompt.events if event in ordered_events
                ] == list(ordered_events)

        class TestWhenAPublicationRanBefore:
            @pytest.fixture
            def publication(self) -> Publication:
                files, card = build_reviewed_files()
                files.add_directory(PUBLICATIONS_DIRECTORY / "1")
                engine = build_engine(files, build_report(card))
                published = publish_candidate(
                    CandidatePublication(build_card_request(), REPOSITORY),
                    FakeClone(),
                    FakePrompt(),
                    engine,
                    files,
                )
                return Publication(published, engine, files, FakePrompt())

            def test_should_number_the_new_one_after_it(self, publication: Publication) -> None:
                assert publication.published.placement.output_directory == (
                    BUILD_DIRECTORY / "publications" / "2" / "output"
                )

        class TestWhenTheConfirmationDiffers:
            @pytest.fixture
            def publish_act(self) -> Callable[[], Publication]:
                return partial(publish_reviewed_card, FakePrompt(MISTYPED_REPOSITORY))

            def test_should_refuse_it(self, publish_act: Callable[[], Publication]) -> None:
                with pytest.raises(RepositoryFilesError, match="nothing was published"):
                    publish_act()

        class TestWhenTheConfirmationDiffersAndItsRefusalIsCaught:
            @pytest.fixture
            def prompt(self) -> FakePrompt:
                prompt = FakePrompt(MISTYPED_REPOSITORY)
                with contextlib.suppress(RepositoryFilesError):
                    publish_reviewed_card(prompt)
                return prompt

            def test_should_stop_before_it_asks_for_the_token(self, prompt: FakePrompt) -> None:
                assert prompt.events[-1] == REPOSITORY_ASKED

        class TestWhenTheHubAlreadyHeldEveryFile:
            @pytest.fixture
            def publication(self) -> Publication:
                return publish_reviewed_card(is_uploaded=False, revision=PARENT_REVISION)

            def test_should_accept_the_report_at_the_parents_revision(
                self, publication: Publication
            ) -> None:
                assert (
                    publication.published.report.revision,
                    publication.published.report.has_uploaded,
                ) == (PARENT_REVISION, False)

        class TestWhenAFileLeftInPlaceHasAnotherSha256:
            @pytest.fixture
            def publish_act(self) -> Callable[[], Publication]:
                return partial(
                    publish_reviewed_card,
                    encoder=ENCODER + b"\x00",
                    is_uploaded=False,
                    revision=PARENT_REVISION,
                )

            def test_should_refuse_its_report(self, publish_act: Callable[[], Publication]) -> None:
                with pytest.raises(
                    PublicationReportError, match=re.escape("the report gives 5 bytes of SHA-256")
                ):
                    publish_act()

        class TestWhenTheReportClaimsAnUploadAtItsParentsRevision:
            @pytest.fixture
            def publish_act(self) -> Callable[[], Publication]:
                return partial(publish_reviewed_card, revision=PARENT_REVISION)

            def test_should_refuse_it(self, publish_act: Callable[[], Publication]) -> None:
                with pytest.raises(
                    PublicationReportError,
                    match=re.escape(
                        "the report's revision is its parent, though it uploaded files"
                    ),
                ):
                    publish_act()

        class TestWhenTheContainerCommittedOtherBytes:
            @pytest.fixture
            def publish_act(self) -> Callable[[], Publication]:
                return partial(publish_reviewed_card, encoder=ENCODER + b"\x00")

            def test_should_refuse_its_report(self, publish_act: Callable[[], Publication]) -> None:
                with pytest.raises(
                    PublicationReportError,
                    match=re.escape("the report gives 5 bytes of SHA-256"),
                ):
                    publish_act()

        class TestWhenNoCardWasWrittenForReview:
            @pytest.fixture
            def publish_act(self) -> Callable[[], PublishedCandidate]:
                files = build_files()
                return partial(
                    publish_candidate,
                    CandidatePublication(build_card_request(), REPOSITORY),
                    FakeClone(),
                    FakePrompt(),
                    build_engine(files, build_report(b"")),
                    files,
                )

            def test_should_refuse_it_and_name_card(
                self, publish_act: Callable[[], PublishedCandidate]
            ) -> None:
                with pytest.raises(
                    PublicationError,
                    match=re.escape(
                        f"{BUILD_DIRECTORY}/README.md is missing: write it with ./model-build.sh "
                        "--work <dir> --recipe <name> card"
                    ),
                ):
                    publish_act()

        class TestWhenTheReviewedCardDiffers:
            @pytest.fixture
            def publish_act(self) -> Callable[[], PublishedCandidate]:
                files = build_files()
                files.add_file(CARD_PATH, build_file_facts(8, 0o444), b"# other\n")
                return partial(
                    publish_candidate,
                    CandidatePublication(build_card_request(), REPOSITORY),
                    FakeClone(),
                    FakePrompt(),
                    build_engine(files, build_report(b"")),
                    files,
                )

            def test_should_refuse_it(self, publish_act: Callable[[], PublishedCandidate]) -> None:
                with pytest.raises(
                    PublicationError,
                    match=re.escape("differs from the card the build's records give now"),
                ):
                    publish_act()

        class TestWhenTheRecipeDescribesNoModel:
            @pytest.fixture
            def publish_act(self) -> Callable[[], PublishedCandidate]:
                files = build_files()
                return partial(
                    publish_candidate,
                    CandidatePublication(build_card_request(model=None), REPOSITORY),
                    FakeClone(),
                    FakePrompt(),
                    build_engine(files, build_report(b"")),
                    files,
                )

            def test_should_refuse_it(self, publish_act: Callable[[], PublishedCandidate]) -> None:
                with pytest.raises(PublicationError, match="describes no model"):
                    publish_act()

        class TestWhenTheBuildIsRejected:
            @pytest.fixture
            def publish_act(self) -> Callable[[], PublishedCandidate]:
                files = build_files()
                return partial(
                    publish_candidate,
                    CandidatePublication(build_card_request(is_rejected=True), REPOSITORY),
                    FakeClone(),
                    FakePrompt(),
                    build_engine(files, build_report(b"")),
                    files,
                )

            def test_should_refuse_it(self, publish_act: Callable[[], PublishedCandidate]) -> None:
                with pytest.raises(BuildRejectedError, match="the build fails 1 of its rules"):
                    publish_act()

        class TestWhenTheRepositoryIsMalformed:
            @pytest.fixture
            def publish_act(self) -> Callable[[], PublishedCandidate]:
                files = build_files()
                return partial(
                    publish_candidate,
                    replace(
                        CandidatePublication(build_card_request(), REPOSITORY),
                        repository="<user>/parakeet-ultra-onnx",
                    ),
                    FakeClone(),
                    FakePrompt(),
                    build_engine(files, build_report(b"")),
                    files,
                )

            def test_should_refuse_it_first(
                self, publish_act: Callable[[], PublishedCandidate]
            ) -> None:
                with pytest.raises(RepositoryFilesError, match="'<user>/parakeet-ultra-onnx'"):
                    publish_act()
