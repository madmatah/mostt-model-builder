import contextlib
import hashlib
import json
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
from card_arrangement import CARD_PATH, MODEL, FakeClone, build_card_request
from run_step_fakes import FakeContainerEngine, FakeStepFiles, build_file_facts, build_step_output

from mostt_model_builder.application.check_published import RefusedListingError
from mostt_model_builder.application.fetch_sources import DownloadError
from mostt_model_builder.application.prepare_publication import (
    CardRequest,
    PublicationError,
    prepare_publication,
    write_model_card,
)
from mostt_model_builder.application.publish_candidate import (
    CandidatePublication,
    DryRunEnded,
    PublicationEvent,
    PublicationSummary,
    PublishedCandidate,
    SummarizedFile,
    SummaryUnavailable,
    publish_candidate,
)
from mostt_model_builder.application.run_step import ImageFacts
from mostt_model_builder.application.verify_build import BuildRejectedError
from mostt_model_builder.domain.containers.container_steps import HUB_IMAGE
from mostt_model_builder.domain.containers.step_peaks import PEAKS_FILE_NAME
from mostt_model_builder.domain.publication.commit_message import CommitMessage
from mostt_model_builder.domain.publication.git_blobs import compute_git_blob_id
from mostt_model_builder.domain.publication.hub_token import HubToken, parse_hub_token
from mostt_model_builder.domain.publication.publication_report import (
    PublicationReport,
    PublicationReportError,
    ReportedFile,
    render_publication_report,
)
from mostt_model_builder.domain.publication.publication_request import parse_publication_request
from mostt_model_builder.domain.publication.published_tree import PublishedTreeError
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


UNREACHABLE_HUB = "curl exited with the status 28 while reading the listing: timed out"
HEADER_AND_FILE_LINE_COUNT = 4
COMMIT_TITLE = (
    "Keep the encoder's lengths in float32 in float16 and bfloat16; measure five languages"
)
COMMIT_DESCRIPTION = (
    "The float16 and bfloat16 encoders of revision 9dcab459 computed the encoder's output\n"
    "lengths (encoded_lengths), and the lengths of its subsampling stages, in their own\n"
    "reduced precision. The decoder then ignored the last frames of most recordings, and\n"
    "onnx-asr could fail on a long one. This revision keeps that computation in float32;\n"
    "the float32 and int8 files are unchanged.\n"
    "\n"
    "The card now measures English, Latin American Spanish, French, Brazilian Portuguese\n"
    "and German on FLEURS."
)
COMMIT_MESSAGE = CommitMessage(COMMIT_TITLE, COMMIT_DESCRIPTION)
COMMIT_MESSAGE_BLOCK = [
    "with the commit message:",
    "  | Keep the encoder's lengths in float32 in float16 and bfloat16; measure five languages",
    "  |",
    "  | The float16 and bfloat16 encoders of revision 9dcab459 computed the encoder's output",
    "  | lengths (encoded_lengths), and the lengths of its subsampling stages, in their own",
    "  | reduced precision. The decoder then ignored the last frames of most recordings, and",
    "  | onnx-asr could fail on a long one. This revision keeps that computation in float32;",
    "  | the float32 and int8 files are unchanged.",
    "  |",
    "  | The card now measures English, Latin American Spanish, French, Brazilian Portuguese",
    "  | and German on FLEURS.",
]
CARD_REQUEST = build_card_request()
PARAKEET_CARD_REQUEST = build_card_request(model=replace(MODEL, title="Parakeet Ultra in ONNX"))
REQUEST_PATH = PUBLICATIONS_DIRECTORY / "1" / "request" / "publication-request.json"


@dataclass
class FakeListing:
    content: bytes | None = None
    urls: list[str] = field(default_factory=list)
    is_reachable: bool = True

    def read_listing(self, url: str, max_size_bytes: int) -> bytes:
        self.urls.append(url)
        if not self.is_reachable:
            raise DownloadError(UNREACHABLE_HUB)
        if self.content is None:
            raise RefusedListingError(f"curl exited with the status 22 while reading {url}")
        if len(self.content) > max_size_bytes:
            raise LookupError(f"the fake listing holds more than {max_size_bytes} bytes")
        return self.content


@dataclass
class FakePrompt:
    typed_repository: str = REPOSITORY
    events: list[str] = field(default_factory=list)
    publication_events: list[PublicationEvent] = field(default_factory=list)

    def record(self, event: PublicationEvent) -> None:
        self.events.append(type(event).__name__)
        self.publication_events.append(event)

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


def publish_with(
    publication: CandidatePublication,
    prompt: FakePrompt,
    listing: FakeListing,
    engine: FakeContainerEngine,
    files: FakeStepFiles,
) -> PublishedCandidate | None:
    return publish_candidate(
        publication, FakeClone(), prompt, listing, engine, files, prompt.record
    )


def publish_for_real(
    publication: CandidatePublication,
    clone: FakeClone,
    prompt: FakePrompt,
    engine: FakeContainerEngine,
    files: FakeStepFiles,
) -> PublishedCandidate:
    published = publish_candidate(
        publication, clone, prompt, FakeListing(), engine, files, prompt.record
    )
    if published is None:
        raise LookupError("the publication stopped before its container")
    return published


def describe(name: str, content: bytes, is_uploaded: bool = True) -> ReportedFile:
    return ReportedFile(name, len(content), hashlib.sha256(content).hexdigest(), is_uploaded)


def build_reviewed_files(card_request: CardRequest = CARD_REQUEST) -> tuple[FakeStepFiles, bytes]:
    files = build_files()
    prepared = prepare_publication(card_request, FakeClone(), files)
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
    commit_message: CommitMessage | None = None,
) -> Publication:
    files, card = build_reviewed_files()
    engine = build_engine(files, build_report(card, encoder, is_uploaded, revision))
    prompt = prompt or FakePrompt()
    published = publish_for_real(
        CandidatePublication(build_card_request(), REPOSITORY, commit_message=commit_message),
        FakeClone(),
        prompt,
        engine,
        files,
    )
    return Publication(published, engine, files, prompt)


def render_head_listing(*entries: dict[str, object]) -> bytes:
    return json.dumps(
        [{"type": "file", "path": ".gitattributes", "size": 1519, "oid": "f" * 40}, *entries]
    ).encode()


def list_file(name: str, content: bytes) -> dict[str, object]:
    return {"type": "file", "path": name, "size": len(content), "oid": compute_git_blob_id(content)}


@dataclass(frozen=True)
class SummarizedPublication:
    published: PublishedCandidate | None
    prompt: FakePrompt
    listing: FakeListing
    engine: FakeContainerEngine
    files: FakeStepFiles


def summarize_beside(
    listing: FakeListing,
    is_dry_run: bool = False,
    commit_message: CommitMessage | None = None,
    card_request: CardRequest = CARD_REQUEST,
) -> SummarizedPublication:
    files, card = build_reviewed_files(card_request)
    engine = build_engine(files, build_report(card))
    prompt = FakePrompt()
    published = publish_with(
        CandidatePublication(
            card_request,
            REPOSITORY,
            is_dry_run,
            commit_message,
        ),
        prompt,
        listing,
        engine,
        files,
    )
    return SummarizedPublication(published, prompt, listing, engine, files)


OTHER_VOCABULARY = VOCABULARY.replace(b"chat", b"chut")
ALIKE_AND_CHANGED_LISTING = render_head_listing(
    list_file(ENCODER_NAME, ENCODER), list_file("vocab.txt", OTHER_VOCABULARY)
)


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
                published = publish_for_real(
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
                    publish_for_real,
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
                    publish_for_real,
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
                    publish_for_real,
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
                    publish_for_real,
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
                    publish_for_real,
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

        class TestWhenItIsADryRunWithACommitMessage:
            @pytest.fixture
            def summarized(self) -> SummarizedPublication:
                return summarize_beside(
                    FakeListing(ALIKE_AND_CHANGED_LISTING),
                    is_dry_run=True,
                    commit_message=COMMIT_MESSAGE,
                )

            def test_should_show_the_message_after_the_files_then_the_summary_and_the_end(
                self, summarized: SummarizedPublication
            ) -> None:
                assert summarized.prompt.events[HEADER_AND_FILE_LINE_COUNT:] == [
                    *COMMIT_MESSAGE_BLOCK,
                    "PublicationSummary",
                    "DryRunEnded",
                ]

        class TestWhenItIsADryRunWithoutACommitMessage:
            @pytest.fixture
            def summarized(self) -> SummarizedPublication:
                return summarize_beside(
                    FakeListing(None), is_dry_run=True, card_request=PARAKEET_CARD_REQUEST
                )

            def test_should_show_the_title_from_the_cards_title_alone(
                self, summarized: SummarizedPublication
            ) -> None:
                assert summarized.prompt.events[HEADER_AND_FILE_LINE_COUNT:] == [
                    "with the commit message:",
                    "  | Publish Parakeet Ultra in ONNX",
                    "SummaryUnavailable",
                    "DryRunEnded",
                ]

        class TestWhenAPublicationCarriesACommitMessage:
            @pytest.fixture
            def publication(self) -> Publication:
                return publish_reviewed_card(commit_message=COMMIT_MESSAGE)

            def test_should_show_the_message_before_it_asks_for_the_repository_and_the_token(
                self, publication: Publication
            ) -> None:
                ordered_events = (
                    COMMIT_MESSAGE_BLOCK[0],
                    COMMIT_MESSAGE_BLOCK[-1],
                    REPOSITORY_ASKED,
                    TOKEN_ASKED,
                )
                assert [
                    event for event in publication.prompt.events if event in ordered_events
                ] == list(ordered_events)

            def test_should_write_the_messages_title_and_description_into_its_request(
                self, publication: Publication
            ) -> None:
                publication_request = parse_publication_request(
                    publication.files.contents[REQUEST_PATH]
                )
                assert (
                    publication_request.commit_message,
                    publication_request.commit_description,
                ) == (COMMIT_TITLE, COMMIT_DESCRIPTION)

        class TestWhenTheHubListsTheHeadWithoutAToken:
            @pytest.fixture
            def summarized(self) -> SummarizedPublication:
                return summarize_beside(FakeListing(ALIKE_AND_CHANGED_LISTING))

            def test_should_read_the_listing_of_the_main_branch(
                self, summarized: SummarizedPublication
            ) -> None:
                assert summarized.listing.urls == [
                    f"https://huggingface.co/api/models/{REPOSITORY}/tree/main"
                ]

            def test_should_show_the_summary_after_the_files_and_before_the_repository(
                self, summarized: SummarizedPublication
            ) -> None:
                ordered_events = (ENCODER_LINE, "PublicationSummary", REPOSITORY_ASKED)
                assert [
                    event for event in summarized.prompt.events if event in ordered_events
                ] == list(ordered_events)

            def test_should_keep_a_file_whose_blob_id_is_the_builds_and_send_the_others(
                self, summarized: SummarizedPublication
            ) -> None:
                summary = summarized.prompt.publication_events[0]
                card_size = len(summarized.files.contents[CARD_PATH])
                assert summary == PublicationSummary(
                    REPOSITORY,
                    (
                        SummarizedFile(ENCODER_NAME, len(ENCODER), is_kept=True),
                        SummarizedFile("vocab.txt", len(VOCABULARY), is_kept=False),
                        SummarizedFile("README.md", card_size, is_kept=False),
                    ),
                )

            def test_should_go_on_to_the_token_and_the_container(
                self, summarized: SummarizedPublication
            ) -> None:
                assert (TOKEN_ASKED in summarized.prompt.events, len(summarized.engine.runs)) == (
                    True,
                    1,
                )

        class TestWhenItIsADryRun:
            @pytest.fixture
            def summarized(self) -> SummarizedPublication:
                return summarize_beside(FakeListing(ALIKE_AND_CHANGED_LISTING), is_dry_run=True)

            def test_should_show_the_summary_then_say_nothing_was_published(
                self, summarized: SummarizedPublication
            ) -> None:
                assert summarized.prompt.events[-2:] == ["PublicationSummary", "DryRunEnded"]

            def test_should_ask_neither_the_repository_nor_the_token(
                self, summarized: SummarizedPublication
            ) -> None:
                assert {REPOSITORY_ASKED, TOKEN_ASKED} & set(summarized.prompt.events) == set()

            def test_should_run_no_container_and_write_no_publication(
                self, summarized: SummarizedPublication
            ) -> None:
                assert (
                    summarized.published,
                    summarized.engine.runs,
                    PUBLICATIONS_DIRECTORY in summarized.files.entries,
                ) == (None, [], False)

            def test_should_name_the_repository_in_its_last_line(
                self, summarized: SummarizedPublication
            ) -> None:
                assert summarized.prompt.publication_events[-1] == DryRunEnded(REPOSITORY)

        class TestWhenTheHubGivesNoListingWithoutAToken:
            @pytest.fixture
            def summarized(self) -> SummarizedPublication:
                return summarize_beside(FakeListing(None))

            def test_should_say_there_is_no_summary(
                self, summarized: SummarizedPublication
            ) -> None:
                assert summarized.prompt.publication_events == [SummaryUnavailable(REPOSITORY)]

            def test_should_still_ask_the_token_and_run_the_container(
                self, summarized: SummarizedPublication
            ) -> None:
                assert (TOKEN_ASKED in summarized.prompt.events, len(summarized.engine.runs)) == (
                    True,
                    1,
                )

        class TestWhenItIsADryRunAndTheHubGivesNoListing:
            @pytest.fixture
            def summarized(self) -> SummarizedPublication:
                return summarize_beside(FakeListing(None), is_dry_run=True)

            def test_should_say_why_there_is_no_summary_then_that_nothing_was_published(
                self, summarized: SummarizedPublication
            ) -> None:
                assert summarized.prompt.publication_events == [
                    SummaryUnavailable(REPOSITORY),
                    DryRunEnded(REPOSITORY),
                ]

        class TestWhenTheListingNeverArrives:
            @pytest.fixture
            def refused(self) -> tuple[Callable[[], object], FakePrompt]:
                files, card = build_reviewed_files()
                prompt = FakePrompt()
                return (
                    partial(
                        publish_with,
                        CandidatePublication(build_card_request(), REPOSITORY),
                        prompt,
                        FakeListing(is_reachable=False),
                        build_engine(files, build_report(card)),
                        files,
                    ),
                    prompt,
                )

            def test_should_fail_with_curls_message_before_it_asks_for_the_token(
                self, refused: tuple[Callable[[], object], FakePrompt]
            ) -> None:
                publish_act, prompt = refused
                with pytest.raises(DownloadError, match=re.escape(UNREACHABLE_HUB)):
                    publish_act()
                assert TOKEN_ASKED not in prompt.events

        @pytest.mark.parametrize(
            ("listing", "error", "reason"),
            [
                pytest.param(
                    render_head_listing({"type": "directory", "path": "onnx", "size": 0}),
                    PublishedTreeError,
                    "the revision holds the directory onnx",
                    id="a-directory",
                ),
                pytest.param(
                    render_head_listing(list_file("model.safetensors", b"weights")),
                    RepositoryFilesError,
                    "'model.safetensors'",
                    id="a-file-of-no-build",
                ),
                pytest.param(b"[{", PublishedTreeError, "not a JSON document", id="malformed"),
            ],
        )
        class TestWhenTheListingIsRefused:
            @pytest.fixture
            def refused(self, listing: bytes) -> tuple[Callable[[], object], FakePrompt]:
                files, card = build_reviewed_files()
                prompt = FakePrompt()
                return (
                    partial(
                        publish_with,
                        CandidatePublication(build_card_request(), REPOSITORY),
                        prompt,
                        FakeListing(listing),
                        build_engine(files, build_report(card)),
                        files,
                    ),
                    prompt,
                )

            def test_should_fail_before_it_asks_for_the_token(
                self,
                refused: tuple[Callable[[], object], FakePrompt],
                error: type[Exception],
                reason: str,
            ) -> None:
                publish_act, prompt = refused
                with pytest.raises(error, match=re.escape(reason)):
                    publish_act()
                assert TOKEN_ASKED not in prompt.events
