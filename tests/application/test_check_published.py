import contextlib
import hashlib
import json
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from functools import partial
from pathlib import PurePosixPath

import pytest
from accepted_build import (
    BUILD_DIRECTORY,
    ENCODER,
    ENCODER_NAME,
    RECIPE_NAME,
    VOCABULARY,
    WORK_ROOT,
    build_files,
    build_request,
)
from run_step_fakes import FakeStepFiles, build_file_facts
from source_fakes import FakeSourceFiles

from mostt_model_builder.application.check_published import (
    CheckedFile,
    PublishedCheckError,
    PublishedCheckRequest,
    check_published,
)
from mostt_model_builder.application.fetch_sources import DownloadError, FetchError
from mostt_model_builder.domain.contract.model_description import (
    ModelDescription,
    VariantDescription,
)
from mostt_model_builder.domain.publication.git_blobs import compute_git_blob_id
from mostt_model_builder.domain.publication.published_tree import (
    PUBLISHED_LISTING_MAX_SIZE_BYTES,
    PublishedTreeError,
)

REPOSITORY = "example-user/parakeet-ultra-onnx"
REVISION = "0123456789abcdef0123456789abcdef01234567"
CARD = b'---\nlicense: "cc-by-4.0"\n---\n# Weights in ONNX\n'
CARD_PATH = WORK_ROOT / BUILD_DIRECTORY / "README.md"
TREE_URL = f"https://huggingface.co/api/models/{REPOSITORY}/tree/{REVISION}"
PUBLISHED = {"README.md": CARD, ENCODER_NAME: ENCODER, "vocab.txt": VOCABULARY}
MODEL = ModelDescription(
    title="Weights in ONNX",
    template_path=PurePosixPath("model-card.md"),
    measuring_library_names=("onnx-asr",),
    variants=(VariantDescription("float32", "float32", "The exported graph."),),
)


def render_listing(published: Mapping[str, bytes]) -> bytes:
    return json.dumps(
        [{"type": "file", "path": ".gitattributes", "size": 1519, "oid": "f" * 40}]
        + [
            {
                "type": "file",
                "path": name,
                "size": len(content),
                "oid": compute_git_blob_id(content),
            }
            for name, content in published.items()
        ]
    ).encode()


@dataclass
class FakeTreeListing:
    listing: bytes
    urls: list[str] = field(default_factory=list)

    def read_listing(self, url: str, max_size_bytes: int) -> bytes:
        self.urls.append(url)
        if len(self.listing) > max_size_bytes:
            raise DownloadError(f"the listing at {url} holds more than {max_size_bytes} bytes")
        return self.listing


@dataclass
class FakeHubDownloader:
    source_files: FakeSourceFiles
    served: Mapping[str, bytes]
    urls: list[str] = field(default_factory=list)

    def download(self, url: str, partial_path: PurePosixPath, size_bytes: int) -> None:
        self.urls.append(url)
        self.source_files.write(partial_path, self.served[url.rsplit("/", 1)[-1]][:size_bytes])


@dataclass(frozen=True)
class PublishedCheck:
    checked_files: tuple[CheckedFile, ...]
    listing: FakeTreeListing
    downloader: FakeHubDownloader
    source_files: FakeSourceFiles


def build_check_files(card: bytes | None = CARD) -> FakeStepFiles:
    files = build_files()
    if card is not None:
        files.add_file(CARD_PATH, build_file_facts(len(card), 0o444), card)
    return files


def build_check_request(
    repository: str = REPOSITORY, revision: str = REVISION, model: ModelDescription | None = MODEL
) -> PublishedCheckRequest:
    return PublishedCheckRequest(
        RECIPE_NAME, repository, revision, build_request().candidate, model
    )


def run_check(
    published: Mapping[str, bytes] = PUBLISHED,
    served: Mapping[str, bytes] = PUBLISHED,
    request: PublishedCheckRequest | None = None,
    files: FakeStepFiles | None = None,
    listing_content: bytes | None = None,
) -> PublishedCheck:
    source_files = FakeSourceFiles(WORK_ROOT)
    listing = FakeTreeListing(listing_content or render_listing(published))
    downloader = FakeHubDownloader(source_files, served)
    checked_files = check_published(
        request or build_check_request(),
        listing,
        downloader,
        files or build_check_files(),
        source_files,
    )
    return PublishedCheck(checked_files, listing, downloader, source_files)


@dataclass(frozen=True)
class RefusedCheck:
    listing: FakeTreeListing
    downloader: FakeHubDownloader
    source_files: FakeSourceFiles


def run_refused_check(
    published: Mapping[str, bytes] = PUBLISHED,
    served: Mapping[str, bytes] = PUBLISHED,
    request: PublishedCheckRequest | None = None,
) -> RefusedCheck:
    source_files = FakeSourceFiles(WORK_ROOT)
    listing = FakeTreeListing(render_listing(published))
    downloader = FakeHubDownloader(source_files, served)
    with contextlib.suppress(PublishedTreeError, FetchError):
        check_published(
            request or build_check_request(),
            listing,
            downloader,
            build_check_files(),
            source_files,
        )
    return RefusedCheck(listing, downloader, source_files)


class TestThePublishedCheck:
    class TestTheCheckPublishedFunction:
        class TestWhenTheRevisionHoldsExactlyTheBuildsFilesAndCard:
            @pytest.fixture
            def published_check(self) -> PublishedCheck:
                return run_check()

            def test_should_read_the_listing_of_the_revision_on_the_hub(
                self, published_check: PublishedCheck
            ) -> None:
                assert published_check.listing.urls == [TREE_URL]

            def test_should_download_each_file_at_the_revision_without_a_token(
                self, published_check: PublishedCheck
            ) -> None:
                assert published_check.downloader.urls == [
                    f"https://huggingface.co/{REPOSITORY}/resolve/{REVISION}/{name}"
                    for name in (ENCODER_NAME, "vocab.txt", "README.md")
                ]

            def test_should_keep_each_download_once_its_sha256_matches_the_builds(
                self, published_check: PublishedCheck
            ) -> None:
                assert [
                    (checked.name, checked.sha256) for checked in published_check.checked_files
                ] == [
                    (name, hashlib.sha256(PUBLISHED[name]).hexdigest())
                    for name in (ENCODER_NAME, "vocab.txt", "README.md")
                ]

            def test_should_keep_the_downloads_beside_the_sources_at_the_revision(
                self, published_check: PublishedCheck
            ) -> None:
                assert published_check.checked_files[0].source_path == PurePosixPath(
                    f"sources/{REPOSITORY}/{REVISION}/{ENCODER_NAME}"
                )

        class TestWhenTheRevisionHoldsAnExtraFile:
            @pytest.fixture
            def check_act(self) -> Callable[[], PublishedCheck]:
                return partial(run_check, {**PUBLISHED, "model.safetensors": b"x"})

            def test_should_refuse_it(self, check_act: Callable[[], PublishedCheck]) -> None:
                with pytest.raises(
                    PublishedTreeError,
                    match=re.escape("the revision holds model.safetensors, which is not a file"),
                ):
                    check_act()

        class TestWhenTheRevisionLacksAFile:
            @pytest.fixture
            def check_act(self) -> Callable[[], PublishedCheck]:
                return partial(
                    run_check, {name: PUBLISHED[name] for name in (ENCODER_NAME, "vocab.txt")}
                )

            def test_should_refuse_it(self, check_act: Callable[[], PublishedCheck]) -> None:
                with pytest.raises(
                    PublishedTreeError,
                    match=re.escape("the revision lacks README.md, a file of the build"),
                ):
                    check_act()

        class TestWhenAPublishedSizeDiffers:
            @pytest.fixture
            def check_act(self) -> Callable[[], PublishedCheck]:
                return partial(run_check, {**PUBLISHED, "vocab.txt": VOCABULARY + b"x"})

            def test_should_refuse_it(self, check_act: Callable[[], PublishedCheck]) -> None:
                with pytest.raises(PublishedTreeError, match=re.escape("bytes of vocab.txt")):
                    check_act()

        class TestWhenAPublishedSizeDiffersAndTheRefusalIsCaught:
            @pytest.fixture
            def refused_check(self) -> RefusedCheck:
                return run_refused_check({**PUBLISHED, "vocab.txt": VOCABULARY + b"x"})

            def test_should_download_nothing(self, refused_check: RefusedCheck) -> None:
                assert refused_check.downloader.urls == []

        class TestWhenADownloadHasAnotherSha256:
            @pytest.fixture
            def check_act(self) -> Callable[[], PublishedCheck]:
                served = {**PUBLISHED, "vocab.txt": VOCABULARY[:-1] + b"X"}
                return partial(run_check, PUBLISHED, served)

            def test_should_refuse_it_and_say_it_was_removed(
                self, check_act: Callable[[], PublishedCheck]
            ) -> None:
                with pytest.raises(
                    FetchError,
                    match=re.escape(
                        f"the download of sources/{REPOSITORY}/{REVISION}/vocab.txt does not match "
                        "its pin, so it was removed"
                    ),
                ):
                    check_act()

        class TestWhenADownloadHasAnotherSha256AndTheRefusalIsCaught:
            @pytest.fixture
            def refused_check(self) -> RefusedCheck:
                return run_refused_check(served={**PUBLISHED, "vocab.txt": VOCABULARY[:-1] + b"X"})

            def test_should_keep_no_file_of_it_among_the_sources(
                self, refused_check: RefusedCheck
            ) -> None:
                assert (
                    [
                        path
                        for path in refused_check.source_files.contents
                        if "vocab.txt" in path.name
                    ],
                    len(refused_check.downloader.urls),
                ) == ([], 2)

        class TestWhenTheListingExceedsItsBound:
            @pytest.fixture
            def check_act(self) -> Callable[[], PublishedCheck]:
                return partial(
                    run_check, listing_content=b" " * (PUBLISHED_LISTING_MAX_SIZE_BYTES + 1)
                )

            def test_should_refuse_it_as_the_download_of_the_listing(
                self, check_act: Callable[[], PublishedCheck]
            ) -> None:
                with pytest.raises(
                    DownloadError,
                    match=re.escape(f"holds more than {PUBLISHED_LISTING_MAX_SIZE_BYTES} bytes"),
                ):
                    check_act()

        class TestWhenTheCardIsMissing:
            @pytest.fixture
            def check_act(self) -> Callable[[], PublishedCheck]:
                return partial(run_check, files=build_check_files(card=None))

            def test_should_refuse_it_and_name_card(
                self, check_act: Callable[[], PublishedCheck]
            ) -> None:
                with pytest.raises(
                    PublishedCheckError,
                    match=re.escape(
                        f"{BUILD_DIRECTORY}/README.md is missing, or is not the card that card "
                        "writes: write it with ./model-build.sh --work <dir> --recipe <name> card"
                    ),
                ):
                    check_act()

        class TestWhenTheRecipeDescribesNoModel:
            @pytest.fixture
            def check_act(self) -> Callable[[], PublishedCheck]:
                return partial(run_check, request=build_check_request(model=None))

            def test_should_refuse_it(self, check_act: Callable[[], PublishedCheck]) -> None:
                with pytest.raises(
                    PublishedCheckError,
                    match=re.escape("the recipe family describes no model, so it publishes none"),
                ):
                    check_act()

        @pytest.mark.parametrize(
            ("repository", "revision", "reason"),
            [
                pytest.param(
                    "example-user", REVISION, "the repository 'example-user'", id="a repository"
                ),
                pytest.param(
                    REPOSITORY, "main", "the revision 'main' is not a commit", id="a revision"
                ),
            ],
        )
        class TestWhenAValueOfTheCommandLineIsMalformed:
            @pytest.fixture
            def check_act(self, repository: str, revision: str) -> Callable[[], PublishedCheck]:
                return partial(run_check, request=build_check_request(repository, revision))

            def test_should_refuse_it(
                self, check_act: Callable[[], PublishedCheck], reason: str
            ) -> None:
                with pytest.raises(PublishedTreeError, match=re.escape(reason)):
                    check_act()

        @pytest.mark.parametrize(
            ("repository", "revision"),
            [
                pytest.param("example-user", REVISION, id="a repository"),
                pytest.param(REPOSITORY, "main", id="a revision"),
            ],
        )
        class TestWhenAValueOfTheCommandLineIsMalformedAndTheRefusalIsCaught:
            @pytest.fixture
            def refused_check(self, repository: str, revision: str) -> RefusedCheck:
                return run_refused_check(request=build_check_request(repository, revision))

            def test_should_send_no_request(self, refused_check: RefusedCheck) -> None:
                assert (refused_check.listing.urls, refused_check.downloader.urls) == ([], [])
