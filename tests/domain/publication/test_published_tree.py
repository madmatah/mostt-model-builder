import json
import random
import re
from collections.abc import Callable
from functools import partial

import pytest

from mostt_model_builder.domain.publication.published_tree import (
    ExpectedFile,
    ListedEntry,
    PublishedTreeError,
    judge_published_listing,
    judge_published_revision,
    parse_published_listing,
    pin_published_file,
)
from mostt_model_builder.domain.sources.source_pins import RepositoryType, SourcePin

REPOSITORY = "example-user/parakeet-ultra-onnx"
REVISION = "0123456789abcdef0123456789abcdef01234567"
PROPERTY_SEED = 20261104
EXPECTED_FILES = (
    ExpectedFile("encoder-model.onnx", 4096, "a" * 64),
    ExpectedFile("encoder-model.onnx.data", 2_476_376_064, "b" * 64),
    ExpectedFile("vocab.txt", 21, "c" * 64),
    ExpectedFile("README.md", 900, "d" * 64),
)
ATTRIBUTES_ENTRY = ListedEntry("file", ".gitattributes", 1519)
EXACT_ENTRIES = (
    ATTRIBUTES_ENTRY,
    *(ListedEntry("file", expected.name, expected.size_bytes) for expected in EXPECTED_FILES),
)


def render_listing(*entries: object) -> bytes:
    return json.dumps(list(entries)).encode()


def describe_entry(entry: ListedEntry) -> dict[str, object]:
    return {"type": entry.entry_type, "oid": "e" * 40, "size": entry.size_bytes, "path": entry.path}


EXACT_LISTING = render_listing(
    *(describe_entry(entry) for entry in EXACT_ENTRIES[:2]),
    {**describe_entry(EXACT_ENTRIES[2]), "lfs": {"oid": "b" * 64, "size": 2_476_376_064}},
    *(describe_entry(entry) for entry in EXACT_ENTRIES[3:]),
)
REFUSED_LISTINGS = {
    "a document that is not JSON": (b"[{", "the listing is not a JSON document"),
    "a document that is not UTF-8": (b'["\xff"]', "the listing is not a JSON document"),
    "a document nested beyond its depth": (b"[" * 9 + b"]" * 9, "nests deeper than 8 levels"),
    "a top level that is not an array": (b"{}", "the listing is not an array"),
    "more than 1024 entries": (
        render_listing(*([describe_entry(ATTRIBUTES_ENTRY)] * 1025)),
        "the listing holds more than 1024 entries",
    ),
    "a document beyond its size bound": (b" " * (2**20 + 1), "holds more than 1048576 bytes"),
    "an entry that is not an object": (render_listing("README.md"), "is not an object"),
    "an entry of an unknown type": (
        render_listing({"type": "link", "path": "README.md", "size": 1}),
        "has the type 'link'",
    ),
    "an entry without a path": (
        render_listing({"type": "file", "size": 1}),
        "has the path None",
    ),
    "a path that a reader cannot see whole": (
        render_listing({"type": "file", "path": "README‮md", "size": 1}),
        "has the path 'README\\u202emd'",
    ),
    "a size that is not a whole number": (
        render_listing({"type": "file", "path": "README.md", "size": 1.5}),
        "the entry README.md of the listing has the size 1.5",
    ),
    "a size that is a boolean": (
        render_listing({"type": "file", "path": "README.md", "size": True}),
        "the entry README.md of the listing has the size True",
    ),
}
REFUSED_TREES = {
    "a directory": (
        (*EXACT_ENTRIES, ListedEntry("directory", "onnx", 0)),
        "the revision holds the directory onnx, which the build never publishes",
    ),
    "a file of no build": (
        (*EXACT_ENTRIES, ListedEntry("file", "model.safetensors", 7)),
        "the revision holds model.safetensors, which is not a file of the build",
    ),
    "a missing file": (
        EXACT_ENTRIES[:-1],
        "the revision lacks README.md, a file of the build",
    ),
    "a size that differs": (
        (*EXACT_ENTRIES[:-1], ListedEntry("file", "README.md", 901)),
        "the revision holds 901 bytes of README.md, where the build holds 900",
    ),
    "a path listed twice": (
        (*EXACT_ENTRIES, EXACT_ENTRIES[-1]),
        "the listing names one path twice",
    ),
}


def perturb_listing(
    entries: tuple[ListedEntry, ...], generator: random.Random
) -> tuple[tuple[ListedEntry, ...], str]:
    index = generator.randrange(1, len(entries))
    perturbation = generator.choice(("drop", "add", "resize", "directory"))
    if perturbation == "drop":
        return (*entries[:index], *entries[index + 1 :]), perturbation
    if perturbation == "add":
        return (*entries, ListedEntry("file", f"extra-{index}.onnx", index)), perturbation
    if perturbation == "resize":
        entry = entries[index]
        resized = ListedEntry(entry.entry_type, entry.path, entry.size_bytes + 1)
        return (*entries[:index], resized, *entries[index + 1 :]), perturbation
    return (*entries, ListedEntry("directory", f"folder-{index}", 0)), perturbation


def is_refused(entries: tuple[ListedEntry, ...]) -> bool:
    try:
        judge_published_listing(entries, EXPECTED_FILES)
    except PublishedTreeError:
        return True
    return False


class TestThePublishedTree:
    class TestTheParsePublishedListingFunction:
        class TestWhenGivenTheHubsListingOfARevision:
            @pytest.fixture
            def entries(self) -> tuple[ListedEntry, ...]:
                return parse_published_listing(EXACT_LISTING)

            def test_should_read_the_type_path_and_size_of_each_entry_and_skip_the_rest(
                self, entries: tuple[ListedEntry, ...]
            ) -> None:
                assert entries == EXACT_ENTRIES

        @pytest.mark.parametrize(
            ("listing", "reason"),
            list(REFUSED_LISTINGS.values()),
            ids=list(REFUSED_LISTINGS.keys()),
        )
        class TestWhenTheListingIsMalformed:
            @pytest.fixture
            def parse_act(self, listing: bytes) -> Callable[[], tuple[ListedEntry, ...]]:
                return partial(parse_published_listing, listing)

            def test_should_refuse_it_and_say_why(
                self, parse_act: Callable[[], tuple[ListedEntry, ...]], reason: str
            ) -> None:
                with pytest.raises(PublishedTreeError, match=re.escape(reason)):
                    parse_act()

    class TestTheJudgePublishedListingFunction:
        class TestWhenTheRevisionHoldsExactlyTheBuildsFiles:
            @pytest.fixture
            def is_listing_refused(self) -> bool:
                return is_refused(EXACT_ENTRIES)

            def test_should_accept_it_beside_the_hubs_attributes_file(
                self, is_listing_refused: bool
            ) -> None:
                assert is_listing_refused is False

        @pytest.mark.parametrize(
            ("entries", "reason"),
            list(REFUSED_TREES.values()),
            ids=list(REFUSED_TREES.keys()),
        )
        class TestWhenTheRevisionDiffersFromTheBuild:
            @pytest.fixture
            def judge_act(self, entries: tuple[ListedEntry, ...]) -> Callable[[], None]:
                return partial(judge_published_listing, entries, EXPECTED_FILES)

            def test_should_refuse_it_and_say_why(
                self, judge_act: Callable[[], None], reason: str
            ) -> None:
                with pytest.raises(PublishedTreeError, match=re.escape(reason)):
                    judge_act()

        class TestWhenGivenAnyListingOfThePublishedFiles:
            def test_should_refuse_every_listing_perturbed_in_one_way(self) -> None:
                generator = random.Random(PROPERTY_SEED)
                perturbed = [perturb_listing(EXACT_ENTRIES, generator) for _ in range(200)]
                accepted = [
                    (entries, perturbation)
                    for entries, perturbation in perturbed
                    if not is_refused(entries)
                ]
                assert (accepted, len(perturbed)) == ([], 200), f"seed {PROPERTY_SEED}"

    class TestTheJudgePublishedRevisionFunction:
        class TestWhenTheRepositoryIsOutsideItsGrammar:
            @pytest.fixture
            def judge_act(self) -> Callable[[], None]:
                return partial(judge_published_revision, "<user>/parakeet-ultra-onnx", REVISION)

            def test_should_refuse_it_before_any_request(
                self, judge_act: Callable[[], None]
            ) -> None:
                with pytest.raises(
                    PublishedTreeError,
                    match=re.escape("the repository '<user>/parakeet-ultra-onnx'"),
                ):
                    judge_act()

        class TestWhenTheRevisionIsABranch:
            @pytest.fixture
            def judge_act(self) -> Callable[[], None]:
                return partial(judge_published_revision, REPOSITORY, "main")

            def test_should_refuse_it_since_a_branch_moves(
                self, judge_act: Callable[[], None]
            ) -> None:
                with pytest.raises(
                    PublishedTreeError,
                    match=re.escape("the revision 'main' is not a commit of 40 hexadecimal digits"),
                ):
                    judge_act()

    class TestThePinPublishedFileFunction:
        class TestWhenGivenAFileOfTheBuild:
            @pytest.fixture
            def source_pin(self) -> SourcePin:
                return pin_published_file(REPOSITORY, REVISION, EXPECTED_FILES[0])

            def test_should_pin_it_at_the_published_revision_by_its_size_and_sha256(
                self, source_pin: SourcePin
            ) -> None:
                assert source_pin == SourcePin(
                    RepositoryType.MODEL, REPOSITORY, REVISION, "encoder-model.onnx", 4096, "a" * 64
                )
