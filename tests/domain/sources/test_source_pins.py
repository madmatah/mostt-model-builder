import json
import re
from collections.abc import Callable, Mapping
from functools import partial

import pytest

from mostt_model_builder.domain.formats.file_names import REFUSED_SUFFIXES
from mostt_model_builder.domain.sources.source_pins import (
    MAX_SIZE_BYTES,
    REFUSED_SOURCE_SUFFIXES,
    RepositoryType,
    SourcePin,
    SourcePinError,
    is_pinned_content,
    parse_source_pins,
)

type PinFields = Mapping[str, object]

REVISION = "70bb2e84b976b7e960aa89f1c648e09c59f894dd"
OTHER_REVISION = "0123456789abcdef0123456789abcdef01234567"
SHA256 = "d23690e102f373554d1b544cd2ff1e76e4fedeb04953c0b72751a1b7c518cfdd"
SIZE_BYTES = 349036055
GOOD_PIN: PinFields = {
    "repository_type": "dataset",
    "repository": "google/fleurs",
    "revision": REVISION,
    "path": "data/fr_fr/audio/test.tar.gz",
    "size_bytes": SIZE_BYTES,
    "sha256": SHA256,
}
GOOD_SOURCE_PIN = SourcePin(
    repository_type=RepositoryType.DATASET,
    repository="google/fleurs",
    revision=REVISION,
    path="data/fr_fr/audio/test.tar.gz",
    size_bytes=SIZE_BYTES,
    sha256=SHA256,
)


def render_toml_value(field: object) -> str:
    if isinstance(field, bool):
        return "true" if field else "false"
    if isinstance(field, int | float):
        return repr(field)
    return json.dumps(field)


def render_sources(*pins: PinFields) -> str:
    return "".join(
        "[[sources]]\n"
        + "".join(f"{key} = {render_toml_value(field)}\n" for key, field in pin.items())
        for pin in pins
    )


def change_pin(**fields: object) -> PinFields:
    return {**GOOD_PIN, **fields}


def drop_key(key: str) -> PinFields:
    return {name: field for name, field in GOOD_PIN.items() if name != key}


def match_exactly(message: str) -> str:
    return f"^{re.escape(message)}$"


DEEP_NESTING_DEPTH = 3000

REFUSED_DOCUMENTS = {
    "text that is not TOML": ("[[sources]\n", "the sources are not valid TOML"),
    "an array nested 3 000 deep": (
        "value = " + "[" * DEEP_NESTING_DEPTH,
        "the sources are not valid TOML",
    ),
    "an unknown top-level key": (
        f"schema = 1\n{render_sources(GOOD_PIN)}",
        "the sources: unknown key 'schema'",
    ),
    "no sources": ("", "the sources: missing key sources"),
    "sources that are not an array": ('sources = "fleurs"\n', "sources must be a non-empty"),
    "an empty array of sources": ("sources = []\n", "sources must be a non-empty"),
    "a pin that is not a table": ("sources = [1]\n", "sources[0] must be a table"),
}
REFUSED_PINS = {
    "an unknown key": (change_pin(license="CC-BY-4.0"), "sources[0]: unknown key 'license'"),
    **{f"a missing {key}": (drop_key(key), f"sources[0]: missing key {key}") for key in GOOD_PIN},
    "an unknown repository type": (
        change_pin(repository_type="space"),
        "sources[0].repository_type: 'space' is not one of model, dataset",
    ),
    "a repository without its namespace": (
        change_pin(repository="fleurs"),
        "sources[0].repository: 'fleurs' does not match",
    ),
    "a repository with three segments": (
        change_pin(repository="google/fleurs/data"),
        "sources[0].repository: 'google/fleurs/data' does not match",
    ),
    "a repository that starts with a dot": (
        change_pin(repository="google/..fleurs"),
        "sources[0].repository: 'google/..fleurs' does not match",
    ),
    "a repository name of 97 characters": (
        change_pin(repository=f"google/{'a' * 97}"),
        "sources[0].repository: ",
    ),
    "a repository with a space": (
        change_pin(repository="google/fle urs"),
        "sources[0].repository: 'google/fle urs' does not match",
    ),
    "a branch as the revision": (
        change_pin(revision="main"),
        "sources[0].revision: 'main' does not match",
    ),
    "a revision in uppercase": (
        change_pin(revision=REVISION.upper()),
        "sources[0].revision: ",
    ),
    "a revision of 39 digits": (change_pin(revision=REVISION[:-1]), "sources[0].revision: "),
    "an empty path": (change_pin(path=""), "sources[0].path: the segment '' of '' is not a name"),
    "a path with an empty segment": (
        change_pin(path="data//test.tsv"),
        "sources[0].path: the segment '' of 'data//test.tsv' is not a name",
    ),
    "an absolute path": (
        change_pin(path="/data/test.tsv"),
        "sources[0].path: the segment '' of '/data/test.tsv' is not a name",
    ),
    "a path with a dot segment": (
        change_pin(path="data/./test.tsv"),
        "sources[0].path: the segment '.' of 'data/./test.tsv' is not a name",
    ),
    "a path that climbs": (
        change_pin(path="../test.tsv"),
        "sources[0].path: the segment '..' of '../test.tsv' is not a name",
    ),
    "a path with a backslash": (
        change_pin(path="data\\test.tsv"),
        "sources[0].path: the segment 'data\\\\test.tsv' of 'data\\\\test.tsv' is not a name",
    ),
    "a path with a character beyond ascii": (
        change_pin(path="données/test.tsv"),
        "sources[0].path: the segment 'donn\\xe9es' of 'donn\\xe9es/test.tsv' is not a name",
    ),
    "a path that is not a string": (change_pin(path=1), "sources[0].path: 1 is not a string"),
    "a size of 0": (change_pin(size_bytes=0), "sources[0].size_bytes: 0 is not a whole number"),
    "a negative size": (
        change_pin(size_bytes=-1),
        "sources[0].size_bytes: -1 is not a whole number",
    ),
    "a size beyond the bound": (
        change_pin(size_bytes=MAX_SIZE_BYTES + 1),
        f"sources[0].size_bytes: {MAX_SIZE_BYTES + 1} is not a whole number",
    ),
    "a size as a string": (
        change_pin(size_bytes=str(SIZE_BYTES)),
        f"sources[0].size_bytes: '{SIZE_BYTES}' is not a whole number",
    ),
    "a size as a boolean": (
        change_pin(size_bytes=True),
        "sources[0].size_bytes: True is not a whole number",
    ),
    "a size as a float": (
        change_pin(size_bytes=1.0),
        "sources[0].size_bytes: 1.0 is not a whole number",
    ),
    "a hash in uppercase": (change_pin(sha256=SHA256.upper()), "sources[0].sha256: "),
    "a hash of 63 digits": (change_pin(sha256=SHA256[:-1]), "sources[0].sha256: "),
    "a hash with its algorithm": (change_pin(sha256=f"sha256:{SHA256}"), "sources[0].sha256: "),
    "a mark of a file that runs code as a string": (
        change_pin(runs_code_when_loaded="true"),
        "sources[0].runs_code_when_loaded: 'true' is not true or false",
    ),
    "a mark of a file that runs code as a number": (
        change_pin(runs_code_when_loaded=1),
        "sources[0].runs_code_when_loaded: 1 is not true or false",
    ),
}
NEMO_ARCHIVE_PATH = "parakeet-tdt-0.6b-v3.nemo"
OVERLAPPING_PINS = {
    "the same file pinned twice": (
        (GOOD_PIN, GOOD_PIN),
        "data/fr_fr/audio/test.tar.gz of google/fleurs is pinned twice",
    ),
    "the same file as a model and as a dataset": (
        (GOOD_PIN, change_pin(repository_type="model")),
        "data/fr_fr/audio/test.tar.gz of google/fleurs is pinned twice",
    ),
    "a file under the file of another pin": (
        (change_pin(path="data/fr_fr"), GOOD_PIN),
        "data/fr_fr/audio/test.tar.gz of google/fleurs would lie under the file of another pin",
    ),
}
DISTINCT_PINS = {
    "two files of one repository": (GOOD_PIN, change_pin(path="data/fr_fr/test.tsv")),
    "one file at two revisions": (GOOD_PIN, change_pin(revision=OTHER_REVISION)),
}


class TestTheSourcePins:
    class TestTheParseSourcePinsFunction:
        class TestWhenThePinIsWellFormed:
            @pytest.fixture
            def source_pins(self) -> tuple[SourcePin, ...]:
                return parse_source_pins(render_sources(GOOD_PIN))

            def test_should_read_every_field(self, source_pins: tuple[SourcePin, ...]) -> None:
                assert source_pins == (GOOD_SOURCE_PIN,)

        class TestWhenTheRepositoryIsAModel:
            @pytest.fixture
            def source_pins(self) -> tuple[SourcePin, ...]:
                return parse_source_pins(render_sources(change_pin(repository_type="model")))

            def test_should_read_its_type(self, source_pins: tuple[SourcePin, ...]) -> None:
                assert source_pins[0].repository_type is RepositoryType.MODEL

        @pytest.mark.parametrize(
            ("text", "message_start"),
            list(REFUSED_DOCUMENTS.values()),
            ids=list(REFUSED_DOCUMENTS.keys()),
        )
        class TestWhenTheDocumentIsMalformed:
            @pytest.fixture
            def parse_act(self, text: str) -> Callable[[], tuple[SourcePin, ...]]:
                return partial(parse_source_pins, text)

            def test_should_refuse_it_and_say_why(
                self, parse_act: Callable[[], tuple[SourcePin, ...]], message_start: str
            ) -> None:
                with pytest.raises(SourcePinError, match=f"^{re.escape(message_start)}"):
                    parse_act()

        @pytest.mark.parametrize(
            ("pin", "message_start"),
            list(REFUSED_PINS.values()),
            ids=list(REFUSED_PINS.keys()),
        )
        class TestWhenAPinIsMalformed:
            @pytest.fixture
            def parse_act(self, pin: PinFields) -> Callable[[], tuple[SourcePin, ...]]:
                return partial(parse_source_pins, render_sources(pin))

            def test_should_refuse_it_and_name_the_field(
                self, parse_act: Callable[[], tuple[SourcePin, ...]], message_start: str
            ) -> None:
                with pytest.raises(SourcePinError, match=f"^{re.escape(message_start)}"):
                    parse_act()

        @pytest.mark.parametrize("suffix", sorted(REFUSED_SOURCE_SUFFIXES))
        class TestWhenAFileNameHasARefusedSuffix:
            @pytest.fixture
            def parse_act(self, suffix: str) -> Callable[[], tuple[SourcePin, ...]]:
                pin = change_pin(path=f"weights/model{suffix.upper()}.part")
                return partial(parse_source_pins, render_sources(pin))

            def test_should_refuse_it_whatever_its_case_and_wherever_it_stands(
                self, parse_act: Callable[[], tuple[SourcePin, ...]], suffix: str
            ) -> None:
                with pytest.raises(
                    SourcePinError,
                    match=match_exactly(
                        f"sources[0].path: 'weights/model{suffix.upper()}.part' has the refused "
                        f"suffix {suffix}, which marks code, an archive or a format that runs "
                        "code when it loads, and the pin does not say runs_code_when_loaded = true"
                    ),
                ):
                    parse_act()

        class TestWhenAPinMarksItsFileAsRunningCodeWhenItLoads:
            @pytest.fixture
            def source_pins(self) -> tuple[SourcePin, ...]:
                return parse_source_pins(
                    render_sources(change_pin(path=NEMO_ARCHIVE_PATH, runs_code_when_loaded=True))
                )

            def test_should_accept_its_refused_suffix(
                self, source_pins: tuple[SourcePin, ...]
            ) -> None:
                assert source_pins[0].path == NEMO_ARCHIVE_PATH

            def test_should_keep_the_mark_for_the_images_that_may_read_it(
                self, source_pins: tuple[SourcePin, ...]
            ) -> None:
                assert source_pins[0].runs_code_when_loaded is True

        class TestWhenAPinSaysItsFileRunsNoCode:
            @pytest.fixture
            def parse_act(self) -> Callable[[], tuple[SourcePin, ...]]:
                return partial(
                    parse_source_pins,
                    render_sources(change_pin(path=NEMO_ARCHIVE_PATH, runs_code_when_loaded=False)),
                )

            def test_should_refuse_its_refused_suffix(
                self, parse_act: Callable[[], tuple[SourcePin, ...]]
            ) -> None:
                with pytest.raises(SourcePinError, match=re.escape("has the refused suffix .nemo")):
                    parse_act()

        class TestWhenADirectoryHasARefusedSuffix:
            @pytest.fixture
            def source_pins(self) -> tuple[SourcePin, ...]:
                return parse_source_pins(render_sources(change_pin(path="model.bin/test.tsv")))

            def test_should_accept_it_since_only_the_file_name_names_its_format(
                self, source_pins: tuple[SourcePin, ...]
            ) -> None:
                assert source_pins[0].path == "model.bin/test.tsv"

        @pytest.mark.parametrize(
            ("pins", "message"),
            list(OVERLAPPING_PINS.values()),
            ids=list(OVERLAPPING_PINS.keys()),
        )
        class TestWhenTwoPinsOverlapInTheLayout:
            @pytest.fixture
            def parse_act(self, pins: tuple[PinFields, ...]) -> Callable[[], tuple[SourcePin, ...]]:
                return partial(parse_source_pins, render_sources(*pins))

            def test_should_refuse_them(
                self, parse_act: Callable[[], tuple[SourcePin, ...]], message: str
            ) -> None:
                with pytest.raises(SourcePinError, match=match_exactly(message)):
                    parse_act()

        @pytest.mark.parametrize(
            "pins", list(DISTINCT_PINS.values()), ids=list(DISTINCT_PINS.keys())
        )
        class TestWhenTwoPinsLieApart:
            @pytest.fixture
            def source_pins(self, pins: tuple[PinFields, ...]) -> tuple[SourcePin, ...]:
                return parse_source_pins(render_sources(*pins))

            def test_should_accept_both(self, source_pins: tuple[SourcePin, ...]) -> None:
                assert len(source_pins) == 2

    class TestTheRefusedSourceSuffixes:
        def test_should_be_the_refused_suffixes_without_those_of_the_benchs_tar_gz_archives(
            self,
        ) -> None:
            assert REFUSED_SUFFIXES - {".tar", ".gz"} == REFUSED_SOURCE_SUFFIXES

    class TestTheLayoutSegmentsProperty:
        def test_should_place_the_file_under_its_namespace_name_and_revision(self) -> None:
            assert GOOD_SOURCE_PIN.layout_segments == (
                "google",
                "fleurs",
                REVISION,
                "data",
                "fr_fr",
                "audio",
                "test.tar.gz",
            )

    class TestTheIsPinnedContentFunction:
        @pytest.mark.parametrize(
            ("size_bytes", "sha256", "is_pinned"),
            [
                pytest.param(SIZE_BYTES, SHA256, True, id="the pinned size and hash"),
                pytest.param(SIZE_BYTES - 1, SHA256, False, id="another size"),
                pytest.param(SIZE_BYTES, "0" * 64, False, id="another hash"),
            ],
        )
        class TestWhenGivenAFilesSizeAndHash:
            @pytest.fixture
            def is_pinned_file(self, size_bytes: int, sha256: str) -> bool:
                return is_pinned_content(GOOD_SOURCE_PIN, size_bytes, sha256)

            def test_should_accept_only_the_pinned_size_and_hash_together(
                self, is_pinned_file: bool, is_pinned: bool
            ) -> None:
                assert is_pinned_file is is_pinned
