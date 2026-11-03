import re
from collections.abc import Callable
from functools import partial
from pathlib import PurePosixPath

import pytest

from mostt_model_builder.domain.bench.fleurs_check_request import (
    BenchFiles,
    FleursCheckRequest,
    FleursCheckRequestError,
    parse_fleurs_check_request,
    render_fleurs_check_request,
)

INDEX = PurePosixPath("/sources/google/fleurs/70bb2e84/data/fr_fr/test.tsv")
ARCHIVE = PurePosixPath("/sources/google/fleurs/70bb2e84/data/fr_fr/audio/test.tar.gz")
REQUEST = FleursCheckRequest((BenchFiles("fr_fr", INDEX, ARCHIVE),))
FRENCH_ENTRY = f'{{"language": "fr_fr", "index": "{INDEX}", "archive": "{ARCHIVE}"}}'
REFUSED_REQUESTS = {
    "not json": (b"[", "the request is not JSON"),
    "a document nested 3 000 deep": (b"[" * 3000, "the document nests deeper than 32 levels"),
    "an unknown key": (b'{"languages": [], "x": 1}', "not an object of ['languages']"),
    "no language": (b'{"languages": []}', "the request names no language"),
    "a language outside the bench": (
        f'{{"languages": [{FRENCH_ENTRY.replace("fr_fr", "de_de", 1)}]}}'.encode(),
        "'de_de' is not a language of the bench",
    ),
    "a language twice": (
        f'{{"languages": [{FRENCH_ENTRY}, {FRENCH_ENTRY}]}}'.encode(),
        "names a language twice",
    ),
    "a path outside the sources": (
        f'{{"languages": [{FRENCH_ENTRY.replace(str(INDEX), "/etc/passwd")}]}}'.encode(),
        "'/etc/passwd' is not a path under /sources",
    ),
    "a path that climbs": (
        f'{{"languages": [{FRENCH_ENTRY.replace(str(INDEX), "/sources/../etc/passwd")}]}}'.encode(),
        "is not a path under /sources",
    ),
}


class TestTheFleursCheckRequest:
    class TestTheParseFleursCheckRequestFunction:
        class TestWhenTheHostWroteIt:
            @pytest.fixture
            def request_(self) -> FleursCheckRequest:
                return parse_fleurs_check_request(render_fleurs_check_request(REQUEST))

            def test_should_read_each_languages_files(self, request_: FleursCheckRequest) -> None:
                assert request_ == REQUEST

        @pytest.mark.parametrize(
            ("content", "reason"),
            list(REFUSED_REQUESTS.values()),
            ids=list(REFUSED_REQUESTS.keys()),
        )
        class TestWhenTheRequestIsMalformed:
            @pytest.fixture
            def parse_act(self, content: bytes) -> Callable[[], FleursCheckRequest]:
                return partial(parse_fleurs_check_request, content)

            def test_should_refuse_it_and_say_why(
                self, parse_act: Callable[[], FleursCheckRequest], reason: str
            ) -> None:
                with pytest.raises(FleursCheckRequestError, match=re.escape(reason)):
                    parse_act()
