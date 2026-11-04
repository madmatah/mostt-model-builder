import json
import re
from collections.abc import Callable
from dataclasses import replace
from functools import partial

import pytest

from mostt_model_builder.domain.publication.publication_report import (
    PublicationReport,
    PublicationReportError,
    judge_publication_report,
    parse_publication_report,
    render_publication_report,
)
from mostt_model_builder.domain.publication.publication_request import (
    PublicationRequest,
    PublishedFile,
)

REPOSITORY = "example-user/parakeet-ultra-onnx"
PARENT_REVISION = "0123456789abcdef0123456789abcdef01234567"
REVISION = "89abcdef0123456789abcdef0123456789abcdef"
FILES = (
    PublishedFile("encoder-model.onnx", 4096, "a" * 64),
    PublishedFile("README.md", 900, "c" * 64),
)
REQUEST = PublicationRequest(REPOSITORY, "Publish it", FILES[:1], FILES[1])
REPORT = PublicationReport(REPOSITORY, True, PARENT_REVISION, REVISION, FILES)
DOCUMENT = json.loads(render_publication_report(REPORT))


def change_document(**fields: object) -> bytes:
    return json.dumps({**DOCUMENT, **fields}).encode()


REFUSED_REPORTS = {
    "an extra key": (change_document(author="someone"), "the report is not an object of exactly"),
    "a privacy that is not a boolean": (
        change_document(is_private="yes"),
        "the report says the repository is private: 'yes'",
    ),
    "a revision that is a branch": (
        change_document(revision="main"),
        "the report's revision is 'main', not 40 hexadecimal digits",
    ),
    "a parent revision in upper case": (
        change_document(parent_revision=PARENT_REVISION.upper()),
        "the report's parent revision is",
    ),
    "files that are not a list": (change_document(files={}), "are not a list of at most 256"),
    "a document nested beyond its depth": (b"[" * 40 + b"]" * 40, "is not ASCII JSON"),
}
DISAGREEING_REPORTS = {
    "another repository": (
        replace(REPORT, repository="example-user/other"),
        "the report names the repository example-user/other, not example-user/parakeet-ultra-onnx",
    ),
    "no new revision": (
        replace(REPORT, revision=PARENT_REVISION),
        "the report's revision is its parent: nothing was committed",
    ),
    "a missing file": (replace(REPORT, files=FILES[:1]), "the report lacks README.md"),
    "an extra file": (
        replace(REPORT, files=(*FILES, PublishedFile("model.safetensors", 7, "d" * 64))),
        "the report names model.safetensors, which the request does not",
    ),
    "a file named twice": (
        replace(REPORT, files=(*FILES, FILES[0])),
        "the report names one file twice",
    ),
    "a size that differs": (
        replace(REPORT, files=(replace(FILES[0], size_bytes=4095), FILES[1])),
        "the report gives 4095 bytes of SHA-256",
    ),
    "a hash that differs": (
        replace(REPORT, files=(FILES[0], replace(FILES[1], sha256="e" * 64))),
        f"for README.md, where the verification accepted 900 bytes of SHA-256 {'c' * 64}",
    ),
}


class TestThePublicationReport:
    class TestTheParsePublicationReportFunction:
        class TestWhenGivenARenderedReport:
            @pytest.fixture
            def publication_report(self) -> PublicationReport:
                return parse_publication_report(render_publication_report(REPORT))

            def test_should_give_it_back_unchanged(
                self, publication_report: PublicationReport
            ) -> None:
                assert publication_report == REPORT

        @pytest.mark.parametrize(
            ("content", "reason"),
            list(REFUSED_REPORTS.values()),
            ids=list(REFUSED_REPORTS.keys()),
        )
        class TestWhenTheReportIsMalformed:
            @pytest.fixture
            def parse_act(self, content: bytes) -> Callable[[], PublicationReport]:
                return partial(parse_publication_report, content)

            def test_should_refuse_it_and_say_why(
                self, parse_act: Callable[[], PublicationReport], reason: str
            ) -> None:
                with pytest.raises(PublicationReportError, match=re.escape(reason)):
                    parse_act()

    class TestTheJudgePublicationReportFunction:
        class TestWhenTheReportAgreesWithTheRequest:
            @pytest.fixture
            def judge_act(self) -> Callable[[], object]:
                return partial(judge_publication_report, REPORT, REQUEST)

            def test_should_accept_it(self, judge_act: Callable[[], object]) -> None:
                assert judge_act() is None

        @pytest.mark.parametrize(
            ("publication_report", "reason"),
            list(DISAGREEING_REPORTS.values()),
            ids=list(DISAGREEING_REPORTS.keys()),
        )
        class TestWhenTheReportDisagreesWithTheRequest:
            @pytest.fixture
            def judge_act(self, publication_report: PublicationReport) -> Callable[[], None]:
                return partial(judge_publication_report, publication_report, REQUEST)

            def test_should_refuse_it_and_say_why(
                self, judge_act: Callable[[], None], reason: str
            ) -> None:
                with pytest.raises(PublicationReportError, match=re.escape(reason)):
                    judge_act()
