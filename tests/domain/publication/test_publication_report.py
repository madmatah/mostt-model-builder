import json
import re
from collections.abc import Callable
from dataclasses import replace
from functools import partial

import pytest

from mostt_model_builder.domain.publication.publication_report import (
    PublicationReport,
    PublicationReportError,
    ReportedFile,
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
REPORTED_FILES = (
    ReportedFile(FILES[0].name, FILES[0].size_bytes, FILES[0].sha256, is_uploaded=False),
    ReportedFile(FILES[1].name, FILES[1].size_bytes, FILES[1].sha256, is_uploaded=True),
)
KEPT_FILES = tuple(replace(reported, is_uploaded=False) for reported in REPORTED_FILES)
REPORT = PublicationReport(REPOSITORY, True, PARENT_REVISION, REVISION, REPORTED_FILES)
NOTHING_UPLOADED = PublicationReport(REPOSITORY, True, PARENT_REVISION, PARENT_REVISION, KEPT_FILES)
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
    "a file that does not say whether it was uploaded": (
        change_document(files=[{"name": "README.md", "size_bytes": 900, "sha256": "c" * 64}]),
        "a file of the report is not an object that says whether it was uploaded",
    ),
    "a document nested beyond its depth": (b"[" * 40 + b"]" * 40, "is not ASCII JSON"),
}
DISAGREEING_REPORTS = {
    "another repository": (
        replace(REPORT, repository="example-user/other"),
        "the report names the repository example-user/other, not example-user/parakeet-ultra-onnx",
    ),
    "an upload claimed at its parent's revision": (
        replace(REPORT, revision=PARENT_REVISION),
        "the report's revision is its parent, though it uploaded files: nothing was committed",
    ),
    "a new revision without an upload": (
        replace(NOTHING_UPLOADED, revision=REVISION),
        "the report gives a new revision, though it uploaded no file",
    ),
    "a missing file": (
        replace(REPORT, files=REPORTED_FILES[1:]),
        "the report lacks encoder-model.onnx",
    ),
    "an extra file": (
        replace(
            REPORT,
            files=(*REPORTED_FILES, ReportedFile("model.safetensors", 7, "d" * 64, True)),
        ),
        "the report names model.safetensors, which the request does not",
    ),
    "a file named twice": (
        replace(REPORT, files=(*REPORTED_FILES, REPORTED_FILES[0])),
        "the report names one file twice",
    ),
    "a size that differs": (
        replace(REPORT, files=(replace(REPORTED_FILES[0], size_bytes=4095), REPORTED_FILES[1])),
        "the report gives 4095 bytes of SHA-256",
    ),
    "a file left in place with another hash": (
        replace(REPORT, files=(REPORTED_FILES[0], replace(REPORTED_FILES[1], sha256="e" * 64))),
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

        class TestWhenNothingWasUploadedAndTheRevisionIsTheParent:
            @pytest.fixture
            def judge_act(self) -> Callable[[], object]:
                return partial(judge_publication_report, NOTHING_UPLOADED, REQUEST)

            def test_should_accept_it_since_the_hub_held_every_file(
                self, judge_act: Callable[[], object]
            ) -> None:
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
