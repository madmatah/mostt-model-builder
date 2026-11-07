import json
from collections.abc import Callable
from dataclasses import replace
from functools import partial

import pytest

from mostt_model_builder.domain.cache.output_keys import PlainData, compute_key
from mostt_model_builder.domain.cache.output_records import (
    OutputRecord,
    OutputRecordError,
    RecordedOutput,
    judge_record_file,
    judge_record_identity,
    judge_recorded_outputs,
    parse_output_record,
    render_output_record,
    show_key,
)
from mostt_model_builder.domain.containers.container_files import FileFacts, FileKind

USER_ID = 1000
KEY_DOCUMENT: dict[str, PlainData] = {"key_format": 1, "code": {"modules": {"a.py": "0" * 64}}}
KEY = compute_key(KEY_DOCUMENT)
DEEP_DOCUMENT = json.loads("[" * 40 + "]" * 40)
ROW_NAME = "transcribe-float32-cpu-fr_fr-0"
LINES = RecordedOutput("lines.jsonl", 412873, "1" * 64)
PEAKS = RecordedOutput("peaks.json", 120, "2" * 64)
RECORD = OutputRecord(KEY, ROW_NAME, KEY_DOCUMENT, (LINES, PEAKS))
RECORD_FACTS = FileFacts(FileKind.REGULAR, USER_ID, 0o444, 2048, 1)


def render_document(document: object) -> bytes:
    return json.dumps(document).encode()


def render_record_without(field: str) -> bytes:
    document = json.loads(render_output_record(RECORD))
    del document[field]
    return render_document(document)


def render_record_with(field: str, value: object) -> bytes:
    document = json.loads(render_output_record(RECORD))
    document[field] = value
    return render_document(document)


class TestTheOutputRecords:
    class TestTheParseOutputRecordFunction:
        class TestWhenGivenARenderedRecord:
            @pytest.fixture
            def parsed_record(self) -> OutputRecord:
                return parse_output_record(render_output_record(RECORD))

            def test_should_read_back_the_record_it_was_rendered_from(
                self, parsed_record: OutputRecord
            ) -> None:
                assert parsed_record == RECORD

            def test_should_give_the_digest_of_each_output_by_its_name(
                self, parsed_record: OutputRecord
            ) -> None:
                assert parsed_record.digests == {LINES.name: LINES.sha256, PEAKS.name: PEAKS.sha256}

        @pytest.mark.parametrize(
            ("content", "reason"),
            [
                pytest.param(b"{not json", "not ASCII JSON", id="not-json"),
                pytest.param(
                    render_record_with("document", {"nested": DEEP_DOCUMENT}),
                    "nests deeper than 32 levels",
                    id="deeper-than-the-json-bound",
                ),
                pytest.param('{"key": "é"}'.encode(), "not ASCII JSON", id="not-ascii"),
                pytest.param(render_document([]), "does not hold exactly", id="not-an-object"),
                pytest.param(
                    render_record_without("document"), "does not hold exactly", id="field"
                ),
                pytest.param(render_record_with("key", "ab"), "key is not under", id="short-key"),
                pytest.param(render_record_with("row", "../row"), "row is not under", id="row"),
                pytest.param(render_record_with("document", []), "not an object", id="document"),
                pytest.param(render_record_with("outputs", {}), "not a list", id="outputs"),
                pytest.param(
                    render_record_with(
                        "outputs", [{"name": "../x", "size_bytes": 1, "sha256": "1" * 64}]
                    ),
                    "not named by a file name",
                    id="output-name",
                ),
                pytest.param(
                    render_record_with(
                        "outputs", [{"name": "x", "size_bytes": True, "sha256": "1" * 64}]
                    ),
                    "no size in bytes",
                    id="output-size",
                ),
                pytest.param(
                    render_record_with(
                        "outputs", [{"name": "x", "size_bytes": 1, "sha256": "sha"}]
                    ),
                    "no SHA-256",
                    id="output-digest",
                ),
                pytest.param(
                    render_record_with(
                        "outputs",
                        [{"name": "x", "size_bytes": 1, "sha256": "1" * 64}] * 2,
                    ),
                    "names an output twice",
                    id="output-twice",
                ),
            ],
        )
        class TestWhenGivenAMalformedRecord:
            @pytest.fixture
            def parse(self, content: bytes) -> Callable[[], OutputRecord]:
                return partial(parse_output_record, content)

            def test_should_fail_loudly_with_the_reason(
                self, parse: Callable[[], OutputRecord], reason: str
            ) -> None:
                with pytest.raises(OutputRecordError, match=reason):
                    parse()

    class TestTheJudgeRecordFileFunction:
        @pytest.mark.parametrize(
            "facts",
            [
                pytest.param(replace(RECORD_FACTS, kind=FileKind.LINK), id="link"),
                pytest.param(replace(RECORD_FACTS, owner_id=65531), id="container-user"),
                pytest.param(replace(RECORD_FACTS, permission_bits=0o644), id="writable"),
                pytest.param(replace(RECORD_FACTS, link_count=2), id="hard-link"),
                pytest.param(replace(RECORD_FACTS, size_bytes=5 * 2**20), id="oversized"),
            ],
        )
        class TestWhenTheRecordIsNotTheHostsOwn:
            @pytest.fixture
            def judge(self, facts: FileFacts) -> Callable[[], None]:
                return partial(judge_record_file, facts, USER_ID)

            def test_should_refuse_it(self, judge: Callable[[], None]) -> None:
                with pytest.raises(OutputRecordError, match="not the host's own"):
                    judge()

        class TestWhenTheRecordIsTheHostsOwn:
            @pytest.fixture
            def judge(self) -> Callable[[], None]:
                return partial(judge_record_file, RECORD_FACTS, USER_ID)

            def test_should_accept_it(self, judge: Callable[[], object]) -> None:
                assert judge() is None

    class TestTheJudgeRecordIdentityFunction:
        class TestWhenTheRecordIsTheEntrys:
            @pytest.fixture
            def judge(self) -> Callable[[], None]:
                return partial(judge_record_identity, RECORD, KEY, ROW_NAME)

            def test_should_accept_it(self, judge: Callable[[], object]) -> None:
                assert judge() is None

        @pytest.mark.parametrize(
            ("record", "key", "reason"),
            [
                pytest.param(RECORD, "f" * 64, "holds the key", id="another-key"),
                pytest.param(replace(RECORD, row="export"), KEY, "names the row", id="row"),
                pytest.param(
                    replace(RECORD, document={"key_format": 2}),
                    KEY,
                    "does not give its key",
                    id="edited-document",
                ),
            ],
        )
        class TestWhenTheRecordIsAnotherEntrys:
            @pytest.fixture
            def judge(self, record: OutputRecord, key: str) -> Callable[[], None]:
                return partial(judge_record_identity, record, key, ROW_NAME)

            def test_should_refuse_it_with_the_reason(
                self, judge: Callable[[], None], reason: str
            ) -> None:
                with pytest.raises(OutputRecordError, match=reason):
                    judge()

    class TestTheJudgeRecordedOutputsFunction:
        class TestWhenTheOutputsHashTheSameAgain:
            @pytest.fixture
            def judge(self) -> Callable[[], None]:
                return partial(judge_recorded_outputs, RECORD, (LINES, PEAKS))

            def test_should_accept_them(self, judge: Callable[[], object]) -> None:
                assert judge() is None

        @pytest.mark.parametrize(
            ("outputs", "reason"),
            [
                pytest.param(
                    (replace(LINES, sha256="3" * 64), PEAKS), "and its record says", id="changed"
                ),
                pytest.param(
                    (replace(LINES, size_bytes=1), PEAKS), "and its record says", id="resized"
                ),
                pytest.param((LINES,), "and the entry holds", id="missing"),
            ],
        )
        class TestWhenAnOutputDiffersFromItsRecord:
            @pytest.fixture
            def judge(self, outputs: tuple[RecordedOutput, ...]) -> Callable[[], None]:
                return partial(judge_recorded_outputs, RECORD, outputs)

            def test_should_refuse_it_with_the_reason(
                self, judge: Callable[[], None], reason: str
            ) -> None:
                with pytest.raises(OutputRecordError, match=reason):
                    judge()

    class TestTheShowKeyFunction:
        class TestWhenGivenAKey:
            @pytest.fixture
            def shown_key(self) -> str:
                return show_key(KEY)

            def test_should_show_the_first_twelve_digits(self, shown_key: str) -> None:
                assert shown_key == KEY[:12]
