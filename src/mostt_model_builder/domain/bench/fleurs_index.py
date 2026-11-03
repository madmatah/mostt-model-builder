import re
from dataclasses import dataclass

BENCH_LANGUAGES = ("fr_fr", "en_us")
FIELD_SEPARATOR = "\t"
LINE_SEPARATOR = "\n"
# FLEURS's index: the sentence's id, the file of its recording, the raw transcription, the
# normalized transcription, the characters, the count of samples and the speaker's gender.
FIELD_COUNT = 7
SENTENCE_ID_GRAMMAR = re.compile(r"[0-9]{1,10}")
RECORDING_NAME_GRAMMAR = re.compile(r"[0-9]{1,24}\.wav")
SAMPLE_COUNT_GRAMMAR = re.compile(r"[0-9]{1,10}")
MAX_UTTERANCE_COUNT = 10_000
ARCHIVE_MEMBER_DIRECTORY = "test"


class FleursIndexError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class FleursUtterance:
    sentence_id: str
    recording_name: str
    raw_transcription: str
    transcription: str
    sample_count: int


def locate_bench_files(language: str) -> tuple[str, str]:
    return (f"data/{language}/test.tsv", f"data/{language}/audio/test.tar.gz")


def locate_archive_member(recording_name: str) -> str:
    return f"{ARCHIVE_MEMBER_DIRECTORY}/{recording_name}"


def parse_fleurs_index(text: str) -> tuple[FleursUtterance, ...]:
    if not text.endswith(LINE_SEPARATOR):
        raise FleursIndexError("the index does not end with a line feed")
    lines = text[: -len(LINE_SEPARATOR)].split(LINE_SEPARATOR)
    if len(lines) > MAX_UTTERANCE_COUNT:
        raise FleursIndexError(
            f"the index holds {len(lines)} lines, more than {MAX_UTTERANCE_COUNT}"
        )
    utterances = tuple(
        parse_utterance(line, line_number) for line_number, line in enumerate(lines, start=1)
    )
    names = [utterance.recording_name for utterance in utterances]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise FleursIndexError(f"the recording {duplicates[0]} is indexed twice")
    return utterances


def parse_utterance(line: str, line_number: int) -> FleursUtterance:
    where = f"line {line_number} of the index"
    fields = line.split(FIELD_SEPARATOR)
    if len(fields) != FIELD_COUNT:
        raise FleursIndexError(f"{where} holds {len(fields)} fields, not {FIELD_COUNT}")
    sentence_id, recording_name, raw_transcription, transcription, _, sample_count, _ = fields
    if not SENTENCE_ID_GRAMMAR.fullmatch(sentence_id):
        raise FleursIndexError(f"{where}: the sentence id {sentence_id!a} is not a number")
    if not RECORDING_NAME_GRAMMAR.fullmatch(recording_name):
        raise FleursIndexError(
            f"{where}: the recording {recording_name!a} does not match "
            f"{RECORDING_NAME_GRAMMAR.pattern}"
        )
    if not SAMPLE_COUNT_GRAMMAR.fullmatch(sample_count):
        raise FleursIndexError(f"{where}: the count of samples {sample_count!a} is not a number")
    if not transcription:
        raise FleursIndexError(f"{where}: the transcription is empty")
    return FleursUtterance(
        sentence_id=sentence_id,
        recording_name=recording_name,
        raw_transcription=raw_transcription,
        transcription=transcription,
        sample_count=int(sample_count),
    )
