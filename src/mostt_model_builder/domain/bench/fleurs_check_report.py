import json
from collections.abc import Sequence
from dataclasses import dataclass

from mostt_model_builder.domain.bench.fleurs_index import (
    MAX_UTTERANCE_COUNT,
    RECORDING_NAME_GRAMMAR,
)
from mostt_model_builder.domain.bench.wav_format import SAMPLE_RATE_HZ
from mostt_model_builder.domain.json_documents.json_depth import MAX_JSON_DEPTH, refuse_deep_json

FLEURS_CHECK_REPORT_FILE_NAME = "fleurs-check.json"
FLEURS_CHECK_REPORT_MAX_SIZE_BYTES = 2**20
REPORT_KEYS = frozenset({"languages"})
LANGUAGE_KEYS = frozenset({"language", "utterances"})
UTTERANCE_KEYS = frozenset({"recording_name", "duration_samples"})
MIN_DURATION_SAMPLES = 1
# FLEURS's recordings last under 40 seconds; the bound only refuses a duration no recording of
# the bench could have.
MAX_DURATION_SAMPLES = 600 * SAMPLE_RATE_HZ


class FleursCheckReportError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class RecordingDuration:
    recording_name: str
    duration_samples: int


@dataclass(frozen=True, slots=True)
class LanguageCheck:
    language: str
    recordings: tuple[RecordingDuration, ...]

    @property
    def duration_s(self) -> float:
        return sum(recording.duration_samples for recording in self.recordings) / SAMPLE_RATE_HZ


@dataclass(frozen=True, slots=True)
class FleursCheckReport:
    languages: tuple[LanguageCheck, ...]


def render_fleurs_check_report(report: FleursCheckReport) -> bytes:
    document = {
        "languages": [
            {
                "language": language.language,
                "utterances": [
                    {
                        "recording_name": recording.recording_name,
                        "duration_samples": recording.duration_samples,
                    }
                    for recording in language.recordings
                ],
            }
            for language in report.languages
        ]
    }
    return json.dumps(document, indent=1).encode("ascii")


def parse_fleurs_check_report(
    content: bytes, expected_languages: Sequence[str]
) -> FleursCheckReport:
    if len(content) > FLEURS_CHECK_REPORT_MAX_SIZE_BYTES:
        raise FleursCheckReportError(
            f"the report holds {len(content)} bytes, more than {FLEURS_CHECK_REPORT_MAX_SIZE_BYTES}"
        )
    document = load_strict_json(content)
    report = require_table(document, REPORT_KEYS, "the report")
    entries = report["languages"]
    if not isinstance(entries, list):
        raise FleursCheckReportError("the report's languages are not a list")
    languages = tuple(
        parse_language(entry, f"the report's language {entry_index}")
        for entry_index, entry in enumerate(entries)
    )
    found_languages = [language.language for language in languages]
    if found_languages != list(expected_languages):
        raise FleursCheckReportError(
            f"the report checks the languages {found_languages!a}, not {list(expected_languages)}"
        )
    return FleursCheckReport(languages)


def parse_language(entry: object, where: str) -> LanguageCheck:
    table = require_table(entry, LANGUAGE_KEYS, where)
    language = table["language"]
    if not isinstance(language, str):
        raise FleursCheckReportError(f"{where}: its name is not a string")
    utterances = table["utterances"]
    if not isinstance(utterances, list) or not 0 < len(utterances) <= MAX_UTTERANCE_COUNT:
        raise FleursCheckReportError(
            f"{where}: its utterances are not a list of 1 to {MAX_UTTERANCE_COUNT} entries"
        )
    recordings = tuple(
        parse_recording(utterance, f"{where}, utterance {utterance_index}")
        for utterance_index, utterance in enumerate(utterances)
    )
    names = [recording.recording_name for recording in recordings]
    if len(set(names)) != len(names):
        raise FleursCheckReportError(f"{where}: a recording is reported twice")
    return LanguageCheck(language, recordings)


def parse_recording(entry: object, where: str) -> RecordingDuration:
    table = require_table(entry, UTTERANCE_KEYS, where)
    recording_name = table["recording_name"]
    if not isinstance(recording_name, str) or not RECORDING_NAME_GRAMMAR.fullmatch(recording_name):
        raise FleursCheckReportError(
            f"{where}: the recording {recording_name!a} does not match "
            f"{RECORDING_NAME_GRAMMAR.pattern}"
        )
    duration_samples = table["duration_samples"]
    if (
        not isinstance(duration_samples, int)
        or isinstance(duration_samples, bool)
        or not MIN_DURATION_SAMPLES <= duration_samples <= MAX_DURATION_SAMPLES
    ):
        raise FleursCheckReportError(
            f"{where}: the duration {duration_samples!a} is not a count of samples "
            f"from {MIN_DURATION_SAMPLES} to {MAX_DURATION_SAMPLES}"
        )
    return RecordingDuration(recording_name, duration_samples)


def load_strict_json(content: bytes) -> object:
    try:
        return json.loads(
            refuse_deep_json(content.decode("utf-8"), MAX_JSON_DEPTH),
            object_pairs_hook=refuse_duplicate_keys,
            parse_constant=refuse_constant,
        )
    except (UnicodeDecodeError, ValueError) as error:
        raise FleursCheckReportError(f"the report is not strict JSON: {error}") from error


def refuse_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    keys = [key for key, _ in pairs]
    if len(set(keys)) != len(keys):
        raise ValueError("a key appears twice in an object")
    return dict(pairs)


def refuse_constant(constant: str) -> object:
    raise ValueError(f"{constant} is not a JSON number")


def require_table(entry: object, keys: frozenset[str], where: str) -> dict[str, object]:
    if not isinstance(entry, dict):
        raise FleursCheckReportError(f"{where} is not an object")
    if set(entry) != keys:
        raise FleursCheckReportError(
            f"{where} holds the keys {sorted(entry)!a}, not {sorted(keys)}"
        )
    return entry
