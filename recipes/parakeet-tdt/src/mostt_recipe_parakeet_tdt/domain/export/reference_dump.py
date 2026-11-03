import json
from dataclasses import dataclass

from mostt_model_builder.domain.json_documents.json_depth import MAX_JSON_DEPTH, refuse_deep_json

UTTERANCES_KEY = "utterances"
DURATIONS_KEY = "durations"
BLANK_ID_KEY = "blank_id"
UTTERANCE_ID_KEY = "utterance_id"
FEATURE_FRAMES_KEY = "feature_frames"
ENCODED_FRAMES_KEY = "encoded_frames"
STEPS_KEY = "steps"
FRAME_KEY = "frame"
PREFIX_KEY = "prefix"
DOCUMENT_KEYS = frozenset({UTTERANCES_KEY, DURATIONS_KEY, BLANK_ID_KEY})
UTTERANCE_KEYS = frozenset({UTTERANCE_ID_KEY, FEATURE_FRAMES_KEY, ENCODED_FRAMES_KEY, STEPS_KEY})
STEP_KEYS = frozenset({FRAME_KEY, PREFIX_KEY})
MAX_DOCUMENT_SIZE_BYTES = 2**20


class ReferenceDumpError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class JointStep:
    frame: int
    prefix: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class DumpedUtterance:
    utterance_id: str
    feature_frames: int
    encoded_frames: int
    steps: tuple[JointStep, ...]


@dataclass(frozen=True, slots=True)
class ReferenceSteps:
    utterances: tuple[DumpedUtterance, ...]
    durations: tuple[int, ...]
    blank_id: int


def name_features(utterance_position: int) -> str:
    return f"features_{utterance_position}"


def name_encoder_outputs(utterance_position: int) -> str:
    return f"encoded_{utterance_position}"


def name_joint_logits(utterance_position: int, step_position: int) -> str:
    return f"logits_{utterance_position}_{step_position}"


def render_reference_steps(reference_steps: ReferenceSteps) -> bytes:
    document = {
        UTTERANCES_KEY: [
            {
                UTTERANCE_ID_KEY: utterance.utterance_id,
                FEATURE_FRAMES_KEY: utterance.feature_frames,
                ENCODED_FRAMES_KEY: utterance.encoded_frames,
                STEPS_KEY: [
                    {FRAME_KEY: step.frame, PREFIX_KEY: list(step.prefix)}
                    for step in utterance.steps
                ],
            }
            for utterance in reference_steps.utterances
        ],
        DURATIONS_KEY: list(reference_steps.durations),
        BLANK_ID_KEY: reference_steps.blank_id,
    }
    return json.dumps(document, indent=1).encode("ascii")


def parse_reference_steps(content: bytes) -> ReferenceSteps:
    if len(content) > MAX_DOCUMENT_SIZE_BYTES:
        raise ReferenceDumpError(
            f"the reference's steps hold {len(content)} bytes, more than {MAX_DOCUMENT_SIZE_BYTES}"
        )
    try:
        document = json.loads(refuse_deep_json(content.decode("ascii"), MAX_JSON_DEPTH))
    except (UnicodeDecodeError, ValueError) as error:
        raise ReferenceDumpError(f"the reference's steps are not ASCII JSON: {error}") from error
    fields = require_object(document, DOCUMENT_KEYS, "the reference's steps")
    return ReferenceSteps(
        utterances=tuple(
            parse_utterance(entry, f"utterance {position}")
            for position, entry in enumerate(require_list(fields[UTTERANCES_KEY], UTTERANCES_KEY))
        ),
        durations=require_integers(fields[DURATIONS_KEY], DURATIONS_KEY),
        blank_id=require_integer(fields[BLANK_ID_KEY], BLANK_ID_KEY),
    )


def parse_utterance(entry: object, where: str) -> DumpedUtterance:
    fields = require_object(entry, UTTERANCE_KEYS, where)
    utterance_id = fields[UTTERANCE_ID_KEY]
    if not isinstance(utterance_id, str):
        raise ReferenceDumpError(f"{where}: its {UTTERANCE_ID_KEY} is {utterance_id!a}")
    return DumpedUtterance(
        utterance_id=utterance_id,
        feature_frames=require_integer(fields[FEATURE_FRAMES_KEY], f"{where}: its feature frames"),
        encoded_frames=require_integer(fields[ENCODED_FRAMES_KEY], f"{where}: its encoded frames"),
        steps=tuple(
            parse_step(step, f"{where}, step {position}")
            for position, step in enumerate(require_list(fields[STEPS_KEY], f"{where}: its steps"))
        ),
    )


def parse_step(entry: object, where: str) -> JointStep:
    fields = require_object(entry, STEP_KEYS, where)
    return JointStep(
        frame=require_integer(fields[FRAME_KEY], f"{where}: its frame"),
        prefix=require_integers(fields[PREFIX_KEY], f"{where}: its prefix"),
    )


def require_object(entry: object, keys: frozenset[str], where: str) -> dict[str, object]:
    if not isinstance(entry, dict) or set(entry) != keys:
        raise ReferenceDumpError(f"{where} is not an object of exactly {sorted(keys)}")
    return entry


def require_list(entry: object, where: str) -> list[object]:
    if not isinstance(entry, list):
        raise ReferenceDumpError(f"{where} is {entry!a}, not a list")
    return entry


def require_integers(entry: object, where: str) -> tuple[int, ...]:
    return tuple(require_integer(value, where) for value in require_list(entry, where))


def require_integer(entry: object, where: str) -> int:
    if isinstance(entry, bool) or not isinstance(entry, int) or entry < 0:
        raise ReferenceDumpError(f"{where} holds {entry!a}, not a whole number of at least 0")
    return entry


def list_tensor_names(reference_steps: ReferenceSteps) -> tuple[str, ...]:
    return tuple(
        name
        for position, utterance in enumerate(reference_steps.utterances)
        for name in (
            name_features(position),
            name_encoder_outputs(position),
            *(name_joint_logits(position, step) for step in range(len(utterance.steps))),
        )
    )
