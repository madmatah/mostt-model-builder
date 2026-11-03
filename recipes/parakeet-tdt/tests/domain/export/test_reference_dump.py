import json
import re
from collections.abc import Callable
from functools import partial

import pytest

from mostt_recipe_parakeet_tdt.domain.export.reference_dump import (
    DumpedUtterance,
    JointStep,
    ReferenceDumpError,
    ReferenceSteps,
    list_tensor_names,
    parse_reference_steps,
    render_reference_steps,
)

BLANK_ID = 8192
DURATIONS = (0, 1, 2, 3, 4)
REFERENCE_STEPS = ReferenceSteps(
    utterances=(
        DumpedUtterance(
            utterance_id="1.wav",
            feature_frames=801,
            encoded_frames=101,
            steps=(JointStep(0, ()), JointStep(2, (17,))),
        ),
        DumpedUtterance(
            utterance_id="2.wav",
            feature_frames=400,
            encoded_frames=50,
            steps=(JointStep(0, ()),),
        ),
    ),
    durations=DURATIONS,
    blank_id=BLANK_ID,
)
FIRST_UTTERANCE_DOCUMENT: dict[str, object] = {
    "utterance_id": "1.wav",
    "feature_frames": 801,
    "encoded_frames": 101,
    "steps": [{"frame": 0, "prefix": []}, {"frame": 2, "prefix": [17]}],
}
SECOND_UTTERANCE_DOCUMENT: dict[str, object] = {
    "utterance_id": "2.wav",
    "feature_frames": 400,
    "encoded_frames": 50,
    "steps": [{"frame": 0, "prefix": []}],
}
REFERENCE_DOCUMENT: dict[str, object] = {
    "utterances": [FIRST_UTTERANCE_DOCUMENT, SECOND_UTTERANCE_DOCUMENT],
    "durations": list(DURATIONS),
    "blank_id": BLANK_ID,
}
TENSOR_NAMES = (
    "features_0",
    "encoded_0",
    "logits_0_0",
    "logits_0_1",
    "features_1",
    "encoded_1",
    "logits_1_0",
)
MAX_DOCUMENT_SIZE_BYTES = 2**20


def render_document(document: object) -> bytes:
    return json.dumps(document).encode()


def replace_in_first_utterance(key: str, field: object) -> dict[str, object]:
    return REFERENCE_DOCUMENT | {
        "utterances": [FIRST_UTTERANCE_DOCUMENT | {key: field}, SECOND_UTTERANCE_DOCUMENT]
    }


REFUSED_DOCUMENTS = {
    "a document nested 3 000 deep": (
        b"[" * 3000,
        "the reference's steps are not ASCII JSON: the document nests deeper than 32 levels",
    ),
    "a document beyond its bound": (
        b" " * (MAX_DOCUMENT_SIZE_BYTES + 1),
        f"the reference's steps hold {MAX_DOCUMENT_SIZE_BYTES + 1} bytes, more than "
        f"{MAX_DOCUMENT_SIZE_BYTES}",
    ),
    "a byte beyond ASCII": (
        b'{"blank_id": "\xc3\xa9"}',
        "the reference's steps are not ASCII JSON",
    ),
    "text that is not JSON": (b"blank_id: 8192", "the reference's steps are not ASCII JSON"),
    "a key missing": (
        render_document({"utterances": [], "durations": []}),
        "the reference's steps is not an object of exactly ['blank_id', 'durations', 'utterances']",
    ),
    "an unexpected key": (
        render_document(REFERENCE_DOCUMENT | {"text": "the cat"}),
        "the reference's steps is not an object of exactly ['blank_id', 'durations', 'utterances']",
    ),
    "utterances that are not a list": (
        render_document(REFERENCE_DOCUMENT | {"utterances": {}}),
        "utterances is {}, not a list",
    ),
    "a blank that is a boolean": (
        render_document(REFERENCE_DOCUMENT | {"blank_id": True}),
        "blank_id holds True, not a whole number of at least 0",
    ),
    "a negative duration": (
        render_document(REFERENCE_DOCUMENT | {"durations": [0, -1]}),
        "durations holds -1, not a whole number of at least 0",
    ),
    "an utterance id that is not a string": (
        render_document(replace_in_first_utterance("utterance_id", 1)),
        "utterance 0: its utterance_id is 1",
    ),
    "a count of frames that is a decimal": (
        render_document(replace_in_first_utterance("encoded_frames", 1.5)),
        "utterance 0: its encoded frames holds 1.5, not a whole number of at least 0",
    ),
    "a step without its prefix": (
        render_document(replace_in_first_utterance("steps", [{"frame": 0}])),
        "utterance 0, step 0 is not an object of exactly ['frame', 'prefix']",
    ),
    "a prefix holding a string": (
        render_document(replace_in_first_utterance("steps", [{"frame": 0, "prefix": ["17"]}])),
        "utterance 0, step 0: its prefix holds '17', not a whole number of at least 0",
    ),
}


class TestTheReferenceDump:
    class TestTheRenderReferenceStepsFunction:
        class TestWhenTheStepsOfTwoUtterancesAreRendered:
            @pytest.fixture
            def rendered_document(self) -> object:
                return json.loads(render_reference_steps(REFERENCE_STEPS))

            def test_should_write_each_utterance_its_frames_and_its_steps_under_their_keys(
                self, rendered_document: object
            ) -> None:
                assert rendered_document == REFERENCE_DOCUMENT

    class TestTheParseReferenceStepsFunction:
        class TestWhenItReadsWhatTheRenderingWrote:
            @pytest.fixture
            def reference_steps(self) -> ReferenceSteps:
                return parse_reference_steps(render_reference_steps(REFERENCE_STEPS))

            def test_should_give_back_the_same_steps(self, reference_steps: ReferenceSteps) -> None:
                assert reference_steps == REFERENCE_STEPS

        @pytest.mark.parametrize(
            ("content", "reason"),
            list(REFUSED_DOCUMENTS.values()),
            ids=list(REFUSED_DOCUMENTS.keys()),
        )
        class TestWhenTheDocumentIsMalformed:
            @pytest.fixture
            def parse_act(self, content: bytes) -> Callable[[], ReferenceSteps]:
                return partial(parse_reference_steps, content)

            def test_should_refuse_it_and_say_why(
                self, parse_act: Callable[[], ReferenceSteps], reason: str
            ) -> None:
                with pytest.raises(ReferenceDumpError, match=re.escape(reason)):
                    parse_act()

    class TestTheListTensorNamesFunction:
        class TestWhenTwoUtterancesHoldTwoStepsAndOne:
            @pytest.fixture
            def tensor_names(self) -> tuple[str, ...]:
                return list_tensor_names(REFERENCE_STEPS)

            def test_should_name_the_features_the_encoders_outputs_and_each_steps_logits(
                self, tensor_names: tuple[str, ...]
            ) -> None:
                assert tensor_names == TENSOR_NAMES
