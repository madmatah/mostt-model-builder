from enum import Enum, auto

# json's C decoder recurses once per level, and where it gives up depends on how the interpreter
# was built, so a nested document must be refused before it runs. The deepest documents the
# project writes nest 6 levels, such as the inspection's report and the reference's dump. A
# document that no parser can decode, such as one whose string never closes, must cost no more to
# scan than one that it can.
MAX_JSON_DEPTH = 32
OPENING_BRACKETS = frozenset("[{")
CLOSING_BRACKETS = frozenset("]}")
QUOTE = '"'
BACKSLASH = "\\"


class JsonScanState(Enum):
    OUTSIDE_A_STRING = auto()
    INSIDE_A_STRING = auto()
    AFTER_A_BACKSLASH = auto()


class JsonDepthError(ValueError):
    pass


def refuse_deep_json(text: str, max_depth: int) -> str:
    depth = 0
    state = JsonScanState.OUTSIDE_A_STRING
    for character in text:
        if state is JsonScanState.AFTER_A_BACKSLASH:
            state = JsonScanState.INSIDE_A_STRING
        elif state is JsonScanState.INSIDE_A_STRING:
            if character == BACKSLASH:
                state = JsonScanState.AFTER_A_BACKSLASH
            elif character == QUOTE:
                state = JsonScanState.OUTSIDE_A_STRING
        elif character in OPENING_BRACKETS:
            depth += 1
            if depth > max_depth:
                raise JsonDepthError(f"the document nests deeper than {max_depth} levels")
        elif character in CLOSING_BRACKETS:
            depth -= 1
        elif character == QUOTE:
            state = JsonScanState.INSIDE_A_STRING
    return text
