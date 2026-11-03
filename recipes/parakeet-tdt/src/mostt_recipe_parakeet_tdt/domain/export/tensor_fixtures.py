import re
from collections.abc import Iterable
from dataclasses import dataclass

# A name list holds one line per tensor, under a thousand of them.
NAME_LIST_MAX_SIZE_BYTES = 2**20
MAX_TENSOR_COUNT = 10_000
FIELD_SEPARATOR = "\t"
LINE_SEPARATOR = "\n"
DIMENSION_SEPARATOR = ","
FIELD_COUNT = 3
# Each grammar holds printable ASCII alone, so that no character a reader cannot see, such as a
# control, a format or a zero-width character, passes for part of a name.
TENSOR_NAME_GRAMMAR = re.compile(r"[A-Za-z0-9_]{1,64}(\.[A-Za-z0-9_]{1,64}){0,15}")
ELEMENT_TYPE_GRAMMAR = re.compile(r"[A-Za-z][A-Za-z0-9]{0,15}")
SHAPE_GRAMMAR = re.compile(r"\[((0|[1-9][0-9]{0,9})(,(0|[1-9][0-9]{0,9})){0,7})?\]")


class TensorFixtureError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class TensorDescription:
    name: str
    element_type: str
    shape: tuple[int, ...]


def render_name_list(tensors: Iterable[TensorDescription]) -> bytes:
    return "".join(
        FIELD_SEPARATOR.join(
            (
                tensor.name,
                tensor.element_type,
                f"[{DIMENSION_SEPARATOR.join(str(dimension) for dimension in tensor.shape)}]",
            )
        )
        + LINE_SEPARATOR
        for tensor in tensors
    ).encode("ascii")


def parse_name_list(content: bytes) -> tuple[TensorDescription, ...]:
    if len(content) > NAME_LIST_MAX_SIZE_BYTES:
        raise TensorFixtureError(
            f"the name list holds {len(content)} bytes, more than {NAME_LIST_MAX_SIZE_BYTES}"
        )
    try:
        text = content.decode("ascii")
    except UnicodeDecodeError as error:
        raise TensorFixtureError(f"the name list holds a byte beyond ASCII: {error}") from error
    if not text.endswith(LINE_SEPARATOR):
        raise TensorFixtureError("the name list does not end with a line feed")
    lines = text[: -len(LINE_SEPARATOR)].split(LINE_SEPARATOR)
    if len(lines) > MAX_TENSOR_COUNT:
        raise TensorFixtureError(
            f"the name list holds {len(lines)} lines, more than {MAX_TENSOR_COUNT}"
        )
    tensors = tuple(
        parse_tensor(line, line_number) for line_number, line in enumerate(lines, start=1)
    )
    names = [tensor.name for tensor in tensors]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise TensorFixtureError(f"the tensor {duplicates[0]} is listed twice")
    return tensors


def parse_tensor(line: str, line_number: int) -> TensorDescription:
    where = f"line {line_number} of the name list"
    fields = line.split(FIELD_SEPARATOR)
    if len(fields) != FIELD_COUNT:
        raise TensorFixtureError(f"{where} holds {len(fields)} fields, not {FIELD_COUNT}")
    name, element_type, shape = fields
    if not TENSOR_NAME_GRAMMAR.fullmatch(name):
        raise TensorFixtureError(
            f"{where}: the name {name!a} does not match {TENSOR_NAME_GRAMMAR.pattern}"
        )
    if not ELEMENT_TYPE_GRAMMAR.fullmatch(element_type):
        raise TensorFixtureError(
            f"{where}: the type {element_type!a} does not match {ELEMENT_TYPE_GRAMMAR.pattern}"
        )
    if not SHAPE_GRAMMAR.fullmatch(shape):
        raise TensorFixtureError(
            f"{where}: the shape {shape!a} does not match {SHAPE_GRAMMAR.pattern}"
        )
    dimensions = shape[1:-1]
    return TensorDescription(
        name=name,
        element_type=element_type,
        shape=tuple(int(dimension) for dimension in dimensions.split(DIMENSION_SEPARATOR))
        if dimensions
        else (),
    )
