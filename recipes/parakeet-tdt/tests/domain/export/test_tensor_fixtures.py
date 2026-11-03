import re
from collections.abc import Callable
from functools import partial
from pathlib import Path

import pytest

from mostt_recipe_parakeet_tdt.domain.export.tensor_fixtures import (
    TensorDescription,
    TensorFixtureError,
    parse_name_list,
    render_name_list,
)

FIXTURES_DIRECTORY = Path(__file__).resolve().parents[2] / "fixtures"
FIXTURE_TENSOR_COUNTS = {"moondream-tensors": 729, "nemo-state-dict": 725}
TENSORS = (
    TensorDescription("encoder.layers.0.conv.pointwise_conv1.weight", "float32", (2048, 1024, 1)),
    TensorDescription("encoder.layers.0.conv.norm.num_batches_tracked", "I64", ()),
)
GOOD_LINE = "encoder.pre_encode.out.weight\tfloat32\t[1024,4096]\n"
REFUSED_NAME_LISTS = {
    "a zero-width space in a name": (
        "encoder.pre_encode\u200b.out.weight\tfloat32\t[1024]\n".encode(),
        "the name list holds a byte beyond ASCII",
    ),
    "an escape character in a name": (
        b"encoder.pre_encode\x1b[2J.out.weight\tfloat32\t[1024]\n",
        "line 1 of the name list: the name 'encoder.pre_encode\\x1b[2J.out.weight' does not match",
    ),
    "a carriage return before the line feed": (
        b"encoder.pre_encode.out.weight\tfloat32\t[1024]\r\n",
        "line 1 of the name list: the shape '[1024]\\r' does not match",
    ),
    "a name that climbs a level": (
        b"..\tfloat32\t[1024]\n",
        "line 1 of the name list: the name '..' does not match",
    ),
    "an empty segment in a name": (
        b"encoder..weight\tfloat32\t[1024]\n",
        "line 1 of the name list: the name 'encoder..weight' does not match",
    ),
    "a type that is no word": (
        b"encoder.weight\tfloat 32\t[1024]\n",
        "line 1 of the name list: the type 'float 32' does not match",
    ),
    "a negative dimension": (
        b"encoder.weight\tfloat32\t[-1]\n",
        "line 1 of the name list: the shape '[-1]' does not match",
    ),
    "a dimension with a leading zero": (
        b"encoder.weight\tfloat32\t[01]\n",
        "line 1 of the name list: the shape '[01]' does not match",
    ),
    "a trailing comma in a shape": (
        b"encoder.weight\tfloat32\t[1,]\n",
        "line 1 of the name list: the shape '[1,]' does not match",
    ),
    "a missing field": (b"encoder.weight\t[1024]\n", "line 1 of the name list holds 2 fields"),
    "an extra field": (
        b"encoder.weight\tfloat32\t[1024]\textra\n",
        "line 1 of the name list holds 4 fields",
    ),
    "an empty line": (
        GOOD_LINE.encode("ascii") + b"\n",
        "line 2 of the name list holds 1 fields",
    ),
    "a tensor listed twice": (
        (GOOD_LINE * 2).encode("ascii"),
        "the tensor encoder.pre_encode.out.weight is listed twice",
    ),
    "no line feed at the end": (
        GOOD_LINE.rstrip("\n").encode("ascii"),
        "the name list does not end with a line feed",
    ),
    "more than 1 MiB": (b"a" * (2**20 + 1), "holds 1048577 bytes, more than 1048576"),
}


class TestTheTensorFixtures:
    class TestTheParseNameListFunction:
        @pytest.mark.parametrize(
            ("fixture_name", "tensor_count"),
            list(FIXTURE_TENSOR_COUNTS.items()),
        )
        class TestWhenACommittedFixtureIsRead:
            @pytest.fixture
            def tensors(self, fixture_name: str) -> tuple[TensorDescription, ...]:
                return parse_name_list((FIXTURES_DIRECTORY / fixture_name).read_bytes())

            def test_should_read_every_tensor_that_describe_listed(
                self, tensors: tuple[TensorDescription, ...], tensor_count: int
            ) -> None:
                assert len(tensors) == tensor_count

        class TestWhenAListThatDescribeRenderedIsRead:
            @pytest.fixture
            def tensors(self) -> tuple[TensorDescription, ...]:
                return parse_name_list(render_name_list(TENSORS))

            def test_should_read_each_name_type_and_shape_back(
                self, tensors: tuple[TensorDescription, ...]
            ) -> None:
                assert tensors == TENSORS

        @pytest.mark.parametrize(
            ("content", "reason"),
            list(REFUSED_NAME_LISTS.values()),
            ids=list(REFUSED_NAME_LISTS.keys()),
        )
        class TestWhenTheListIsMalformed:
            @pytest.fixture
            def parse_act(self, content: bytes) -> Callable[[], tuple[TensorDescription, ...]]:
                return partial(parse_name_list, content)

            def test_should_refuse_it_and_say_why(
                self, parse_act: Callable[[], tuple[TensorDescription, ...]], reason: str
            ) -> None:
                with pytest.raises(TensorFixtureError, match=re.escape(reason)):
                    parse_act()
