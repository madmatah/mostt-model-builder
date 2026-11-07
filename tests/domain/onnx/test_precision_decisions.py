import pytest

from mostt_model_builder.domain.onnx.conversion_request import ConversionKind
from mostt_model_builder.domain.onnx.element_types import ElementType
from mostt_model_builder.domain.onnx.precision_decisions import (
    CORE_FLOAT32_OPERATORS_IN_BFLOAT16,
    MAX_FINITE_VALUE,
    MIN_POSITIVE_VALUE,
    list_kept_float32_operators,
    realign_cast_target,
    retype_for_bfloat16,
)

INT64 = ElementType.INT64
KEPT_OPERATORS = {
    "float16 keeps no operator of the core": (ConversionKind.FLOAT16, (), ()),
    "float16 keeps the operators its recipe adds": (
        ConversionKind.FLOAT16,
        ("Range",),
        ("Range",),
    ),
    "bfloat16 keeps the operators that ONNX Runtime's CUDA provider lacks in bfloat16": (
        ConversionKind.BFLOAT16,
        (),
        ("Pad", "Tile", "Floor"),
    ),
    "bfloat16 keeps the core's operators, then those its recipe adds": (
        ConversionKind.BFLOAT16,
        ("Range",),
        ("Pad", "Tile", "Floor", "Range"),
    ),
    "bfloat16 keeps an operator that the recipe adds again only once": (
        ConversionKind.BFLOAT16,
        ("Floor", "Range"),
        ("Pad", "Tile", "Floor", "Range"),
    ),
}
CAST_TARGETS = {
    "a float32 target whose output is declared float16 targets float16": (
        ElementType.FLOAT,
        ElementType.FLOAT16,
        ElementType.FLOAT16,
    ),
    "a float32 target whose output is declared float32 stays": (
        ElementType.FLOAT,
        ElementType.FLOAT,
        ElementType.FLOAT,
    ),
    "a float32 target whose output has no declared type stays": (
        ElementType.FLOAT,
        None,
        ElementType.FLOAT,
    ),
    "an integer target stays whatever its output declares": (INT64, ElementType.FLOAT16, INT64),
    "a float16 target stays": (ElementType.FLOAT16, ElementType.FLOAT16, ElementType.FLOAT16),
}
RETYPED_ELEMENTS = {
    "float16 becomes bfloat16": (ElementType.FLOAT16, ElementType.BFLOAT16),
    "float32, the type of the inputs and outputs, stays": (ElementType.FLOAT, ElementType.FLOAT),
    "an integer stays": (INT64, INT64),
    "bfloat16 stays": (ElementType.BFLOAT16, ElementType.BFLOAT16),
}


class TestThePrecisionDecisions:
    class TestTheListKeptFloat32OperatorsFunction:
        @pytest.mark.parametrize(
            ("kind", "added_operators", "expected_operators"),
            list(KEPT_OPERATORS.values()),
            ids=list(KEPT_OPERATORS),
        )
        class TestWhenAConversionKeepsOperatorsInFloat32:
            @pytest.fixture
            def kept_operators(
                self, kind: ConversionKind, added_operators: tuple[str, ...]
            ) -> tuple[str, ...]:
                return list_kept_float32_operators(kind, added_operators)

            def test_should_keep_the_cores_operators_for_its_kind_and_the_recipes(
                self, kept_operators: tuple[str, ...], expected_operators: tuple[str, ...]
            ) -> None:
                assert kept_operators == expected_operators

    class TestTheRealignCastTargetFunction:
        @pytest.mark.parametrize(
            ("cast_target", "output_element_type", "expected_target"),
            list(CAST_TARGETS.values()),
            ids=list(CAST_TARGETS),
        )
        class TestWhenACastIsRead:
            @pytest.fixture
            def realigned_target(self, cast_target: int, output_element_type: int | None) -> int:
                return realign_cast_target(cast_target, output_element_type)

            def test_should_target_the_type_its_output_declares_when_float16(
                self, realigned_target: int, expected_target: int
            ) -> None:
                assert realigned_target == expected_target

    class TestTheRetypeForBfloat16Function:
        @pytest.mark.parametrize(
            ("element_type", "expected_type"),
            list(RETYPED_ELEMENTS.values()),
            ids=list(RETYPED_ELEMENTS),
        )
        class TestWhenAnElementTypeIsRetyped:
            @pytest.fixture
            def retyped_element(self, element_type: int) -> int:
                return retype_for_bfloat16(element_type)

            def test_should_turn_float16_alone_into_bfloat16(
                self, retyped_element: int, expected_type: int
            ) -> None:
                assert retyped_element == expected_type

    class TestTheCoreFloat32OperatorsInBfloat16Constant:
        def test_should_hold_the_operators_without_a_bfloat16_kernel_on_cuda(self) -> None:
            assert CORE_FLOAT32_OPERATORS_IN_BFLOAT16 == ("Pad", "Tile", "Floor")

    class TestTheElementTypeConstant:
        def test_should_number_the_types_as_onnx_does(self) -> None:
            assert (ElementType.FLOAT, ElementType.FLOAT16, ElementType.BFLOAT16) == (1, 10, 16)

    class TestTheClampBoundConstants:
        def test_should_be_the_defaults_of_onnxconverter_common(self) -> None:
            assert (MIN_POSITIVE_VALUE, MAX_FINITE_VALUE) == (1e-7, 1e4)
