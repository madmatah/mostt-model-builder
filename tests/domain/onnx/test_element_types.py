import pytest

from mostt_model_builder.domain.onnx.element_types import INTEGER_ELEMENT_TYPES, ElementType

INTEGER = "an integer"
NO_INTEGER = "no integer"
ONNX_ELEMENT_TYPE_COUNT = 29
INTEGER_KINDS = {
    ElementType.UNDEFINED: NO_INTEGER,
    ElementType.FLOAT: NO_INTEGER,
    ElementType.UINT8: INTEGER,
    ElementType.INT8: INTEGER,
    ElementType.UINT16: INTEGER,
    ElementType.INT16: INTEGER,
    ElementType.INT32: INTEGER,
    ElementType.INT64: INTEGER,
    ElementType.STRING: NO_INTEGER,
    ElementType.BOOL: NO_INTEGER,
    ElementType.FLOAT16: NO_INTEGER,
    ElementType.DOUBLE: NO_INTEGER,
    ElementType.UINT32: INTEGER,
    ElementType.UINT64: INTEGER,
    ElementType.COMPLEX64: NO_INTEGER,
    ElementType.COMPLEX128: NO_INTEGER,
    ElementType.BFLOAT16: NO_INTEGER,
    ElementType.FLOAT8E4M3FN: NO_INTEGER,
    ElementType.FLOAT8E4M3FNUZ: NO_INTEGER,
    ElementType.FLOAT8E5M2: NO_INTEGER,
    ElementType.FLOAT8E5M2FNUZ: NO_INTEGER,
    ElementType.UINT4: INTEGER,
    ElementType.INT4: INTEGER,
    ElementType.FLOAT4E2M1: NO_INTEGER,
    ElementType.FLOAT8E8M0: NO_INTEGER,
    ElementType.UINT2: INTEGER,
    ElementType.INT2: INTEGER,
    ElementType.FLOAT6E2M3: NO_INTEGER,
    ElementType.FLOAT6E3M2: NO_INTEGER,
}


def name_integer_kind(element_type: ElementType) -> str:
    return INTEGER if element_type in INTEGER_ELEMENT_TYPES else NO_INTEGER


class TestTheElementTypes:
    class TestTheElementTypeEnumeration:
        def test_should_hold_every_data_type_of_onnx_1_23_2_once(self) -> None:
            assert (len(ElementType), len(INTEGER_KINDS)) == (
                ONNX_ELEMENT_TYPE_COUNT,
                ONNX_ELEMENT_TYPE_COUNT,
            )

    class TestTheIntegerElementTypesConstant:
        class TestWhenEachMemberOfTheEnumerationIsSorted:
            @pytest.mark.parametrize(
                ("element_type", "expected_kind"),
                list(INTEGER_KINDS.items()),
                ids=[element_type.name for element_type in INTEGER_KINDS],
            )
            def test_should_hold_the_signed_and_unsigned_integers_of_every_width_alone(
                self, element_type: ElementType, expected_kind: str
            ) -> None:
                assert name_integer_kind(element_type) == expected_kind
