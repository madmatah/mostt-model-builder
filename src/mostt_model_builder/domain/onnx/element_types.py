from enum import IntEnum


# Every member of TensorProto.DataType in onnx/onnx.proto of onnx 1.23.2, the images' pin, under
# onnx's own names: the messages carry an element type as its plain number.
class ElementType(IntEnum):
    UNDEFINED = 0
    FLOAT = 1
    UINT8 = 2
    INT8 = 3
    UINT16 = 4
    INT16 = 5
    INT32 = 6
    INT64 = 7
    STRING = 8
    BOOL = 9
    FLOAT16 = 10
    DOUBLE = 11
    UINT32 = 12
    UINT64 = 13
    COMPLEX64 = 14
    COMPLEX128 = 15
    BFLOAT16 = 16
    FLOAT8E4M3FN = 17
    FLOAT8E4M3FNUZ = 18
    FLOAT8E5M2 = 19
    FLOAT8E5M2FNUZ = 20
    UINT4 = 21
    INT4 = 22
    FLOAT4E2M1 = 23
    FLOAT8E8M0 = 24
    UINT2 = 25
    INT2 = 26
    FLOAT6E2M3 = 27
    FLOAT6E3M2 = 28


INTEGER_ELEMENT_TYPES = frozenset(
    {
        ElementType.UINT2,
        ElementType.INT2,
        ElementType.UINT4,
        ElementType.INT4,
        ElementType.UINT8,
        ElementType.INT8,
        ElementType.UINT16,
        ElementType.INT16,
        ElementType.UINT32,
        ElementType.INT32,
        ElementType.UINT64,
        ElementType.INT64,
    }
)
