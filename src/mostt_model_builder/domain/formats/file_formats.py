from enum import StrEnum


class FileFormat(StrEnum):
    ONNX = "onnx"
    TEXT = "text"
    ONNX_EXTERNAL_DATA = "onnx-external-data"
