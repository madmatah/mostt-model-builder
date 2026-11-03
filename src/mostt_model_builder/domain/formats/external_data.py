from collections.abc import Mapping, Sequence

from mostt_model_builder.domain.onnx.inspection_report import ExternalTensor


class ExternalDataError(ValueError):
    pass


def judge_external_data(
    name: str,
    size_bytes: int,
    external_data_of: str,
    tensors_by_onnx_file: Mapping[str, Sequence[ExternalTensor]],
) -> None:
    where = f"the external data {name}"
    foreign_files = sorted(
        onnx_file
        for onnx_file, tensors in tensors_by_onnx_file.items()
        if onnx_file != external_data_of and any(tensor.location == name for tensor in tensors)
    )
    if foreign_files:
        raise ExternalDataError(
            f"{where} is named by {foreign_files[0]}, not only by {external_data_of}, the onnx "
            "file it belongs to"
        )
    tensors = sorted(
        (
            tensor
            for tensor in tensors_by_onnx_file.get(external_data_of, ())
            if tensor.location == name
        ),
        key=lambda tensor: (tensor.offset_bytes, tensor.length_bytes),
    )
    if not tensors:
        raise ExternalDataError(f"{where} holds no tensor of {external_data_of}")
    end_bytes = 0
    for tensor in tensors:
        if tensor.offset_bytes + tensor.length_bytes > size_bytes:
            raise ExternalDataError(
                f"{where} holds {size_bytes} bytes, and a tensor reads up to byte "
                f"{tensor.offset_bytes + tensor.length_bytes}"
            )
        if tensor.offset_bytes < end_bytes:
            raise ExternalDataError(
                f"{where} has two tensors that overlap, at byte {tensor.offset_bytes}"
            )
        if tensor.offset_bytes > end_bytes:
            raise ExternalDataError(
                f"{where} has a gap of {tensor.offset_bytes - end_bytes} bytes at byte {end_bytes}"
            )
        end_bytes = tensor.offset_bytes + tensor.length_bytes
    if end_bytes != size_bytes:
        raise ExternalDataError(
            f"{where} holds {size_bytes - end_bytes} bytes after its last tensor"
        )
