import dataclasses
import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import PurePosixPath

from mostt_model_builder.domain.cache.hardware_facts import NO_GPU_REASON, HardwareFacts
from mostt_model_builder.domain.containers.base_images import ENVIRONMENT_PYTHON
from mostt_model_builder.domain.containers.container_steps import (
    ContainerImage,
    ContainerRun,
    StepRow,
)
from mostt_model_builder.domain.contract.recipe_contract import (
    HostContentOrigin,
    JudgedCandidateFile,
)
from mostt_model_builder.domain.lock.extra_closure import LockedEntry
from mostt_model_builder.domain.sources.source_pins import SourcePin

# Changing what a key covers bumps this number, so that every output is made again rather than
# reused under a key that meant something else.
KEY_FORMAT = 1
OUTPUT_SEPARATOR = "/"

type PlainData = bool | int | float | str | list[PlainData] | dict[str, PlainData] | None


class OutputKeyError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class RowKeyInputs:
    row: StepRow
    planned_run: ContainerRun
    request_files: Mapping[str, bytes]
    image_files: Mapping[PurePosixPath, bytes]
    image_packages: tuple[LockedEntry, ...]
    bootstrap: bytes
    code_files: Mapping[PurePosixPath, bytes]
    pins: Sequence[SourcePin]
    earlier_output_digests: Mapping[str, str]
    candidate_digests: Mapping[str, str]
    hardware: HardwareFacts | None


def to_plain_data(value: object) -> PlainData:
    if value is None or isinstance(value, bool | int | float | str):
        return value
    if isinstance(value, Enum):
        return to_plain_data(value.value)
    if isinstance(value, PurePosixPath):
        return str(value)
    if isinstance(value, bytes):
        return hashlib.sha256(value).hexdigest()
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: to_plain_data(getattr(value, field.name))
            for field in dataclasses.fields(value)
        }
    if isinstance(value, Mapping):
        return {str(key): to_plain_data(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [to_plain_data(item) for item in value]
    raise OutputKeyError(f"a key cannot hold a {type(value).__name__}")


def compute_key(document: Mapping[str, PlainData]) -> str:
    canonical = json.dumps(document, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("ascii")).hexdigest()


def digest_files(files: Mapping[PurePosixPath, bytes]) -> dict[str, PlainData]:
    return {str(path): hashlib.sha256(content).hexdigest() for path, content in files.items()}


def name_image_extra(image: ContainerImage) -> str | None:
    if image.python != ENVIRONMENT_PYTHON:
        return None
    return f"{image.owner}-{image.name}"


def select_image_files(
    image: ContainerImage, tree_files: Mapping[PurePosixPath, bytes]
) -> dict[PurePosixPath, bytes]:
    image_directory = image.containerfile.parent
    return {
        path: content
        for path, content in tree_files.items()
        if path.is_relative_to(image_directory)
    }


def name_earlier_output(step: str, output: str) -> str:
    return f"{step}{OUTPUT_SEPARATOR}{output}"


# The host's paths, the work directory, the image's ID and the user's ids stay out, so that an
# output keeps its key in a work directory copied to another machine.
def describe_planned_run(planned_run: ContainerRun) -> dict[str, PlainData]:
    return {
        "python": str(planned_run.python),
        "python_arguments": list(planned_run.python_arguments),
        "environment": [[name, setting] for name, setting in planned_run.environment],
        "temporary_filesystems": [
            [str(filesystem.container_path), filesystem.size_mib]
            for filesystem in planned_run.temporary_filesystems
        ],
        "mounts": [
            [str(mount.container_path), mount.is_read_only, mount.is_file]
            for mount in planned_run.mounts
        ],
        "has_network": planned_run.has_network,
        "uses_gpu": planned_run.uses_gpu,
        "has_standard_input": planned_run.has_standard_input,
    }


def describe_source_pins(row: StepRow, pins: Sequence[SourcePin]) -> list[PlainData]:
    pins_by_file = {(pin.repository, pin.path): pin for pin in pins}
    described: list[PlainData] = []
    for source in row.sources:
        pin = pins_by_file.get((source.repository, source.path))
        if pin is None:
            raise OutputKeyError(
                f"the row {row.name} binds {source.path} of {source.repository}, which no pin gives"
            )
        described.append([pin.repository, pin.path, pin.revision, pin.sha256])
    return described


def select_digests(
    names: Sequence[str], digests: Mapping[str, str], what: str, row: StepRow
) -> dict[str, PlainData]:
    missing = [name for name in names if name not in digests]
    if missing:
        raise OutputKeyError(
            f"the key of {row.name} needs the SHA-256 of {what} {missing[0]}, which it was not "
            "given"
        )
    return {name: digests[name] for name in names}


def describe_hardware(row: StepRow, hardware: HardwareFacts | None) -> dict[str, PlainData] | None:
    if row.uses_gpu and (hardware is None or not hardware.has_gpu):
        raise OutputKeyError(f"{row.name}: {NO_GPU_REASON}")
    if hardware is None:
        return None
    if not row.uses_gpu:
        return {"cpu_models": list(hardware.cpu_models)}
    return {
        "cpu_models": list(hardware.cpu_models),
        "gpu_models": list(hardware.gpu_models),
        "driver_version": hardware.driver_version,
    }


def document_row_key(inputs: RowKeyInputs) -> dict[str, PlainData]:
    row = inputs.row
    earlier_outputs = [
        name_earlier_output(earlier_output.step, earlier_output.output.name)
        for earlier_output in row.earlier_outputs
    ]
    candidate_files = (
        sorted(inputs.candidate_digests) if row.binds_candidate else list(row.candidate_files)
    )
    document: dict[str, PlainData] = {
        "key_format": KEY_FORMAT,
        "image": {
            "files": digest_files(inputs.image_files),
            "packages": to_plain_data(inputs.image_packages),
        },
        "code": {
            "bootstrap": hashlib.sha256(inputs.bootstrap).hexdigest(),
            "modules": digest_files(inputs.code_files),
        },
        "inputs": {
            "sources": describe_source_pins(row, inputs.pins),
            "earlier_outputs": select_digests(
                earlier_outputs, inputs.earlier_output_digests, "the earlier output", row
            ),
            "candidate_files": select_digests(
                candidate_files, inputs.candidate_digests, "the candidate file", row
            ),
        },
        "parameters": {
            "row": to_plain_data(row),
            "requests": to_plain_data(dict(inputs.request_files)),
            "run": describe_planned_run(inputs.planned_run),
        },
    }
    hardware = describe_hardware(row, inputs.hardware)
    if hardware is not None:
        document["hardware"] = hardware
    return document


def describe_candidate_origin(
    candidate_file: JudgedCandidateFile, origin_digests: Mapping[str, str]
) -> PlainData:
    origin = candidate_file.origin
    if isinstance(origin, HostContentOrigin):
        return hashlib.sha256(origin.content).hexdigest()
    name = name_earlier_output(origin.step, origin.output)
    if name not in origin_digests:
        raise OutputKeyError(
            f"the key of the candidate needs the SHA-256 of {name}, which it was not given"
        )
    return origin_digests[name]


def document_candidate_key(
    candidate: Sequence[JudgedCandidateFile], origin_digests: Mapping[str, str]
) -> dict[str, PlainData]:
    return {
        "key_format": KEY_FORMAT,
        "files": [
            {
                "name": candidate_file.name,
                "format": candidate_file.file_format.value,
                "variants": list(candidate_file.variants),
                "external_data_of": candidate_file.external_data_of,
                "sha256": describe_candidate_origin(candidate_file, origin_digests),
            }
            for candidate_file in candidate
        ],
    }
