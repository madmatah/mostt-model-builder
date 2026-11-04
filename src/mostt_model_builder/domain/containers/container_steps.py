import re
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import PurePosixPath

from mostt_model_builder.domain.bench.fleurs_check_report import (
    FLEURS_CHECK_REPORT_FILE_NAME,
    FLEURS_CHECK_REPORT_MAX_SIZE_BYTES,
)
from mostt_model_builder.domain.bench.fleurs_index import BENCH_LANGUAGES, locate_bench_files
from mostt_model_builder.domain.containers.base_images import BASE_PYTHON, ENVIRONMENT_PYTHON
from mostt_model_builder.domain.formats.file_names import is_file_name
from mostt_model_builder.domain.lock.lock_check import LOCK_FILE_NAME, LOCK_MAX_SIZE_BYTES
from mostt_model_builder.domain.onnx.conversion_report import (
    CONVERSION_REPORT_FILE_NAME,
    CONVERSION_REPORT_MAX_SIZE_BYTES,
)
from mostt_model_builder.domain.onnx.inspection_report import (
    INSPECTION_REPORT_FILE_NAME,
    INSPECTION_REPORT_MAX_SIZE_BYTES,
)

# Not 65532, the account of mostt's own runtimes, which may run on the same machine.
CONTAINER_USER_ID = 65531
CORE_OWNER = "core"
IMAGE_TAG_PREFIX = "mostt-model-builder"
CONTAINERFILE_NAME = "Containerfile"
IMAGES_DIRECTORY_NAME = "images"
RECIPES_DIRECTORY_NAME = "recipes"
STEP_NAME_GRAMMAR = re.compile(r"[a-z0-9][a-z0-9-]{0,31}")
# A row that a run derives from its step is named after the step, its run and its shard, each
# name of up to 32 characters and a shard of up to 2 digits.
ROW_NAME_GRAMMAR = re.compile(r"[a-z0-9][a-z0-9-]{0,69}")
ENTRY_MODULE_GRAMMAR = re.compile(r"[a-z_][a-z0-9_]*(\.[a-z_][a-z0-9_]*)*")
CORE_PACKAGE = "mostt_model_builder"
RECIPE_PACKAGE_PREFIX = "mostt_recipe_"
STEPS_PACKAGE_NAME = "steps"
CORE_STEPS_PACKAGE = f"{CORE_PACKAGE}.{STEPS_PACKAGE_NAME}"
IMAGE_ROOT = PurePosixPath("/opt/model-builder")
BOOTSTRAP_IN_THE_IMAGE = IMAGE_ROOT / "src" / "mostt_model_builder" / "bootstrap.py"
CONTAINER_ROOT = PurePosixPath("/")
SOURCES_IN_THE_CONTAINER = CONTAINER_ROOT / "sources"
REQUEST_IN_THE_CONTAINER = PurePosixPath("/request")
OUTPUT_IN_THE_CONTAINER = PurePosixPath("/output")
INPUTS_IN_THE_CONTAINER = PurePosixPath("/inputs")
CANDIDATE_IN_THE_CONTAINER = PurePosixPath("/candidate")
TEMPORARY_DIRECTORY_IN_THE_CONTAINER = PurePosixPath("/tmp")
SCRATCH_IN_THE_CONTAINER = PurePosixPath("/scratch")
ISOLATION_OPTIONS = ("-I", "-B")
# The Docker CLI fills each of these, in both cases, from the `proxies` of the user's
# `~/.docker/config.json` into every `docker run` and `docker build`, but only where the command
# does not set it already (`ParseProxyConfig`, in docker/cli's `cli/config/configfile/file.go`),
# so setting them empty keeps the user's proxy out of every container.
PROXY_VARIABLE_NAMES = tuple(
    name
    for upper_case_name in ("HTTP_PROXY", "HTTPS_PROXY", "FTP_PROXY", "NO_PROXY", "ALL_PROXY")
    for name in (upper_case_name, upper_case_name.lower())
)
# Every library looks for its home and caches under /tmp, the one place it may write, and the
# Hugging Face libraries never try the network, which the container does not have anyway.
CONTAINER_ENVIRONMENT = (
    ("HOME", "/tmp/home"),
    ("XDG_CACHE_HOME", "/tmp/cache"),
    ("HF_HOME", "/tmp/huggingface"),
    ("HF_HUB_OFFLINE", "1"),
    ("HF_HUB_DISABLE_TELEMETRY", "1"),
    *((name, "") for name in PROXY_VARIABLE_NAMES),
)
SCRATCH_ENVIRONMENT = (("TMPDIR", str(SCRATCH_IN_THE_CONTAINER)),)
PEAKS_FILE_NAME = "peaks.json"
PEAKS_MAX_SIZE_BYTES = 4096
FLEURS_CHECK_STEP_NAME = "fleurs-check"
CLEAN_STEP_NAME = "clean"
LOCK_STEP_NAME = "lock"
INSPECT_STEP_NAME = "inspect"
CONVERT_STEP_NAME = "convert"
# Docker reads 0 or -1 as no limit, and refuses a memory limit under 6 MiB; the ceilings only
# refuse a value that no step could mean on the machines a build runs on.
MIN_MEMORY_MIB = 6
MAX_MEMORY_MIB = 57_344
MIN_PROCESS_COUNT = 1
MAX_PROCESS_COUNT = 8192
MIN_CPU_COUNT = 1
MAX_CPU_COUNT = 64
MIN_TEMPORARY_FILESYSTEM_MIB = 1
MAX_TEMPORARY_FILESYSTEM_MIB = 16_384
MIN_OUTPUT_SIZE_BYTES = 1
MAX_OUTPUT_SIZE_BYTES = 8 * 2**30
FLEURS_REPOSITORY = "google/fleurs"


class StepTableError(ValueError):
    pass


class Device(StrEnum):
    CPU = "cpu"
    GPU = "gpu"


@dataclass(frozen=True, slots=True)
class ContainerImage:
    owner: str
    name: str
    device: Device
    python: PurePosixPath

    @property
    def tag(self) -> str:
        return f"{IMAGE_TAG_PREFIX}-{self.owner}-{self.name}"

    @property
    def containerfile(self) -> PurePosixPath:
        images_directory = PurePosixPath(IMAGES_DIRECTORY_NAME)
        if self.owner != CORE_OWNER:
            images_directory = PurePosixPath(
                RECIPES_DIRECTORY_NAME, self.owner, IMAGES_DIRECTORY_NAME
            )
        return images_directory / self.name / CONTAINERFILE_NAME


@dataclass(frozen=True, slots=True)
class StepLimits:
    memory_mib: int
    process_count: int
    cpu_count: int
    temporary_directory_mib: int
    scratch_mib: int | None = None


@dataclass(frozen=True, slots=True)
class DeclaredOutput:
    name: str
    max_size_bytes: int


@dataclass(frozen=True, slots=True)
class SourceReference:
    repository: str
    path: str


@dataclass(frozen=True, slots=True)
class EarlierOutput:
    step: str
    output: DeclaredOutput
    step_output_names: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class StepRow:
    name: str
    image: ContainerImage
    entry_module: str
    sources: tuple[SourceReference, ...]
    has_request: bool
    outputs: tuple[DeclaredOutput, ...]
    limits: StepLimits
    uses_gpu: bool = False
    has_network: bool = False
    earlier_outputs: tuple[EarlierOutput, ...] = ()
    binds_candidate: bool = False


@dataclass(frozen=True, slots=True)
class StepPlacement:
    output_directory: PurePosixPath
    request_directory: PurePosixPath | None
    log_path: PurePosixPath
    candidate_directory: PurePosixPath | None = None


@dataclass(frozen=True, slots=True)
class BindMount:
    host_path: PurePosixPath
    container_path: PurePosixPath
    is_read_only: bool
    is_file: bool
    is_written_by_a_step: bool = False


@dataclass(frozen=True, slots=True)
class TemporaryFilesystem:
    container_path: PurePosixPath
    size_mib: int


@dataclass(frozen=True, slots=True)
class ContainerRun:
    image_id: str
    python: PurePosixPath
    entry_module: str
    work_root: PurePosixPath
    mounts: tuple[BindMount, ...]
    temporary_filesystems: tuple[TemporaryFilesystem, ...]
    limits: StepLimits
    uses_gpu: bool
    has_network: bool
    user_id: int
    group_id: int
    environment: tuple[tuple[str, str], ...]

    @property
    def python_arguments(self) -> tuple[str, ...]:
        return (*ISOLATION_OPTIONS, str(BOOTSTRAP_IN_THE_IMAGE), self.entry_module)


BENCH_IMAGE = ContainerImage(CORE_OWNER, "bench", Device.CPU, BASE_PYTHON)
UV_CONTAINER_IMAGE = ContainerImage(CORE_OWNER, "uv", Device.CPU, BASE_PYTHON)
ONNX_INSPECT_IMAGE = ContainerImage(CORE_OWNER, "onnx-inspect", Device.CPU, ENVIRONMENT_PYTHON)
ONNX_CONVERT_IMAGE = ContainerImage(CORE_OWNER, "onnx-convert", Device.CPU, ENVIRONMENT_PYTHON)
HUB_IMAGE = ContainerImage(CORE_OWNER, "hub", Device.CPU, ENVIRONMENT_PYTHON)
CORE_IMAGES = (BENCH_IMAGE, UV_CONTAINER_IMAGE, ONNX_INSPECT_IMAGE, ONNX_CONVERT_IMAGE, HUB_IMAGE)
PEAKS_OUTPUT = DeclaredOutput(PEAKS_FILE_NAME, PEAKS_MAX_SIZE_BYTES)
FLEURS_CHECK_ROW = StepRow(
    name=FLEURS_CHECK_STEP_NAME,
    image=BENCH_IMAGE,
    entry_module=f"{CORE_STEPS_PACKAGE}.fleurs_check",
    sources=tuple(
        SourceReference(FLEURS_REPOSITORY, path)
        for language in BENCH_LANGUAGES
        for path in locate_bench_files(language)
    ),
    has_request=True,
    outputs=(
        DeclaredOutput(FLEURS_CHECK_REPORT_FILE_NAME, FLEURS_CHECK_REPORT_MAX_SIZE_BYTES),
        PEAKS_OUTPUT,
    ),
    limits=StepLimits(memory_mib=1024, process_count=8, cpu_count=2, temporary_directory_mib=64),
)
LOCK_ROW = StepRow(
    name=LOCK_STEP_NAME,
    image=UV_CONTAINER_IMAGE,
    entry_module=f"{CORE_STEPS_PACKAGE}.lock_resolution",
    sources=(),
    has_request=True,
    outputs=(DeclaredOutput(LOCK_FILE_NAME, LOCK_MAX_SIZE_BYTES), PEAKS_OUTPUT),
    limits=StepLimits(
        memory_mib=4096, process_count=256, cpu_count=8, temporary_directory_mib=3072
    ),
    has_network=True,
)
# The inspection reads the graphs of the candidate without their weights, so that it judges every
# onnx file, in an image that produced none of them; it peaked at 2.4 GiB and 20 processes.
INSPECT_ROW = StepRow(
    name=INSPECT_STEP_NAME,
    image=ONNX_INSPECT_IMAGE,
    entry_module=f"{CORE_STEPS_PACKAGE}.onnx_inspection",
    sources=(),
    has_request=True,
    outputs=(
        DeclaredOutput(INSPECTION_REPORT_FILE_NAME, INSPECTION_REPORT_MAX_SIZE_BYTES),
        PEAKS_OUTPUT,
    ),
    limits=StepLimits(memory_mib=3072, process_count=32, cpu_count=2, temporary_directory_mib=64),
    binds_candidate=True,
)
RUNNABLE_CORE_ROWS = (FLEURS_CHECK_ROW, INSPECT_ROW)
CLEAN_LIMITS = StepLimits(memory_mib=256, process_count=32, cpu_count=1, temporary_directory_mib=16)
# The conversions of the encoder, its 2.4 GB of weights loaded whole for each variant, peaked at
# 9.7 GiB and 41 processes.
CONVERT_LIMITS = StepLimits(
    memory_mib=13_312, process_count=64, cpu_count=4, temporary_directory_mib=256
)
CONVERSION_REPORT_OUTPUT = DeclaredOutput(
    CONVERSION_REPORT_FILE_NAME, CONVERSION_REPORT_MAX_SIZE_BYTES
)


def find_package_of_owner(owner: str) -> str:
    if owner == CORE_OWNER:
        return CORE_PACKAGE
    return RECIPE_PACKAGE_PREFIX + owner.replace("-", "_")


def locate_earlier_output(earlier_output: EarlierOutput) -> PurePosixPath:
    return INPUTS_IN_THE_CONTAINER / earlier_output.step / earlier_output.output.name


def build_clean_row(image: ContainerImage) -> StepRow:
    return StepRow(
        name=CLEAN_STEP_NAME,
        image=image,
        entry_module=f"{CORE_STEPS_PACKAGE}.clean_output",
        sources=(),
        has_request=False,
        outputs=(),
        limits=CLEAN_LIMITS,
    )


def build_convert_row(
    inputs: tuple[EarlierOutput, ...], converted_outputs: tuple[DeclaredOutput, ...]
) -> StepRow:
    return StepRow(
        name=CONVERT_STEP_NAME,
        image=ONNX_CONVERT_IMAGE,
        entry_module=f"{CORE_STEPS_PACKAGE}.onnx_conversion",
        sources=(),
        has_request=True,
        outputs=(*converted_outputs, CONVERSION_REPORT_OUTPUT, PEAKS_OUTPUT),
        limits=CONVERT_LIMITS,
        earlier_outputs=inputs,
    )


def judge_row(row: StepRow, host_cpu_count: int) -> None:
    where = f"the step {row.name!a}"
    if not ROW_NAME_GRAMMAR.fullmatch(row.name):
        raise StepTableError(f"{where}: its name does not match {ROW_NAME_GRAMMAR.pattern}")
    refuse_foreign_entry_module(row, where)
    judge_limits(row.limits, host_cpu_count, where)
    judge_outputs(row, where)
    if row.uses_gpu and row.image.device is not Device.GPU:
        raise StepTableError(
            f"{where} uses the GPU in the {row.image.device} image {row.image.tag}"
        )
    if row.has_network and not (row.image.owner == CORE_OWNER and row.name == LOCK_STEP_NAME):
        raise StepTableError(f"{where} asks for the network, which only the core's lock has")


def refuse_foreign_entry_module(row: StepRow, where: str) -> None:
    if not ENTRY_MODULE_GRAMMAR.fullmatch(row.entry_module):
        raise StepTableError(
            f"{where}: its entry module {row.entry_module!a} is not a dotted module name"
        )
    steps_package = find_steps_package(row)
    if not row.entry_module.startswith(f"{steps_package}."):
        raise StepTableError(
            f"{where}: its entry module {row.entry_module} lies outside {steps_package}, "
            f"the steps of the image {row.image.tag}"
        )


def find_steps_package(row: StepRow) -> str:
    if row.name == CLEAN_STEP_NAME:
        return CORE_STEPS_PACKAGE
    return f"{find_package_of_owner(row.image.owner)}.{STEPS_PACKAGE_NAME}"


def judge_limits(limits: StepLimits, host_cpu_count: int, where: str) -> None:
    require_within(limits.memory_mib, MIN_MEMORY_MIB, MAX_MEMORY_MIB, f"{where}: its memory in MiB")
    require_within(
        limits.process_count,
        MIN_PROCESS_COUNT,
        MAX_PROCESS_COUNT,
        f"{where}: its count of processes",
    )
    require_within(
        limits.cpu_count, MIN_CPU_COUNT, min(MAX_CPU_COUNT, host_cpu_count), f"{where}: its CPUs"
    )
    require_within(
        limits.temporary_directory_mib,
        MIN_TEMPORARY_FILESYSTEM_MIB,
        MAX_TEMPORARY_FILESYSTEM_MIB,
        f"{where}: its /tmp in MiB",
    )
    temporary_filesystems_mib = limits.temporary_directory_mib
    if limits.scratch_mib is not None:
        require_within(
            limits.scratch_mib,
            MIN_TEMPORARY_FILESYSTEM_MIB,
            MAX_TEMPORARY_FILESYSTEM_MIB,
            f"{where}: its /scratch in MiB",
        )
        temporary_filesystems_mib += limits.scratch_mib
    if limits.memory_mib <= temporary_filesystems_mib:
        raise StepTableError(
            f"{where}: its memory, {limits.memory_mib} MiB, does not exceed its temporary "
            f"filesystems, {temporary_filesystems_mib} MiB, which count against it"
        )


def require_within(count: object, minimum: int, maximum: int, what: str) -> None:
    if not isinstance(count, int) or isinstance(count, bool) or not minimum <= count <= maximum:
        raise StepTableError(f"{what}, {count!a}, lies outside {minimum} to {maximum}")


def judge_outputs(row: StepRow, where: str) -> None:
    names = [output.name for output in row.outputs]
    for output in row.outputs:
        if not is_file_name(output.name):
            raise StepTableError(f"{where}: its output {output.name!a} is not a file name")
        require_within(
            output.max_size_bytes,
            MIN_OUTPUT_SIZE_BYTES,
            MAX_OUTPUT_SIZE_BYTES,
            f"{where}: the size bound of {output.name} in bytes",
        )
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise StepTableError(f"{where}: its output {duplicates[0]} is declared twice")
    if row.name != CLEAN_STEP_NAME and PEAKS_FILE_NAME not in names:
        raise StepTableError(f"{where}: its outputs lack {PEAKS_FILE_NAME}")


def plan_container_run(
    row: StepRow,
    image_id: str,
    placement: StepPlacement,
    source_paths: Mapping[SourceReference, PurePosixPath],
    work_root: PurePosixPath,
    group_id: int,
    earlier_output_directories: Mapping[str, PurePosixPath],
) -> ContainerRun:
    mounts = [
        BindMount(
            host_path=source_paths[reference],
            container_path=CONTAINER_ROOT / source_paths[reference],
            is_read_only=True,
            is_file=True,
        )
        for reference in row.sources
    ]
    mounts.extend(
        BindMount(
            host_path=earlier_output_directories[earlier_output.step] / earlier_output.output.name,
            container_path=locate_earlier_output(earlier_output),
            is_read_only=True,
            is_file=True,
            is_written_by_a_step=True,
        )
        for earlier_output in row.earlier_outputs
    )
    if row.binds_candidate:
        if placement.candidate_directory is None:
            raise StepTableError(f"the step {row.name} reads the candidate, and was given none")
        mounts.append(
            BindMount(
                placement.candidate_directory, CANDIDATE_IN_THE_CONTAINER, True, is_file=False
            )
        )
    if row.has_request:
        if placement.request_directory is None:
            raise StepTableError(f"the step {row.name} needs a request, and was given none")
        mounts.append(
            BindMount(placement.request_directory, REQUEST_IN_THE_CONTAINER, True, is_file=False)
        )
    mounts.append(
        BindMount(placement.output_directory, OUTPUT_IN_THE_CONTAINER, False, is_file=False)
    )
    temporary_filesystems = [
        TemporaryFilesystem(
            TEMPORARY_DIRECTORY_IN_THE_CONTAINER, row.limits.temporary_directory_mib
        )
    ]
    environment = list(CONTAINER_ENVIRONMENT)
    if row.limits.scratch_mib is not None:
        temporary_filesystems.append(
            TemporaryFilesystem(SCRATCH_IN_THE_CONTAINER, row.limits.scratch_mib)
        )
        environment.extend(SCRATCH_ENVIRONMENT)
    return ContainerRun(
        image_id=image_id,
        python=row.image.python,
        entry_module=row.entry_module,
        work_root=work_root,
        mounts=tuple(mounts),
        temporary_filesystems=tuple(temporary_filesystems),
        limits=row.limits,
        uses_gpu=row.uses_gpu,
        has_network=row.has_network,
        user_id=CONTAINER_USER_ID,
        group_id=group_id,
        environment=tuple(environment),
    )
