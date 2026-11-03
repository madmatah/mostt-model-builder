import json
import os
import re
import subprocess
from collections.abc import Sequence
from pathlib import PurePosixPath

from mostt_model_builder.application.run_step import ContainerExit, ImageFacts
from mostt_model_builder.domain.containers.container_steps import (
    PROXY_VARIABLE_NAMES,
    ContainerImage,
    ContainerRun,
)
from mostt_model_builder.domain.containers.printable_output import (
    filter_printable,
    keep_tail,
    show_printable,
    take_within_cap,
)
from mostt_model_builder.domain.containers.provenance import (
    IMAGE_ID_GRAMMAR,
    IMAGE_TREE_LABEL,
    TREE_HASH_BUILD_ARGUMENT,
)
from mostt_model_builder.domain.json_documents.json_depth import MAX_JSON_DEPTH, refuse_deep_json

DOCKER_PROGRAM = "docker"
# Docker reads a mount's options as comma-separated pairs, so a path holds none of the
# characters that its parser, or a shell, would read as syntax; and it resolves a `.` or `..`
# segment, which would bind another place than the one the host judged.
MOUNT_PATH_GRAMMAR = re.compile(r"(?!(?:.*/)?\.\.?(?:/|$))/[A-Za-z0-9._/-]*")
ENVIRONMENT_GRAMMAR = re.compile(r"[A-Za-z][A-Za-z0-9_]*=[A-Za-z0-9._/-]*")
IMAGE_REFERENCE_GRAMMAR = re.compile(r"[a-z0-9][a-z0-9._-]*|sha256:[0-9a-f]{64}")
TEMPORARY_FILESYSTEM_OPTIONS = "rw,noexec,nosuid,nodev"
GPU_DEVICE = "nvidia.com/gpu=all"
NO_SUCH_IMAGE = "No such image"
INSPECTION_FORMAT = "{{json .Id}} {{json .Config.Labels}}"
LOG_FILE_MODE = 0o600
READ_CHUNK_BYTES = 64 * 1024
INSPECTION_MAX_SIZE_BYTES = 64 * 1024
INSPECTION_TIMEOUT_S = 60


class ContainerEngineError(Exception):
    pass


def require_grammar(text: str, grammar: re.Pattern[str], what: str) -> str:
    if not grammar.fullmatch(text):
        raise ContainerEngineError(f"{what} {text!a} does not match {grammar.pattern}")
    return text


def build_run_arguments(run: ContainerRun) -> list[str]:
    limits = run.limits
    arguments = [
        DOCKER_PROGRAM,
        "run",
        "--rm",
        "--pull",
        "never",
        "--network",
        "bridge" if run.has_network else "none",
        "--read-only",
        "--cap-drop",
        "ALL",
        "--security-opt",
        "no-new-privileges",
        "--ipc",
        "none",
        "--log-driver",
        "none",
        "--ulimit",
        "core=0",
        "--user",
        f"{run.user_id}:{run.group_id}",
        "--memory",
        f"{limits.memory_mib}m",
        "--memory-swap",
        f"{limits.memory_mib}m",
        "--pids-limit",
        str(limits.process_count),
        "--cpus",
        str(limits.cpu_count),
    ]
    for temporary_filesystem in run.temporary_filesystems:
        target = require_grammar(
            str(temporary_filesystem.container_path), MOUNT_PATH_GRAMMAR, "a temporary filesystem"
        )
        arguments += [
            "--tmpfs",
            f"{target}:{TEMPORARY_FILESYSTEM_OPTIONS},size={temporary_filesystem.size_mib}m",
        ]
    for name, value in run.environment:
        arguments += [
            "--env",
            require_grammar(f"{name}={value}", ENVIRONMENT_GRAMMAR, "a variable"),
        ]
    for mount in run.mounts:
        source = require_grammar(
            str(run.work_root / mount.host_path), MOUNT_PATH_GRAMMAR, "a mount"
        )
        target = require_grammar(str(mount.container_path), MOUNT_PATH_GRAMMAR, "a mount")
        read_only = ",readonly" if mount.is_read_only else ""
        arguments += ["--mount", f"type=bind,source={source},target={target}{read_only}"]
    if run.uses_gpu:
        arguments += ["--device", GPU_DEVICE]
    image_id = require_grammar(run.image_id, IMAGE_ID_GRAMMAR, "the image id")
    return [*arguments, "--entrypoint", str(run.python), image_id, *run.python_arguments]


def build_image_arguments(
    containerfile: PurePosixPath, context: PurePosixPath, tag: str, tree_hash: str
) -> list[str]:
    return [
        DOCKER_PROGRAM,
        "build",
        "--progress",
        "plain",
        "--file",
        str(containerfile),
        "--tag",
        require_grammar(tag, IMAGE_REFERENCE_GRAMMAR, "the tag"),
        "--build-arg",
        f"{TREE_HASH_BUILD_ARGUMENT}={tree_hash}",
        *[argument for name in PROXY_VARIABLE_NAMES for argument in ("--build-arg", f"{name}=")],
        str(context),
    ]


def parse_image_inspection(output: bytes) -> ImageFacts:
    try:
        image_id_json, labels_json = output.decode("ascii").strip().split(" ", 1)
        image_id = json.loads(refuse_deep_json(image_id_json, MAX_JSON_DEPTH))
        labels = json.loads(refuse_deep_json(labels_json, MAX_JSON_DEPTH))
    except (UnicodeDecodeError, ValueError) as error:
        raise ContainerEngineError(
            f"docker described the image as {show_printable(output[:200])!a}: {error}"
        ) from error
    if not isinstance(image_id, str) or not IMAGE_ID_GRAMMAR.fullmatch(image_id):
        raise ContainerEngineError(f"docker gave the image id {image_id!a}")
    if labels is not None and not isinstance(labels, dict):
        raise ContainerEngineError("docker gave labels that are not an object")
    tree_label = None if labels is None else labels.get(IMAGE_TREE_LABEL)
    if tree_label is not None and not isinstance(tree_label, str):
        raise ContainerEngineError(f"the label {IMAGE_TREE_LABEL} is not a string")
    return ImageFacts(image_id, tree_label)


def run_logged(arguments: Sequence[str], log_path: PurePosixPath) -> ContainerExit:
    log_descriptor = os.open(
        log_path,
        os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW | os.O_CLOEXEC,
        LOG_FILE_MODE,
    )
    written_size_bytes = 0
    tail = b""
    try:
        with subprocess.Popen(
            list(arguments),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        ) as process:
            if process.stdout is None:
                raise ContainerEngineError(f"{DOCKER_PROGRAM} gave no output stream")
            while chunk := os.read(process.stdout.fileno(), READ_CHUNK_BYTES):
                printable = filter_printable(chunk)
                kept = take_within_cap(written_size_bytes, printable)
                if kept:
                    written_size_bytes += os.write(log_descriptor, kept)
                tail = keep_tail(tail, printable)
            status = process.wait()
    except FileNotFoundError as error:
        raise ContainerEngineError(f"{DOCKER_PROGRAM} is not installed") from error
    finally:
        os.close(log_descriptor)
    return ContainerExit(status, tail)


class DockerEngine:
    def __init__(self, clone_root: PurePosixPath) -> None:
        self.clone_root = clone_root

    def inspect_image(self, tag: str) -> ImageFacts | None:
        arguments = [
            DOCKER_PROGRAM,
            "image",
            "inspect",
            "--format",
            INSPECTION_FORMAT,
            require_grammar(tag, IMAGE_REFERENCE_GRAMMAR, "the tag"),
        ]
        try:
            completed = subprocess.run(
                arguments,
                capture_output=True,
                check=False,
                stdin=subprocess.DEVNULL,
                timeout=INSPECTION_TIMEOUT_S,
            )
        except FileNotFoundError as error:
            raise ContainerEngineError(f"{DOCKER_PROGRAM} is not installed") from error
        except subprocess.TimeoutExpired as error:
            raise ContainerEngineError(
                f"{DOCKER_PROGRAM} image inspect gave no answer within {INSPECTION_TIMEOUT_S} s"
            ) from error
        if completed.returncode != 0:
            if NO_SUCH_IMAGE.encode("ascii") in completed.stderr:
                return None
            raise ContainerEngineError(
                f"{DOCKER_PROGRAM} could not describe {tag}: "
                f"{show_printable(completed.stderr[-INSPECTION_MAX_SIZE_BYTES:])}"
            )
        return parse_image_inspection(completed.stdout[:INSPECTION_MAX_SIZE_BYTES])

    def run_container(self, run: ContainerRun, log_path: PurePosixPath) -> ContainerExit:
        return run_logged(build_run_arguments(run), log_path)

    def build_image(
        self, image: ContainerImage, tree_hash: str, log_path: PurePosixPath
    ) -> ContainerExit:
        arguments = build_image_arguments(
            self.clone_root / image.containerfile, self.clone_root, image.tag, tree_hash
        )
        return run_logged(arguments, log_path)
