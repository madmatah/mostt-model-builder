import errno
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import PurePosixPath

from mostt_model_builder.adapters.docker_engine import build_run_arguments
from mostt_model_builder.application.fetch_sources import SourceFileError
from mostt_model_builder.application.run_step import (
    ContainerExit,
    ImageFacts,
    WrittenFile,
    WrittenOutputs,
)
from mostt_model_builder.domain.containers.container_files import (
    FileFacts,
    FileKind,
    PathComponent,
)
from mostt_model_builder.domain.containers.container_steps import (
    BENCH_IMAGE,
    CONTAINER_USER_ID,
    ContainerRun,
    build_clean_row,
)

USER_ID = 1000
GROUP_ID = 1000
WRITTEN_MODE = 0o640
MISSING = FileFacts(FileKind.MISSING, -1, 0, 0, 0)
CLEAN_ENTRY_MODULE = build_clean_row(BENCH_IMAGE).entry_module


def build_directory_facts(mode: int = 0o700, owner_id: int = USER_ID) -> FileFacts:
    return FileFacts(FileKind.DIRECTORY, owner_id, mode, 4096, 2)


def build_file_facts(size_bytes: int, mode: int = 0o640, owner_id: int = USER_ID) -> FileFacts:
    return FileFacts(FileKind.REGULAR, owner_id, mode, size_bytes, 1)


def build_step_output(content: bytes) -> tuple[FileFacts, bytes]:
    return build_file_facts(len(content), WRITTEN_MODE, CONTAINER_USER_ID), content


@dataclass
class FakeStepFiles:
    entries: dict[PurePosixPath, FileFacts] = field(default_factory=dict)
    contents: dict[PurePosixPath, bytes] = field(default_factory=dict)
    file_modes: dict[PurePosixPath, int] = field(default_factory=dict)
    read_paths: list[PurePosixPath] = field(default_factory=list)

    def add_directory(self, path: PurePosixPath, facts: FileFacts | None = None) -> None:
        for parent in reversed(path.parents):
            self.entries.setdefault(parent, build_directory_facts())
        self.entries[path] = facts or build_directory_facts()

    def add_file(self, path: PurePosixPath, facts: FileFacts, content: bytes) -> None:
        self.add_directory(path.parent, self.entries.get(path.parent))
        self.entries[path] = facts
        self.contents[path] = content

    def list_children(self, directory: PurePosixPath) -> tuple[str, ...]:
        return tuple(sorted(path.name for path in self.entries if path.parent == directory))

    def make_directories(self, root: PurePosixPath, relative_path: PurePosixPath) -> None:
        directory = root
        for part in relative_path.parts:
            directory = directory / part
            facts = self.entries.setdefault(directory, build_directory_facts())
            if facts.kind is not FileKind.DIRECTORY:
                raise SourceFileError(f"{directory} is not a directory, or is a link")

    def make_step_directory(self, path: PurePosixPath, mode: int) -> bool:
        if path in self.entries:
            return False
        self.entries[path] = build_directory_facts(mode)
        return True

    def write_new_file(self, path: PurePosixPath, content: bytes, mode: int) -> None:
        if path in self.entries:
            raise FileExistsError(errno.EEXIST, "File exists", str(path))
        self.entries[path] = build_file_facts(len(content), mode)
        self.contents[path] = content
        self.file_modes[path] = mode

    def inspect_components(
        self, root: PurePosixPath, relative_path: PurePosixPath
    ) -> tuple[PathComponent, ...]:
        components = []
        path = root
        for part in relative_path.parts:
            path = path / part
            facts = self.entries.get(path, MISSING)
            components.append(PathComponent(path, facts))
            if facts.kind is not FileKind.DIRECTORY:
                break
        return tuple(components)

    def read_written_outputs(
        self,
        directory: PurePosixPath,
        max_sizes_bytes: Mapping[str, int],
        parsed_names: frozenset[str],
    ) -> WrittenOutputs:
        if self.entries.get(directory, MISSING).kind is not FileKind.DIRECTORY:
            raise NotADirectoryError(errno.ENOTDIR, "Not a directory", str(directory))
        entry_names = self.list_children(directory)
        written_files = {}
        for name, max_size_bytes in max_sizes_bytes.items():
            if name not in entry_names:
                continue
            facts = self.entries[directory / name]
            content = None
            if (
                name in parsed_names
                and facts.kind is FileKind.REGULAR
                and facts.size_bytes <= max_size_bytes
            ):
                content = self.contents.get(directory / name, b"")
                self.read_paths.append(directory / name)
            written_files[name] = WrittenFile(facts, content)
        return WrittenOutputs(entry_names, written_files)

    def list_entry_names(self, directory: PurePosixPath) -> tuple[str, ...]:
        return self.list_children(directory)

    def replace_file(self, path: PurePosixPath, content: bytes) -> None:
        self.entries[path] = build_file_facts(len(content))
        self.contents[path] = content

    def remove_empty_directory(self, path: PurePosixPath) -> None:
        if self.list_children(path):
            raise OSError(errno.ENOTEMPTY, "Directory not empty", str(path))
        del self.entries[path]

    def remove_request_directory(self, path: PurePosixPath) -> None:
        for name in self.list_children(path):
            del self.entries[path / name]
            self.contents.pop(path / name, None)
        del self.entries[path]

    def remove_file_if_present(self, path: PurePosixPath) -> bool:
        if path not in self.entries:
            return False
        del self.entries[path]
        self.contents.pop(path, None)
        return True


@dataclass
class FakeContainerEngine:
    files: FakeStepFiles
    images: dict[str, ImageFacts]
    step_outputs: dict[str, tuple[FileFacts, bytes]] = field(default_factory=dict)
    status: int = 0
    log_tail: bytes = b""
    runs: list[ContainerRun] = field(default_factory=list)
    log_paths: list[PurePosixPath] = field(default_factory=list)

    def inspect_image(self, tag: str) -> ImageFacts | None:
        return self.images.get(tag)

    def run_container(self, run: ContainerRun, log_path: PurePosixPath) -> ContainerExit:
        build_run_arguments(run)
        self.runs.append(run)
        self.log_paths.append(log_path)
        self.files.add_file(log_path, build_file_facts(len(self.log_tail), 0o600), self.log_tail)
        for mount in run.mounts:
            if not mount.is_read_only:
                self.write_into(run.work_root / mount.host_path, run.entry_module)
        return ContainerExit(self.status, self.log_tail)

    def write_into(self, output_directory: PurePosixPath, entry_module: str) -> None:
        if entry_module == CLEAN_ENTRY_MODULE:
            for name in self.files.list_children(output_directory):
                del self.files.entries[output_directory / name]
                self.files.contents.pop(output_directory / name, None)
            return
        for name, (facts, content) in self.step_outputs.items():
            self.files.add_file(output_directory / name, facts, content)
