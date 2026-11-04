import hashlib
from dataclasses import dataclass, field
from pathlib import PurePosixPath

from mostt_model_builder.application.fetch_sources import FileDigest, SourceFileError


@dataclass
class FakeSourceFiles:
    work_root: PurePosixPath
    contents: dict[PurePosixPath, bytes] = field(default_factory=dict)
    directories: set[PurePosixPath] = field(default_factory=set)
    read_only_files: set[PurePosixPath] = field(default_factory=set)

    def __post_init__(self) -> None:
        self.directories.add(self.work_root)

    def make_directories(self, root: PurePosixPath, relative_path: PurePosixPath) -> None:
        if root not in self.directories:
            raise SourceFileError(f"{root} is not a directory, or is a link")
        directory = root
        for part in relative_path.parts:
            directory = directory / part
            if directory in self.contents:
                raise SourceFileError(f"{directory} is not a directory, or is a link")
            self.directories.add(directory)

    def find_file_size(self, path: PurePosixPath) -> int | None:
        if path in self.directories:
            raise SourceFileError(f"{path} is not a regular file, or is a link")
        content = self.contents.get(path)
        return None if content is None else len(content)

    def digest_file(self, path: PurePosixPath) -> FileDigest:
        content = self.read(path)
        return FileDigest(len(content), hashlib.sha256(content).hexdigest())

    def rename_file(self, source: PurePosixPath, target: PurePosixPath) -> None:
        self.contents[target] = self.read(source)
        del self.contents[source]

    def remove_file(self, path: PurePosixPath) -> None:
        self.read(path)
        del self.contents[path]

    def make_read_only(self, path: PurePosixPath) -> None:
        self.read(path)
        self.read_only_files.add(path)

    def read(self, path: PurePosixPath) -> bytes:
        if path not in self.contents:
            raise SourceFileError(f"{path} cannot be opened as a regular file")
        return self.contents[path]

    def write(self, path: PurePosixPath, content: bytes) -> None:
        if path.parent not in self.directories:
            raise SourceFileError(f"{path.parent} does not exist")
        self.contents[path] = content
