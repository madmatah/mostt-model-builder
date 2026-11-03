import hashlib
import os
import stat
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from pathlib import Path, PurePosixPath

import pytest

from mostt_model_builder.adapters.work_files import COPY_CHUNK_BYTES, WorkFiles
from mostt_model_builder.application.fetch_sources import FileDigest, SourceFileError
from mostt_model_builder.application.run_step import WrittenFile, WrittenOutputs
from mostt_model_builder.domain.containers.container_files import (
    FileFacts,
    FileKind,
    PathComponent,
)
from mostt_model_builder.domain.containers.work_directory import DirectoryFacts, EntryKind

FILE_CONTENT = b"id\tfile\ttranscription\n"
MAX_SIZE_BYTES = 4096
# A sparse file as large as the encoder's external data, which no parser of the host reads.
UNPARSED_SIZE_BYTES = 2_476_376_064

# A file of a full chunk and a part of one, so that the copy reads it in two chunks.
COPIED_CONTENT = bytes(range(256)) * (COPY_CHUNK_BYTES // 256) + FILE_CONTENT


@dataclass(frozen=True)
class FileCopy:
    facts: FileFacts
    observed_chunks: list[bytes]
    copied_content: bytes
    copied_mode: int


@dataclass(frozen=True)
class SkippedCopy:
    facts: FileFacts
    observed_chunks: list[bytes]
    entry_names: list[str]


def build_work_root(tmp_path: Path) -> Path:
    work_root = tmp_path / "work"
    work_root.mkdir(mode=0o700)
    work_root.chmod(0o700)
    return work_root


def build_output_directory(tmp_path: Path) -> Path:
    output_directory = tmp_path / "output"
    output_directory.mkdir()
    (tmp_path / "candidate").mkdir()
    return output_directory


def copy_into_the_candidate(tmp_path: Path, name: str, max_size_bytes: int) -> SkippedCopy:
    observed_chunks: list[bytes] = []
    facts = WorkFiles().copy_written_file(
        PurePosixPath(tmp_path / "output"),
        name,
        max_size_bytes,
        PurePosixPath(tmp_path / "candidate" / name),
        observed_chunks.append,
    )
    return SkippedCopy(facts, observed_chunks, os.listdir(tmp_path / "candidate"))


class TestTheWorkFiles:
    class TestTheInspectDirectoryMethod:
        class TestWhenThePathIsADirectory:
            @pytest.fixture
            def facts(self, tmp_path: Path) -> DirectoryFacts:
                return WorkFiles().inspect_directory(str(build_work_root(tmp_path)))

            def test_should_read_its_kind_owner_and_mode(
                self, facts: DirectoryFacts, tmp_path: Path
            ) -> None:
                assert facts == DirectoryFacts(
                    EntryKind.DIRECTORY, str((tmp_path / "work").resolve()), os.getuid(), 0o700
                )

        class TestWhenThePathIsALink:
            @pytest.fixture
            def facts(self, tmp_path: Path) -> DirectoryFacts:
                work_root = build_work_root(tmp_path)
                (tmp_path / "linked-work").symlink_to(work_root)
                return WorkFiles().inspect_directory(str(tmp_path / "linked-work"))

            def test_should_see_the_link_itself(self, facts: DirectoryFacts) -> None:
                assert facts.kind is EntryKind.LINK

            def test_should_give_where_it_resolves(
                self, facts: DirectoryFacts, tmp_path: Path
            ) -> None:
                assert facts.canonical_path == str((tmp_path / "work").resolve())

        class TestWhenThePathIsMissing:
            @pytest.fixture
            def facts(self, tmp_path: Path) -> DirectoryFacts:
                return WorkFiles().inspect_directory(str(tmp_path / "missing"))

            def test_should_say_so(self, facts: DirectoryFacts) -> None:
                assert facts.kind is EntryKind.MISSING

        class TestWhenThePathIsAFile:
            @pytest.fixture
            def facts(self, tmp_path: Path) -> DirectoryFacts:
                (tmp_path / "file").write_bytes(FILE_CONTENT)
                return WorkFiles().inspect_directory(str(tmp_path / "file"))

            def test_should_say_it_is_another_kind(self, facts: DirectoryFacts) -> None:
                assert facts.kind is EntryKind.OTHER

    class TestTheMakeDirectoriesMethod:
        class TestWhenTheDirectoriesAreMissing:
            @pytest.fixture
            def created_mode(self, tmp_path: Path) -> int:
                work_root = build_work_root(tmp_path)
                WorkFiles().make_directories(
                    PurePosixPath(work_root), PurePosixPath("sources/google/fleurs")
                )
                return stat.S_IMODE((work_root / "sources/google/fleurs").stat().st_mode)

            def test_should_create_each_one_for_the_user_alone(self, created_mode: int) -> None:
                assert created_mode & 0o077 == 0

        class TestWhenAnotherShardMakesTheDirectoryFirst:
            @pytest.fixture
            def is_directory(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> bool:
                work_root = build_work_root(tmp_path)
                make_directory = os.mkdir

                def make_directory_after_another_shard(path: PurePosixPath, mode: int) -> None:
                    make_directory(path, mode)
                    raise FileExistsError(f"another shard made {path} first")

                monkeypatch.setattr(os, "mkdir", make_directory_after_another_shard)
                WorkFiles().make_directories(
                    PurePosixPath(work_root), PurePosixPath("sources/google")
                )
                return (work_root / "sources/google").is_dir()

            def test_should_accept_it(self, is_directory: bool) -> None:
                assert is_directory

        class TestWhenALinkStandsOnThePath:
            @pytest.fixture
            def make_act(self, tmp_path: Path) -> Callable[[], None]:
                work_root = build_work_root(tmp_path)
                elsewhere = tmp_path / "elsewhere"
                elsewhere.mkdir()
                (work_root / "sources").symlink_to(elsewhere)
                return partial(
                    WorkFiles().make_directories,
                    PurePosixPath(work_root),
                    PurePosixPath("sources/google"),
                )

            def test_should_refuse_to_follow_it(self, make_act: Callable[[], None]) -> None:
                with pytest.raises(
                    SourceFileError, match="sources is not a directory, or is a link"
                ):
                    make_act()

    class TestTheFindFileSizeMethod:
        class TestWhenTheFileIsMissing:
            @pytest.fixture
            def size_bytes(self, tmp_path: Path) -> int | None:
                return WorkFiles().find_file_size(PurePosixPath(tmp_path / "missing"))

            def test_should_find_none(self, size_bytes: int | None) -> None:
                assert size_bytes is None

        class TestWhenTheFileIsALink:
            @pytest.fixture
            def find_act(self, tmp_path: Path) -> Callable[[], int | None]:
                (tmp_path / "file").write_bytes(FILE_CONTENT)
                (tmp_path / "linked-file").symlink_to(tmp_path / "file")
                return partial(WorkFiles().find_file_size, PurePosixPath(tmp_path / "linked-file"))

            def test_should_refuse_it(self, find_act: Callable[[], int | None]) -> None:
                with pytest.raises(SourceFileError, match="is not a regular file, or is a link"):
                    find_act()

    class TestTheDigestFileMethod:
        class TestWhenTheFileIsRegular:
            @pytest.fixture
            def digest(self, tmp_path: Path) -> FileDigest:
                (tmp_path / "file").write_bytes(FILE_CONTENT)
                return WorkFiles().digest_file(PurePosixPath(tmp_path / "file"))

            def test_should_give_its_size_and_sha256(self, digest: FileDigest) -> None:
                assert digest == FileDigest(
                    len(FILE_CONTENT), hashlib.sha256(FILE_CONTENT).hexdigest()
                )

        class TestWhenTheFileIsALink:
            @pytest.fixture
            def digest_act(self, tmp_path: Path) -> Callable[[], FileDigest]:
                (tmp_path / "file").write_bytes(FILE_CONTENT)
                (tmp_path / "linked-file").symlink_to(tmp_path / "file")
                return partial(WorkFiles().digest_file, PurePosixPath(tmp_path / "linked-file"))

            def test_should_refuse_to_follow_it(self, digest_act: Callable[[], FileDigest]) -> None:
                with pytest.raises(SourceFileError, match="cannot be opened as a regular file"):
                    digest_act()

    class TestTheMakeReadOnlyMethod:
        class TestWhenTheFileIsRegular:
            @pytest.fixture
            def file_mode(self, tmp_path: Path) -> int:
                (tmp_path / "file").write_bytes(FILE_CONTENT)
                WorkFiles().make_read_only(PurePosixPath(tmp_path / "file"))
                return stat.S_IMODE((tmp_path / "file").stat().st_mode)

            def test_should_leave_it_readable_by_everyone_and_writable_by_no_one(
                self, file_mode: int
            ) -> None:
                assert file_mode == 0o444

    class TestTheMakeStepDirectoryMethod:
        class TestWhenTheDirectoryIsNew:
            @pytest.fixture
            def made_mode(self, tmp_path: Path) -> tuple[bool, int]:
                is_made = WorkFiles().make_step_directory(PurePosixPath(tmp_path / "step"), 0o2770)
                return is_made, stat.S_IMODE((tmp_path / "step").stat().st_mode)

            def test_should_make_it_in_the_mode_asked_whatever_the_umask(
                self, made_mode: tuple[bool, int]
            ) -> None:
                assert made_mode == (True, 0o2770)

        class TestWhenTheDirectoryExists:
            @pytest.fixture
            def is_made(self, tmp_path: Path) -> bool:
                (tmp_path / "step").mkdir()
                return WorkFiles().make_step_directory(PurePosixPath(tmp_path / "step"), 0o2770)

            def test_should_say_it_made_none(self, is_made: bool) -> None:
                assert is_made is False

    class TestTheInspectComponentsMethod:
        class TestWhenALinkStandsOnThePath:
            @pytest.fixture
            def components(self, tmp_path: Path) -> tuple[PathComponent, ...]:
                (tmp_path / "elsewhere").mkdir()
                (tmp_path / "linked").symlink_to(tmp_path / "elsewhere")
                return WorkFiles().inspect_components(
                    PurePosixPath(tmp_path), PurePosixPath("linked/file")
                )

            def test_should_stop_at_the_link_and_say_what_it_is(
                self, components: tuple[PathComponent, ...]
            ) -> None:
                assert [component.facts.kind for component in components] == [FileKind.LINK]

        class TestWhenTheLeafIsMissing:
            @pytest.fixture
            def components(self, tmp_path: Path) -> tuple[PathComponent, ...]:
                (tmp_path / "runs").mkdir(mode=0o700)
                return WorkFiles().inspect_components(
                    PurePosixPath(tmp_path), PurePosixPath("runs/step")
                )

            def test_should_see_the_directory_then_the_missing_leaf(
                self, components: tuple[PathComponent, ...]
            ) -> None:
                assert [component.facts.kind for component in components] == [
                    FileKind.DIRECTORY,
                    FileKind.MISSING,
                ]

    class TestTheReadWrittenOutputsMethod:
        class TestWhenTheOutputsHoldAFileALinkAndAFifo:
            @pytest.fixture
            def written_outputs(self, tmp_path: Path) -> WrittenOutputs:
                (tmp_path / "peaks.json").write_bytes(FILE_CONTENT)
                (tmp_path / "peaks.json").chmod(0o640)
                (tmp_path / "report.json").symlink_to("/etc/passwd")
                os.mkfifo(tmp_path / "pipe")
                return WorkFiles().read_written_outputs(
                    PurePosixPath(tmp_path),
                    {"peaks.json": MAX_SIZE_BYTES, "report.json": MAX_SIZE_BYTES, "pipe": 1},
                    frozenset({"peaks.json", "report.json", "pipe"}),
                )

            def test_should_list_every_entry(self, written_outputs: WrittenOutputs) -> None:
                assert written_outputs.entry_names == ("peaks.json", "pipe", "report.json")

            def test_should_read_the_regular_file_with_its_facts(
                self, written_outputs: WrittenOutputs
            ) -> None:
                assert (
                    written_outputs.files["peaks.json"].facts,
                    written_outputs.files["peaks.json"].content,
                ) == (
                    FileFacts(FileKind.REGULAR, os.getuid(), 0o640, len(FILE_CONTENT), 1),
                    FILE_CONTENT,
                )

            def test_should_follow_neither_the_link_nor_the_fifo(
                self, written_outputs: WrittenOutputs
            ) -> None:
                assert [
                    (written_outputs.files[name].facts.kind, written_outputs.files[name].content)
                    for name in ("report.json", "pipe")
                ] == [(FileKind.LINK, None), (FileKind.FIFO, None)]

        class TestWhenAFileExceedsItsBound:
            @pytest.fixture
            def written_outputs(self, tmp_path: Path) -> WrittenOutputs:
                (tmp_path / "peaks.json").write_bytes(FILE_CONTENT)
                return WorkFiles().read_written_outputs(
                    PurePosixPath(tmp_path), {"peaks.json": 4}, frozenset({"peaks.json"})
                )

            def test_should_read_none_of_it(self, written_outputs: WrittenOutputs) -> None:
                assert written_outputs.files["peaks.json"].content is None

        class TestWhenAFileGrowsBeyondItsBoundAfterItsFactsAreRead:
            @pytest.fixture
            def written_file(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> WrittenFile:
                bound_bytes = len(FILE_CONTENT) - 1
                (tmp_path / "peaks.json").write_bytes(FILE_CONTENT)
                read_status = os.fstat

                def read_status_before_the_file_grew(descriptor: int) -> os.stat_result:
                    status_fields = list(read_status(descriptor))
                    status_fields[stat.ST_SIZE] = bound_bytes
                    return os.stat_result(status_fields)

                monkeypatch.setattr(os, "fstat", read_status_before_the_file_grew)
                written_outputs = WorkFiles().read_written_outputs(
                    PurePosixPath(tmp_path), {"peaks.json": bound_bytes}, frozenset({"peaks.json"})
                )
                return written_outputs.files["peaks.json"]

            def test_should_read_none_of_it_and_give_the_size_it_read_so_it_is_refused(
                self, written_file: WrittenFile
            ) -> None:
                assert (written_file.facts.size_bytes, written_file.content) == (
                    len(FILE_CONTENT),
                    None,
                )

        class TestWhenAnOutputIsNotOneTheHostParses:
            @pytest.fixture
            def written_outputs(self, tmp_path: Path) -> WrittenOutputs:
                with (tmp_path / "encoder-model.onnx.data").open("wb") as stream:
                    stream.truncate(UNPARSED_SIZE_BYTES)
                (tmp_path / "encoder-model.onnx.data").chmod(0o640)
                return WorkFiles().read_written_outputs(
                    PurePosixPath(tmp_path),
                    {"encoder-model.onnx.data": UNPARSED_SIZE_BYTES},
                    frozenset({"peaks.json"}),
                )

            def test_should_judge_it_by_its_facts_alone(
                self, written_outputs: WrittenOutputs
            ) -> None:
                assert written_outputs.files["encoder-model.onnx.data"].facts == FileFacts(
                    FileKind.REGULAR, os.getuid(), 0o640, UNPARSED_SIZE_BYTES, 1
                )

            def test_should_read_none_of_it_since_no_parser_of_the_host_reads_it(
                self, written_outputs: WrittenOutputs
            ) -> None:
                assert written_outputs.files["encoder-model.onnx.data"].content is None

        class TestWhenTheDirectoryIsALink:
            @pytest.fixture
            def read_act(self, tmp_path: Path) -> Callable[[], WrittenOutputs]:
                (tmp_path / "outputs").mkdir()
                (tmp_path / "linked").symlink_to(tmp_path / "outputs")
                return partial(
                    WorkFiles().read_written_outputs,
                    PurePosixPath(tmp_path / "linked"),
                    {},
                    frozenset(),
                )

            def test_should_refuse_to_open_it(self, read_act: Callable[[], WrittenOutputs]) -> None:
                with pytest.raises(NotADirectoryError):
                    read_act()

    class TestTheReplaceFileMethod:
        class TestWhenTheFileExists:
            @pytest.fixture
            def replaced(self, tmp_path: Path) -> tuple[bytes, list[str]]:
                (tmp_path / "uv.lock").write_bytes(b"old")
                WorkFiles().replace_file(PurePosixPath(tmp_path / "uv.lock"), FILE_CONTENT)
                return (tmp_path / "uv.lock").read_bytes(), sorted(os.listdir(tmp_path))

            def test_should_replace_its_content_in_one_step_and_leave_nothing_beside_it(
                self, replaced: tuple[bytes, list[str]]
            ) -> None:
                assert replaced == (FILE_CONTENT, ["uv.lock"])

    class TestTheRemoveRequestDirectoryMethod:
        class TestWhenTheRequestHoldsFiles:
            @pytest.fixture
            def entries_left(self, tmp_path: Path) -> list[str]:
                (tmp_path / "request").mkdir()
                (tmp_path / "request" / "request.json").write_bytes(FILE_CONTENT)
                WorkFiles().remove_request_directory(PurePosixPath(tmp_path / "request"))
                return os.listdir(tmp_path)

            def test_should_remove_its_files_and_itself(self, entries_left: list[str]) -> None:
                assert entries_left == []

    class TestTheCopyWrittenFileMethod:
        class TestWhenTheFileIsRegularAndWithinItsBound:
            @pytest.fixture
            def file_copy(self, tmp_path: Path) -> FileCopy:
                output_directory = build_output_directory(tmp_path)
                (output_directory / "encoder-model.onnx").write_bytes(COPIED_CONTENT)
                (output_directory / "encoder-model.onnx").chmod(0o640)
                target = tmp_path / "candidate" / "encoder-model.onnx"
                observed_chunks: list[bytes] = []
                facts = WorkFiles().copy_written_file(
                    PurePosixPath(output_directory),
                    "encoder-model.onnx",
                    len(COPIED_CONTENT),
                    PurePosixPath(target),
                    observed_chunks.append,
                )
                return FileCopy(
                    facts, observed_chunks, target.read_bytes(), stat.S_IMODE(target.stat().st_mode)
                )

            def test_should_give_back_the_facts_of_the_source(self, file_copy: FileCopy) -> None:
                assert file_copy.facts == FileFacts(
                    FileKind.REGULAR, os.getuid(), 0o640, len(COPIED_CONTENT), 1
                )

            def test_should_copy_it_whole(self, file_copy: FileCopy) -> None:
                assert file_copy.copied_content == COPIED_CONTENT

            def test_should_leave_the_copy_readable_and_writable_by_no_one(
                self, file_copy: FileCopy
            ) -> None:
                assert file_copy.copied_mode == 0o444

            def test_should_read_it_chunk_by_chunk(self, file_copy: FileCopy) -> None:
                assert [len(chunk) for chunk in file_copy.observed_chunks] == [
                    COPY_CHUNK_BYTES,
                    len(COPIED_CONTENT) - COPY_CHUNK_BYTES,
                ]

            def test_should_show_the_observer_the_bytes_it_copies(
                self, file_copy: FileCopy
            ) -> None:
                assert b"".join(file_copy.observed_chunks) == COPIED_CONTENT

        class TestWhenTheFileIsALink:
            @pytest.fixture
            def skipped_copy(self, tmp_path: Path) -> SkippedCopy:
                output_directory = build_output_directory(tmp_path)
                (output_directory / "vocab.txt").symlink_to("/etc/passwd")
                return copy_into_the_candidate(tmp_path, "vocab.txt", MAX_SIZE_BYTES)

            def test_should_give_back_the_facts_of_the_link(
                self, skipped_copy: SkippedCopy
            ) -> None:
                assert skipped_copy.facts.kind is FileKind.LINK

            def test_should_copy_nothing_since_it_follows_no_link(
                self, skipped_copy: SkippedCopy
            ) -> None:
                assert (skipped_copy.observed_chunks, skipped_copy.entry_names) == ([], [])

        class TestWhenTheFileIsAFifo:
            @pytest.fixture
            def skipped_copy(self, tmp_path: Path) -> SkippedCopy:
                output_directory = build_output_directory(tmp_path)
                os.mkfifo(output_directory / "vocab.txt")
                return copy_into_the_candidate(tmp_path, "vocab.txt", MAX_SIZE_BYTES)

            def test_should_give_back_the_facts_of_the_fifo(
                self, skipped_copy: SkippedCopy
            ) -> None:
                assert skipped_copy.facts.kind is FileKind.FIFO

            def test_should_copy_nothing_since_reading_it_could_block(
                self, skipped_copy: SkippedCopy
            ) -> None:
                assert (skipped_copy.observed_chunks, skipped_copy.entry_names) == ([], [])

        class TestWhenTheFileExceedsItsBound:
            @pytest.fixture
            def skipped_copy(self, tmp_path: Path) -> SkippedCopy:
                output_directory = build_output_directory(tmp_path)
                (output_directory / "vocab.txt").write_bytes(FILE_CONTENT)
                return copy_into_the_candidate(tmp_path, "vocab.txt", len(FILE_CONTENT) - 1)

            def test_should_give_back_its_size(self, skipped_copy: SkippedCopy) -> None:
                assert skipped_copy.facts.size_bytes == len(FILE_CONTENT)

            def test_should_copy_none_of_it(self, skipped_copy: SkippedCopy) -> None:
                assert (skipped_copy.observed_chunks, skipped_copy.entry_names) == ([], [])

        class TestWhenTheTargetExists:
            @pytest.fixture
            def copy_act(self, tmp_path: Path) -> Callable[[], SkippedCopy]:
                output_directory = build_output_directory(tmp_path)
                (output_directory / "vocab.txt").write_bytes(FILE_CONTENT)
                (tmp_path / "candidate" / "vocab.txt").write_bytes(b"earlier copy")
                return partial(copy_into_the_candidate, tmp_path, "vocab.txt", MAX_SIZE_BYTES)

            def test_should_refuse_to_overwrite_it(
                self, copy_act: Callable[[], SkippedCopy]
            ) -> None:
                with pytest.raises(FileExistsError):
                    copy_act()

        class TestWhenTheDirectoryIsALink:
            @pytest.fixture
            def copy_act(self, tmp_path: Path) -> Callable[[], SkippedCopy]:
                elsewhere = tmp_path / "elsewhere"
                elsewhere.mkdir()
                (elsewhere / "vocab.txt").write_bytes(FILE_CONTENT)
                (tmp_path / "output").symlink_to(elsewhere)
                (tmp_path / "candidate").mkdir()
                return partial(copy_into_the_candidate, tmp_path, "vocab.txt", MAX_SIZE_BYTES)

            def test_should_refuse_to_open_it(self, copy_act: Callable[[], SkippedCopy]) -> None:
                with pytest.raises(NotADirectoryError):
                    copy_act()

    class TestTheReadHostFileMethod:
        class TestWhenTheFileIsRegular:
            @pytest.fixture
            def host_file(self, tmp_path: Path) -> WrittenFile:
                (tmp_path / "vocab.txt").write_bytes(FILE_CONTENT)
                (tmp_path / "vocab.txt").chmod(0o444)
                return WorkFiles().read_host_file(
                    PurePosixPath(tmp_path), "vocab.txt", MAX_SIZE_BYTES
                )

            def test_should_give_its_facts_and_its_content(self, host_file: WrittenFile) -> None:
                assert host_file == WrittenFile(
                    FileFacts(FileKind.REGULAR, os.getuid(), 0o444, len(FILE_CONTENT), 1),
                    FILE_CONTENT,
                )

        class TestWhenTheFileIsALink:
            @pytest.fixture
            def host_file(self, tmp_path: Path) -> WrittenFile:
                (tmp_path / "vocab.txt").symlink_to("/etc/passwd")
                return WorkFiles().read_host_file(
                    PurePosixPath(tmp_path), "vocab.txt", MAX_SIZE_BYTES
                )

            def test_should_give_no_content_since_it_follows_no_link(
                self, host_file: WrittenFile
            ) -> None:
                assert (host_file.facts.kind, host_file.content) == (FileKind.LINK, None)

        class TestWhenTheFileIsMissing:
            @pytest.fixture
            def host_file(self, tmp_path: Path) -> WrittenFile:
                return WorkFiles().read_host_file(
                    PurePosixPath(tmp_path), "vocab.txt", MAX_SIZE_BYTES
                )

            def test_should_say_it_is_missing_so_that_the_caller_names_the_step_that_makes_it(
                self, host_file: WrittenFile
            ) -> None:
                assert (host_file.facts.kind, host_file.content) == (FileKind.MISSING, None)
