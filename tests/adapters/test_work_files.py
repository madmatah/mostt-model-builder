import hashlib
import os
import stat
from collections.abc import Callable
from functools import partial
from pathlib import Path, PurePosixPath

import pytest

from mostt_model_builder.adapters.work_files import WorkFiles
from mostt_model_builder.application.fetch_sources import FileDigest, SourceFileError
from mostt_model_builder.domain.containers.work_directory import DirectoryFacts, EntryKind

FILE_CONTENT = b"id\tfile\ttranscription\n"


def build_work_root(tmp_path: Path) -> Path:
    work_root = tmp_path / "work"
    work_root.mkdir(mode=0o700)
    work_root.chmod(0o700)
    return work_root


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
