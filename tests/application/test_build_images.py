import hashlib
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import partial
from pathlib import PurePosixPath

import pytest
from run_step_fakes import FakeStepFiles

from mostt_model_builder.adapters.git_tree import GitError
from mostt_model_builder.application.build_images import (
    BuiltImage,
    ImageBuildError,
    TarArchive,
    build_images,
)
from mostt_model_builder.application.run_step import ContainerExit, ImageFacts
from mostt_model_builder.domain.containers.container_steps import CORE_IMAGES, ContainerImage
from mostt_model_builder.domain.containers.provenance import ProvenanceError, require_object_name

WORK_ROOT = PurePosixPath("/data/builds/work")
HEAD_TREE = "3f1d1ffe79bc3336eb583331cfbf9b189fd45f01"
OTHER_TREE = "e949fa3a2eb8bf218e96c2c03625bbd36e4342b1"
ARCHIVE = TarArchive(b"the archive of the tree")
ARCHIVE_FAILURE = "git archive failed: fatal: not a tree object"


def build_image_id(tag: str) -> str:
    return "sha256:" + hashlib.sha256(tag.encode()).hexdigest()


@dataclass
class FakeClone:
    changes: tuple[str, ...] = ()
    archived_trees: list[str] = field(default_factory=list)
    can_archive: bool = True

    def find_head_tree(self) -> str:
        return HEAD_TREE

    def list_changes(self) -> tuple[str, ...]:
        return self.changes

    def archive_tree(self, tree_hash: str) -> TarArchive:
        self.archived_trees.append(require_object_name(tree_hash, "the tree to archive"))
        if not self.can_archive:
            raise GitError(ARCHIVE_FAILURE)
        return ARCHIVE


@dataclass
class FakeImageBuilder:
    status: int = 0
    labeled_tree: str = HEAD_TREE
    builds: list[tuple[str, str, TarArchive, PurePosixPath]] = field(default_factory=list)
    images: dict[str, ImageFacts] = field(default_factory=dict)

    def build_image(
        self,
        image: ContainerImage,
        tree_hash: str,
        context_archive: TarArchive,
        log_path: PurePosixPath,
    ) -> ContainerExit:
        self.builds.append((image.tag, tree_hash, context_archive, log_path))
        if self.status == 0:
            self.images[image.tag] = ImageFacts(build_image_id(image.tag), self.labeled_tree)
        return ContainerExit(self.status, b"ERROR: failed\n\x1b[0m")

    def inspect_image(self, tag: str) -> ImageFacts | None:
        return self.images.get(tag)


@dataclass(frozen=True)
class BuildRecord:
    built_images: tuple[BuiltImage, ...]
    builds: list[tuple[str, str, TarArchive, PurePosixPath]]
    archived_trees: list[str]


@dataclass(frozen=True)
class RefusedBuild:
    build_act: Callable[[], tuple[BuiltImage, ...]]
    clone: FakeClone
    builder: FakeImageBuilder


def build_files() -> FakeStepFiles:
    files = FakeStepFiles()
    files.add_directory(WORK_ROOT)
    return files


class TestTheBuildImages:
    class TestTheBuildImagesFunction:
        class TestWhenTheCloneIsClean:
            @pytest.fixture
            def build_record(self) -> BuildRecord:
                clone = FakeClone()
                builder = FakeImageBuilder()
                built_images = build_images(CORE_IMAGES, WORK_ROOT, clone, builder, build_files())
                return BuildRecord(built_images, builder.builds, clone.archived_trees)

            def test_should_build_each_image_from_the_archive_of_its_labeled_tree_with_its_own_log(
                self, build_record: BuildRecord
            ) -> None:
                assert build_record.builds == [
                    (
                        image.tag,
                        HEAD_TREE,
                        ARCHIVE,
                        WORK_ROOT / "image-builds" / f"{image.tag}.log",
                    )
                    for image in CORE_IMAGES
                ]

            def test_should_archive_the_tree_of_head_once_for_every_image(
                self, build_record: BuildRecord
            ) -> None:
                assert build_record.archived_trees == [HEAD_TREE]

            def test_should_give_each_images_id(self, build_record: BuildRecord) -> None:
                assert build_record.built_images == tuple(
                    BuiltImage(image.tag, build_image_id(image.tag)) for image in CORE_IMAGES
                )

        class TestWhenTheCloneHoldsAnUntrackedFile:
            @pytest.fixture
            def refused_build(self) -> RefusedBuild:
                clone = FakeClone(("notes.txt",))
                builder = FakeImageBuilder()
                return RefusedBuild(
                    partial(build_images, CORE_IMAGES, WORK_ROOT, clone, builder, build_files()),
                    clone,
                    builder,
                )

            def test_should_refuse_it_before_any_build_since_no_commit_names_the_file(
                self, refused_build: RefusedBuild
            ) -> None:
                with pytest.raises(
                    ProvenanceError,
                    match=re.escape(
                        "the clone holds 1 uncommitted changes or untracked files, "
                        "such as 'notes.txt'"
                    ),
                ):
                    refused_build.build_act()
                assert refused_build.builder.builds == []

            def test_should_archive_nothing(self, refused_build: RefusedBuild) -> None:
                with pytest.raises(ProvenanceError, match="the clone holds"):
                    refused_build.build_act()
                assert refused_build.clone.archived_trees == []

        class TestWhenTheTreeCannotBeArchived:
            @pytest.fixture
            def refused_build(self) -> RefusedBuild:
                clone = FakeClone(can_archive=False)
                builder = FakeImageBuilder()
                return RefusedBuild(
                    partial(build_images, CORE_IMAGES, WORK_ROOT, clone, builder, build_files()),
                    clone,
                    builder,
                )

            def test_should_build_no_image_since_no_context_holds_the_tree(
                self, refused_build: RefusedBuild
            ) -> None:
                with pytest.raises(GitError, match=f"^{re.escape(ARCHIVE_FAILURE)}$"):
                    refused_build.build_act()
                assert refused_build.builder.builds == []

        class TestWhenABuildFails:
            @pytest.fixture
            def build_act(self) -> Callable[[], tuple[BuiltImage, ...]]:
                return partial(
                    build_images,
                    CORE_IMAGES,
                    WORK_ROOT,
                    FakeClone(),
                    FakeImageBuilder(status=1),
                    build_files(),
                )

            def test_should_show_the_end_of_its_log_through_the_filter(
                self, build_act: Callable[[], tuple[BuiltImage, ...]]
            ) -> None:
                with pytest.raises(
                    ImageBuildError,
                    match=re.escape(
                        "the end of its log, image-builds/mostt-model-builder-core-bench.log:\n"
                        "ERROR: failed\n?[0m"
                    ),
                ):
                    build_act()

        class TestWhenTheBuiltImageNamesAnotherTree:
            @pytest.fixture
            def build_act(self) -> Callable[[], tuple[BuiltImage, ...]]:
                return partial(
                    build_images,
                    CORE_IMAGES,
                    WORK_ROOT,
                    FakeClone(),
                    FakeImageBuilder(labeled_tree=OTHER_TREE),
                    build_files(),
                )

            def test_should_refuse_it(
                self, build_act: Callable[[], tuple[BuiltImage, ...]]
            ) -> None:
                with pytest.raises(ProvenanceError, match="was built from the tree"):
                    build_act()
