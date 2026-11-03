import hashlib
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import partial
from pathlib import PurePosixPath

import pytest
from run_step_fakes import FakeStepFiles

from mostt_model_builder.application.build_images import (
    BuiltImage,
    ImageBuildError,
    build_images,
)
from mostt_model_builder.application.run_step import ContainerExit, ImageFacts
from mostt_model_builder.domain.containers.container_steps import CORE_IMAGES, ContainerImage
from mostt_model_builder.domain.containers.provenance import ProvenanceError

WORK_ROOT = PurePosixPath("/data/builds/work")
HEAD_TREE = "3f1d1ffe79bc3336eb583331cfbf9b189fd45f01"
OTHER_TREE = "e949fa3a2eb8bf218e96c2c03625bbd36e4342b1"


def build_image_id(tag: str) -> str:
    return "sha256:" + hashlib.sha256(tag.encode()).hexdigest()


@dataclass
class FakeClone:
    changes: tuple[str, ...] = ()

    def find_head_tree(self) -> str:
        return HEAD_TREE

    def list_changes(self) -> tuple[str, ...]:
        return self.changes


@dataclass
class FakeImageBuilder:
    status: int = 0
    labeled_tree: str = HEAD_TREE
    builds: list[tuple[str, str, PurePosixPath]] = field(default_factory=list)
    images: dict[str, ImageFacts] = field(default_factory=dict)

    def build_image(
        self, image: ContainerImage, tree_hash: str, log_path: PurePosixPath
    ) -> ContainerExit:
        self.builds.append((image.tag, tree_hash, log_path))
        if self.status == 0:
            self.images[image.tag] = ImageFacts(build_image_id(image.tag), self.labeled_tree)
        return ContainerExit(self.status, b"ERROR: failed\n\x1b[0m")

    def inspect_image(self, tag: str) -> ImageFacts | None:
        return self.images.get(tag)


@dataclass(frozen=True)
class BuildRecord:
    built_images: tuple[BuiltImage, ...]
    builds: list[tuple[str, str, PurePosixPath]]


@dataclass(frozen=True)
class RefusedBuild:
    build_act: Callable[[], tuple[BuiltImage, ...]]
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
                builder = FakeImageBuilder()
                built_images = build_images(
                    CORE_IMAGES, WORK_ROOT, FakeClone(), builder, build_files()
                )
                return BuildRecord(built_images, builder.builds)

            def test_should_build_each_image_with_the_tree_of_head_and_a_log_of_its_own(
                self, build_record: BuildRecord
            ) -> None:
                assert build_record.builds == [
                    (
                        image.tag,
                        HEAD_TREE,
                        WORK_ROOT / "image-builds" / f"{image.tag}.log",
                    )
                    for image in CORE_IMAGES
                ]

            def test_should_give_each_images_id(self, build_record: BuildRecord) -> None:
                assert build_record.built_images == tuple(
                    BuiltImage(image.tag, build_image_id(image.tag)) for image in CORE_IMAGES
                )

        class TestWhenTheCloneHoldsAnUntrackedFile:
            @pytest.fixture
            def refused_build(self) -> RefusedBuild:
                builder = FakeImageBuilder()
                return RefusedBuild(
                    partial(
                        build_images,
                        CORE_IMAGES,
                        WORK_ROOT,
                        FakeClone(("notes.txt",)),
                        builder,
                        build_files(),
                    ),
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
