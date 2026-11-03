from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Protocol

from mostt_model_builder.application.run_step import ContainerExit, ImageFacts, StepFiles
from mostt_model_builder.domain.containers.container_steps import ContainerImage
from mostt_model_builder.domain.containers.printable_output import show_printable
from mostt_model_builder.domain.containers.provenance import (
    refuse_foreign_image,
    refuse_unclean_clone,
)
from mostt_model_builder.domain.containers.work_directory import locate_image_build_log


class ImageBuildError(Exception):
    pass


class ImageBuilder(Protocol):
    def build_image(
        self, image: ContainerImage, tree_hash: str, log_path: PurePosixPath
    ) -> ContainerExit: ...

    def inspect_image(self, tag: str) -> ImageFacts | None: ...


class CloneState(Protocol):
    def find_head_tree(self) -> str: ...

    def list_changes(self) -> tuple[str, ...]: ...


@dataclass(frozen=True, slots=True)
class BuiltImage:
    tag: str
    image_id: str


def build_images(
    images: Sequence[ContainerImage],
    work_root: PurePosixPath,
    clone: CloneState,
    builder: ImageBuilder,
    files: StepFiles,
) -> tuple[BuiltImage, ...]:
    refuse_unclean_clone(clone.list_changes())
    tree_hash = clone.find_head_tree()
    return tuple(build_image(image, tree_hash, work_root, builder, files) for image in images)


def build_image(
    image: ContainerImage,
    tree_hash: str,
    work_root: PurePosixPath,
    builder: ImageBuilder,
    files: StepFiles,
) -> BuiltImage:
    log_path = locate_image_build_log(image.tag)
    files.make_directories(work_root, log_path.parent)
    build_exit = builder.build_image(image, tree_hash, work_root / log_path)
    if build_exit.status != 0:
        raise ImageBuildError(
            f"the build of {image.tag} exited with the status {build_exit.status}; "
            f"the end of its log, {log_path}:\n{show_printable(build_exit.log_tail)}"
        )
    image_facts = builder.inspect_image(image.tag)
    if image_facts is None:
        raise ImageBuildError(f"the build of {image.tag} left no image under that tag")
    refuse_foreign_image(image.tag, image_facts.tree_label, tree_hash)
    return BuiltImage(image.tag, image_facts.image_id)
