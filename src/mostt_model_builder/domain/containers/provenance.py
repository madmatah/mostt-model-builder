import re
from collections.abc import Sequence

IMAGE_TREE_LABEL = "mostt-model-builder.tree"
TREE_HASH_BUILD_ARGUMENT = "TREE_HASH"
OBJECT_NAME_GRAMMAR = re.compile(r"[0-9a-f]{40}|[0-9a-f]{64}")
IMAGE_ID_GRAMMAR = re.compile(r"sha256:[0-9a-f]{64}")


class ProvenanceError(ValueError):
    pass


def require_object_name(text: str, what: str) -> str:
    if not OBJECT_NAME_GRAMMAR.fullmatch(text):
        raise ProvenanceError(f"{what} {text!a} is not a git object name")
    return text


def refuse_unclean_clone(changed_paths: Sequence[str]) -> None:
    if changed_paths:
        raise ProvenanceError(
            f"the clone holds {len(changed_paths)} uncommitted changes or untracked files, "
            f"such as {changed_paths[0]!a}: an image is built only from a commit, so that its "
            "label names what it holds"
        )


def refuse_foreign_image(tag: str, tree_label: str | None, head_tree: str) -> None:
    if tree_label is None:
        raise ProvenanceError(f"the image {tag} carries no {IMAGE_TREE_LABEL} label")
    if tree_label != head_tree:
        raise ProvenanceError(
            f"the image {tag} was built from the tree {tree_label!a}, not from the tree "
            f"{head_tree} of HEAD: build the images again"
        )
