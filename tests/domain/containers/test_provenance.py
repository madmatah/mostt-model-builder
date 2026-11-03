import re
from collections.abc import Callable
from functools import partial

import pytest

from mostt_model_builder.domain.containers.provenance import (
    ProvenanceError,
    refuse_foreign_image,
    refuse_unclean_clone,
    require_object_name,
)

TAG = "mostt-model-builder-core-bench"
HEAD_TREE = "3f1d1ffe79bc3336eb583331cfbf9b189fd45f01"
OTHER_TREE = "e949fa3a2eb8bf218e96c2c03625bbd36e4342b1"
UNTRACKED_FILE = "src/mostt_model_builder/steps/probe.py"


class TestTheProvenance:
    class TestTheRefuseForeignImageFunction:
        class TestWhenTheLabelNamesTheTreeOfHead:
            @pytest.fixture
            def refuse_act(self) -> Callable[[], object]:
                return partial(refuse_foreign_image, TAG, HEAD_TREE, HEAD_TREE)

            def test_should_accept_the_image(self, refuse_act: Callable[[], object]) -> None:
                assert refuse_act() is None

        class TestWhenTheLabelNamesAnotherTree:
            @pytest.fixture
            def refuse_act(self) -> Callable[[], object]:
                return partial(refuse_foreign_image, TAG, OTHER_TREE, HEAD_TREE)

            def test_should_refuse_the_image_and_ask_for_a_new_build(
                self, refuse_act: Callable[[], object]
            ) -> None:
                with pytest.raises(
                    ProvenanceError,
                    match=re.escape(
                        f"the image {TAG} was built from the tree '{OTHER_TREE}', not from the "
                        f"tree {HEAD_TREE} of HEAD: build the images again"
                    ),
                ):
                    refuse_act()

        class TestWhenTheImageCarriesNoLabel:
            @pytest.fixture
            def refuse_act(self) -> Callable[[], object]:
                return partial(refuse_foreign_image, TAG, None, HEAD_TREE)

            def test_should_refuse_the_image(self, refuse_act: Callable[[], object]) -> None:
                with pytest.raises(
                    ProvenanceError, match=re.escape("carries no mostt-model-builder.tree")
                ):
                    refuse_act()

    class TestTheRefuseUncleanCloneFunction:
        class TestWhenTheCloneHoldsNoChange:
            @pytest.fixture
            def refuse_act(self) -> Callable[[], object]:
                return partial(refuse_unclean_clone, ())

            def test_should_let_the_build_go_on(self, refuse_act: Callable[[], object]) -> None:
                assert refuse_act() is None

        class TestWhenTheCloneHoldsAnUntrackedFile:
            @pytest.fixture
            def refuse_act(self) -> Callable[[], object]:
                return partial(refuse_unclean_clone, (UNTRACKED_FILE,))

            def test_should_refuse_the_build_since_the_label_would_not_name_its_content(
                self, refuse_act: Callable[[], object]
            ) -> None:
                with pytest.raises(
                    ProvenanceError,
                    match=re.escape(
                        f"1 uncommitted changes or untracked files, such as '{UNTRACKED_FILE}'"
                    ),
                ):
                    refuse_act()

    class TestTheRequireObjectNameFunction:
        @pytest.mark.parametrize(
            "text", ["HEAD", HEAD_TREE[:-1], HEAD_TREE.upper(), f"{HEAD_TREE}\n"]
        )
        class TestWhenTheTextIsNotAnObjectName:
            @pytest.fixture
            def require_act(self, text: str) -> Callable[[], str]:
                return partial(require_object_name, text, "the tree of HEAD")

            def test_should_refuse_it(self, require_act: Callable[[], str]) -> None:
                with pytest.raises(ProvenanceError, match="is not a git object name"):
                    require_act()
