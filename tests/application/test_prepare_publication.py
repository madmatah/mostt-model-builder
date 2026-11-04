import re
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial

import pytest
from accepted_build import (
    COMMIT,
    RUN_DIRECTORY,
    WORK_ROOT,
    build_files,
)
from card_arrangement import CARD_PATH, FakeClone, build_card_request
from run_step_fakes import FakeStepFiles, build_file_facts

from mostt_model_builder.application.prepare_publication import (
    CardOutcome,
    PreparedPublication,
    PublicationError,
    prepare_publication,
    write_model_card,
)
from mostt_model_builder.application.verify_build import BuildRejectedError
from mostt_model_builder.domain.publication.model_card import CardError


@dataclass(frozen=True)
class WrittenCard:
    prepared: PreparedPublication
    outcome: CardOutcome
    files: FakeStepFiles


def prepare_and_write(files: FakeStepFiles) -> WrittenCard:
    prepared = prepare_publication(build_card_request(), FakeClone(), files)
    return WrittenCard(prepared, write_model_card(prepared, WORK_ROOT, files), files)


class TestThePublicationPreparation:
    class TestTheWriteModelCardFunction:
        class TestWhenTheBuildIsAcceptedAndNoCardIsWritten:
            @pytest.fixture
            def written_card(self) -> WrittenCard:
                return prepare_and_write(build_files())

            def test_should_write_the_rendered_card_beside_the_verification(
                self, written_card: WrittenCard
            ) -> None:
                assert written_card.files.contents[CARD_PATH] == written_card.prepared.card

            def test_should_leave_it_read_only_as_the_record_that_the_publication_matches(
                self, written_card: WrittenCard
            ) -> None:
                assert written_card.files.file_modes[CARD_PATH] == 0o444

            def test_should_say_it_wrote_it(self, written_card: WrittenCard) -> None:
                assert written_card.outcome is CardOutcome.WRITTEN

            def test_should_hold_neither_the_commit_nor_the_work_directory(
                self, written_card: WrittenCard
            ) -> None:
                card = written_card.prepared.card.decode()
                assert (COMMIT in card, str(WORK_ROOT) in card) == (False, False)

            def test_should_state_the_versions_that_the_images_of_the_variants_runs_pin(
                self, written_card: WrittenCard
            ) -> None:
                card = written_card.prepared.card.decode()
                assert (
                    "onnx-asr 0.12.0 on the CPU" in card,
                    "0.11.0" in card,
                    "numpy" in card,
                ) == (
                    True,
                    False,
                    False,
                )

        class TestWhenTheSameCardIsWrittenAlready:
            @pytest.fixture
            def outcome(self) -> CardOutcome:
                files = build_files()
                prepare_and_write(files)
                return prepare_and_write(files).outcome

            def test_should_keep_it_and_say_it_is_unchanged(self, outcome: CardOutcome) -> None:
                assert outcome is CardOutcome.UNCHANGED

        class TestWhenAnotherCardIsWrittenAlready:
            @pytest.fixture
            def write_act(self) -> Callable[[], WrittenCard]:
                files = build_files()
                files.add_file(CARD_PATH, build_file_facts(8, 0o444), b"# other\n")
                return partial(prepare_and_write, files)

            def test_should_refuse_it_and_name_clean(
                self, write_act: Callable[[], WrittenCard]
            ) -> None:
                with pytest.raises(
                    PublicationError,
                    match=re.escape(
                        f"{RUN_DIRECTORY}/README.md holds another card than the build's records "
                        "give: remove it with ./model-build.sh --work <dir> --recipe <name> clean"
                    ),
                ):
                    write_act()

    class TestThePreparePublicationFunction:
        class TestWhenTheCloneHoldsAChange:
            @pytest.fixture
            def prepare_act(self) -> Callable[[], PreparedPublication]:
                return partial(
                    prepare_publication,
                    build_card_request(),
                    FakeClone(("images/environment.toml",)),
                    build_files(),
                )

            def test_should_refuse_it_since_the_card_states_the_libraries_of_head(
                self, prepare_act: Callable[[], PreparedPublication]
            ) -> None:
                with pytest.raises(
                    CardError,
                    match=re.escape(
                        "the clone holds 1 uncommitted changes or untracked files, such as "
                        "'images/environment.toml'"
                    ),
                ):
                    prepare_act()

        class TestWhenTheRecipeDescribesNoModel:
            @pytest.fixture
            def prepare_act(self) -> Callable[[], PreparedPublication]:
                return partial(
                    prepare_publication,
                    build_card_request(model=None),
                    FakeClone(),
                    build_files(),
                )

            def test_should_refuse_it_and_name_the_missing_description(
                self, prepare_act: Callable[[], PreparedPublication]
            ) -> None:
                with pytest.raises(
                    PublicationError,
                    match=re.escape("the recipe family describes no model: its contract gives no"),
                ):
                    prepare_act()

        class TestWhenTheBuildIsRejected:
            @pytest.fixture
            def prepare_act(self) -> Callable[[], PreparedPublication]:
                return partial(
                    prepare_publication,
                    build_card_request(is_rejected=True),
                    FakeClone(),
                    build_files(),
                )

            def test_should_refuse_to_render_a_card_of_it(
                self, prepare_act: Callable[[], PreparedPublication]
            ) -> None:
                with pytest.raises(BuildRejectedError, match="the build fails 1 of its rules"):
                    prepare_act()
