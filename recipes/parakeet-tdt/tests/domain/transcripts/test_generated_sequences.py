import re
from collections.abc import Callable
from functools import partial

import pytest

from mostt_recipe_parakeet_tdt.domain.transcripts.generated_sequences import (
    GeneratedSequenceError,
    read_generated_token_ids,
)

BLANK_ID = 8192
UNKNOWN_ID = 0
THE_ID = 21
CAT_ID = 1937
SLEEPS_ID = 4410
ABOVE_THE_BLANK_ID = 8193


class TestTheGeneratedSequences:
    class TestTheReadGeneratedTokenIdsFunction:
        class TestWhenTheSequenceStartsWithTheBlankAndHoldsBlanks:
            @pytest.fixture
            def token_ids(self) -> tuple[int, ...]:
                return read_generated_token_ids(
                    (BLANK_ID, THE_ID, BLANK_ID, CAT_ID, BLANK_ID, BLANK_ID, SLEEPS_ID), BLANK_ID
                )

            def test_should_keep_the_tokens_in_their_order_without_a_blank(
                self, token_ids: tuple[int, ...]
            ) -> None:
                assert token_ids == (THE_ID, CAT_ID, SLEEPS_ID)

        class TestWhenTheSequenceHoldsTheUnknownToken:
            @pytest.fixture
            def token_ids(self) -> tuple[int, ...]:
                return read_generated_token_ids((BLANK_ID, THE_ID, UNKNOWN_ID, CAT_ID), BLANK_ID)

            def test_should_keep_id_0_as_nemos_ids_keep_it(
                self, token_ids: tuple[int, ...]
            ) -> None:
                assert token_ids == (THE_ID, UNKNOWN_ID, CAT_ID)

        class TestWhenTheSequenceHoldsAnIdAboveTheBlank:
            @pytest.fixture
            def read_act(self) -> Callable[[], tuple[int, ...]]:
                return partial(
                    read_generated_token_ids, (BLANK_ID, THE_ID, ABOVE_THE_BLANK_ID), BLANK_ID
                )

            def test_should_refuse_it_since_a_token_is_never_above_the_blank(
                self, read_act: Callable[[], tuple[int, ...]]
            ) -> None:
                with pytest.raises(
                    GeneratedSequenceError,
                    match=re.escape(
                        "the generated sequence holds the id 8193, above the blank 8192"
                    ),
                ):
                    read_act()
