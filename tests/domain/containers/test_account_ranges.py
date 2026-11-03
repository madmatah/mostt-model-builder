import re
from collections.abc import Callable
from functools import partial

import pytest

from mostt_model_builder.domain.containers.account_ranges import (
    AccountRangeError,
    HostAccounts,
    SubordinateRange,
    parse_subordinate_ranges,
    refuse_held_container_user,
)

CONTAINER_USER_ID = 65531
ORIGIN = "/etc/subuid"
USER_RANGE = SubordinateRange("user", 100_000, 65_536)
RANGE_HOLDING_THE_CONTAINER_USER = SubordinateRange("other", 65_000, 1_000)
FREE_ACCOUNTS = HostAccounts(False, (USER_RANGE,), (USER_RANGE,))
MALFORMED_LINES = {
    "two fields": "user:100000",
    "four fields": "user:100000:65536:1",
    "a negative id": "user:-1:65536",
    "an id beyond 32 bits": "user:4294967296:1",
    "a space": "user:100000: 65536",
    "an owner with a slash": "us/er:100000:65536",
}
HELD_ACCOUNTS = {
    "an account": (
        HostAccounts(True, (), ()),
        "an account of this machine holds the user id 65531",
    ),
    "a subordinate user range": (
        HostAccounts(False, (RANGE_HOLDING_THE_CONTAINER_USER,), ()),
        "the subordinate user ids of 'other' hold 65531",
    ),
    "a subordinate group range": (
        HostAccounts(False, (), (RANGE_HOLDING_THE_CONTAINER_USER,)),
        "the subordinate group ids of 'other' hold 65531",
    ),
}


class TestTheAccountRanges:
    class TestTheParseSubordinateRangesFunction:
        class TestWhenTheFileHoldsOneRangePerLine:
            @pytest.fixture
            def ranges(self) -> tuple[SubordinateRange, ...]:
                return parse_subordinate_ranges("user:100000:65536\n1001:165536:65536\n", ORIGIN)

            def test_should_read_each_owner_first_id_and_count(
                self, ranges: tuple[SubordinateRange, ...]
            ) -> None:
                assert ranges == (
                    SubordinateRange("user", 100_000, 65_536),
                    SubordinateRange("1001", 165_536, 65_536),
                )

        @pytest.mark.parametrize(
            "line", list(MALFORMED_LINES.values()), ids=list(MALFORMED_LINES.keys())
        )
        class TestWhenALineIsMalformed:
            @pytest.fixture
            def parse_act(self, line: str) -> Callable[[], tuple[SubordinateRange, ...]]:
                return partial(parse_subordinate_ranges, f"{line}\n", ORIGIN)

            def test_should_refuse_it_naming_its_line(
                self, parse_act: Callable[[], tuple[SubordinateRange, ...]]
            ) -> None:
                with pytest.raises(AccountRangeError, match=f"^{re.escape(ORIGIN)}, line 1"):
                    parse_act()

    class TestTheRefuseHeldContainerUserFunction:
        class TestWhenNoAccountAndNoRangeHoldsTheContainerUser:
            @pytest.fixture
            def refuse_act(self) -> Callable[[], object]:
                return partial(refuse_held_container_user, FREE_ACCOUNTS, CONTAINER_USER_ID)

            def test_should_let_the_containers_run(self, refuse_act: Callable[[], object]) -> None:
                assert refuse_act() is None

        @pytest.mark.parametrize(
            ("accounts", "reason"), list(HELD_ACCOUNTS.values()), ids=list(HELD_ACCOUNTS.keys())
        )
        class TestWhenTheContainerUserIsHeld:
            @pytest.fixture
            def refuse_act(self, accounts: HostAccounts) -> Callable[[], object]:
                return partial(refuse_held_container_user, accounts, CONTAINER_USER_ID)

            def test_should_refuse_to_run_and_name_the_holder(
                self, refuse_act: Callable[[], object], reason: str
            ) -> None:
                with pytest.raises(AccountRangeError, match=re.escape(reason)):
                    refuse_act()

    class TestTheSubordinateRange:
        class TestTheHoldsMethod:
            @pytest.mark.parametrize(
                ("account_id", "is_held"),
                [
                    pytest.param(99_999, False, id="just before the range"),
                    pytest.param(100_000, True, id="its first id"),
                    pytest.param(165_535, True, id="its last id"),
                    pytest.param(165_536, False, id="just after the range"),
                ],
            )
            class TestWhenAnIdLiesAroundTheRange:
                @pytest.fixture
                def holds(self, account_id: int) -> bool:
                    return USER_RANGE.holds(account_id)

                def test_should_hold_only_the_ids_from_its_first_for_its_count(
                    self, holds: bool, is_held: bool
                ) -> None:
                    assert holds is is_held
