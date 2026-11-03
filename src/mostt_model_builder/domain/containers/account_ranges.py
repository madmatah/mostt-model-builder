import re
from collections.abc import Sequence
from dataclasses import dataclass

FIELD_SEPARATOR = ":"
LINE_SEPARATOR = "\n"
OWNER_GRAMMAR = re.compile(r"[A-Za-z0-9._][A-Za-z0-9._-]{0,31}\$?")
ID_GRAMMAR = re.compile(r"[0-9]{1,10}")
MAX_ID = 2**32 - 1


class AccountRangeError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class SubordinateRange:
    owner: str
    first_id: int
    count: int

    def holds(self, account_id: int) -> bool:
        return self.first_id <= account_id < self.first_id + self.count


@dataclass(frozen=True, slots=True)
class HostAccounts:
    has_container_user_account: bool
    subordinate_user_ranges: tuple[SubordinateRange, ...]
    subordinate_group_ranges: tuple[SubordinateRange, ...]


def parse_subordinate_ranges(text: str, origin: str) -> tuple[SubordinateRange, ...]:
    return tuple(
        parse_subordinate_range(line, f"{origin}, line {line_number}")
        for line_number, line in enumerate(text.split(LINE_SEPARATOR), start=1)
        if line
    )


def parse_subordinate_range(line: str, where: str) -> SubordinateRange:
    fields = line.split(FIELD_SEPARATOR)
    if len(fields) != 3:
        raise AccountRangeError(f"{where} is not <owner>:<first id>:<count>")
    owner, first_id, count = fields
    if not OWNER_GRAMMAR.fullmatch(owner):
        raise AccountRangeError(f"{where}: the owner {owner!a} is not an account's name or id")
    for field in (first_id, count):
        if not ID_GRAMMAR.fullmatch(field) or int(field) > MAX_ID:
            raise AccountRangeError(f"{where}: {field!a} is not an id")
    return SubordinateRange(owner, int(first_id), int(count))


def refuse_held_container_user(accounts: HostAccounts, container_user_id: int) -> None:
    if accounts.has_container_user_account:
        raise AccountRangeError(
            f"an account of this machine holds the user id {container_user_id}, which the "
            "containers run as, so their files would be that account's"
        )
    for kind, ranges in (
        ("user", accounts.subordinate_user_ranges),
        ("group", accounts.subordinate_group_ranges),
    ):
        refuse_held_id(ranges, container_user_id, kind)


def refuse_held_id(ranges: Sequence[SubordinateRange], container_user_id: int, kind: str) -> None:
    for subordinate_range in ranges:
        if subordinate_range.holds(container_user_id):
            raise AccountRangeError(
                f"the subordinate {kind} ids of {subordinate_range.owner!a} hold "
                f"{container_user_id}, which the containers run as, so a container of that "
                "account could reach their files"
            )
