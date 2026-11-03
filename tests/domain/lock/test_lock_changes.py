import pytest

from mostt_model_builder.domain.lock.lock_changes import (
    LockChanges,
    VersionChange,
    compare_locks,
    describe_lock_changes,
)
from mostt_model_builder.domain.lock.lock_check import LockedPackage

PREVIOUS = (
    LockedPackage("numpy", "2.5.3"),
    LockedPackage("onnx", "1.23.2"),
    LockedPackage("six", "1.17.0"),
)
CURRENT = (
    LockedPackage("numpy", "2.5.4"),
    LockedPackage("onnx", "1.23.2"),
    LockedPackage("torch", "2.15.0"),
)


class TestTheLockChanges:
    class TestTheCompareLocksFunction:
        class TestWhenAPackageIsAddedOneRemovedAndOneChanged:
            @pytest.fixture
            def changes(self) -> LockChanges:
                return compare_locks(PREVIOUS, CURRENT)

            def test_should_name_each_change_and_leave_the_unchanged_out(
                self, changes: LockChanges
            ) -> None:
                assert changes == LockChanges(
                    added=(LockedPackage("torch", "2.15.0"),),
                    removed=(LockedPackage("six", "1.17.0"),),
                    changed=(VersionChange("numpy", ("2.5.3",), ("2.5.4",)),),
                )

        class TestWhenNoLockWasCommittedYet:
            @pytest.fixture
            def changes(self) -> LockChanges:
                return compare_locks(None, CURRENT)

            def test_should_add_every_package(self, changes: LockChanges) -> None:
                assert changes.added == CURRENT

    class TestTheDescribeLockChangesFunction:
        class TestWhenPackagesChanged:
            @pytest.fixture
            def lines(self) -> list[str]:
                return describe_lock_changes(compare_locks(PREVIOUS, CURRENT))

            def test_should_give_one_line_per_change(self, lines: list[str]) -> None:
                assert lines == [
                    "added torch 2.15.0",
                    "removed six 1.17.0",
                    "changed numpy from 2.5.3 to 2.5.4",
                ]

        class TestWhenNothingChanged:
            @pytest.fixture
            def lines(self) -> list[str]:
                return describe_lock_changes(compare_locks(CURRENT, CURRENT))

            def test_should_say_so(self, lines: list[str]) -> None:
                assert lines == ["no package is added, removed or changed in version"]
