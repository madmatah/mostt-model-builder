import re
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from pathlib import PurePosixPath

import pytest
from cache_arrangement import (
    ENCODER,
    HARDWARE_WITHOUT_GPU,
    LANGUAGES,
    TREE_FILES,
    VOCABULARY,
    WORK_ROOT,
    CacheBuild,
    list_entries,
    prune_built_cache,
)
from run_step_fakes import build_file_facts

from mostt_model_builder.application.cache_entries import CacheEntryError
from mostt_model_builder.application.prune_cache import PrunedEntry

TRANSCRIBE_READER = PurePosixPath("recipes/family/src/mostt_recipe_family/domain/reader.py")
CHANGED_TREE = {**TREE_FILES, TRANSCRIBE_READER: b"READ = 2\n"}
TRANSCRIBE_ROWS = sorted(
    f"transcribe-float32-{device}-{language}-0"
    for device in ("cpu", "gpu")
    for language in LANGUAGES
)
UNREADABLE_RECORD = b"{not json"
CACHE_DIRECTORY = WORK_ROOT / "cache/family"
STALE_KEY = "5c0d8a3b9e21" + "0" * 52
CLEANING_LOG = CACHE_DIRECTORY / "export" / f"clean-{STALE_KEY}.log"
PUBLICATION_REPORT = WORK_ROOT / "builds/family/publications/1/output/publication.json"


@dataclass(frozen=True)
class PrunedCache:
    pruned: tuple[PrunedEntry, ...]
    entries_before: list[PurePosixPath]
    entries_after: list[PurePosixPath]
    has_publication: bool


@dataclass(frozen=True)
class RefusedPruning:
    prune: Callable[[], object]
    cache: CacheBuild


def copy_candidate_to_stale_key(cache: CacheBuild) -> None:
    candidate_directory = CACHE_DIRECTORY / "candidate"
    for path in [
        path
        for path in cache.files.entries
        if path.is_relative_to(candidate_directory) and path != candidate_directory
    ]:
        current_key = path.relative_to(candidate_directory).parts[0]
        stale_path = (
            candidate_directory / STALE_KEY / path.relative_to(candidate_directory / current_key)
        )
        cache.files.entries[stale_path] = cache.files.entries[path]
        if path in cache.files.contents:
            cache.files.contents[stale_path] = cache.files.contents[path]


def build_twice() -> CacheBuild:
    cache = CacheBuild()
    cache.resolve()
    cache.resolve(tree_files=CHANGED_TREE)
    cache.files.add_file(PUBLICATION_REPORT, build_file_facts(2, 0o640), b"{}")
    return cache


class TestTheCachePruning:
    class TestThePruneCacheFunction:
        class TestWhenAnEarlierBuildLeftEntriesTheCurrentOneDoesNotUse:
            @pytest.fixture
            def pruned_cache(self) -> PrunedCache:
                cache = build_twice()
                entries_before = list_entries(cache.files)
                pruned = prune_built_cache(cache, tree_files=CHANGED_TREE)
                return PrunedCache(
                    pruned,
                    entries_before,
                    list_entries(cache.files),
                    PUBLICATION_REPORT in cache.files.entries,
                )

            def test_should_remove_the_earlier_entries_alone(
                self, pruned_cache: PrunedCache
            ) -> None:
                assert sorted(entry.name for entry in pruned_cache.pruned) == TRANSCRIBE_ROWS

            def test_should_keep_every_entry_of_the_current_build(
                self, pruned_cache: PrunedCache
            ) -> None:
                assert (len(pruned_cache.entries_before), len(pruned_cache.entries_after)) == (
                    12,
                    8,
                )

            def test_should_give_the_bytes_of_each_removed_entry(
                self, pruned_cache: PrunedCache
            ) -> None:
                assert all(entry.size_bytes > 0 for entry in pruned_cache.pruned)

            def test_should_never_touch_a_publication(self, pruned_cache: PrunedCache) -> None:
                assert pruned_cache.has_publication

        class TestWhenAnEarlierBuildLeftACandidate:
            @pytest.fixture
            def pruned(self) -> tuple[PrunedEntry, ...]:
                cache = CacheBuild()
                cache.resolve()
                copy_candidate_to_stale_key(cache)
                return prune_built_cache(cache)

            def test_should_remove_it_with_the_bytes_of_its_copies(
                self, pruned: tuple[PrunedEntry, ...]
            ) -> None:
                assert pruned == (
                    PrunedEntry("candidate", STALE_KEY, len(ENCODER) + len(VOCABULARY)),
                )

        class TestWhenACleaningLeftItsLog:
            @pytest.fixture
            def pruned(self) -> tuple[PrunedEntry, ...]:
                cache = build_twice()
                cache.files.add_file(CLEANING_LOG, build_file_facts(0, 0o600), b"")
                return prune_built_cache(cache, tree_files=CHANGED_TREE)

            def test_should_go_on_past_it_and_prune_the_earlier_entries(
                self, pruned: tuple[PrunedEntry, ...]
            ) -> None:
                assert sorted(entry.name for entry in pruned) == TRANSCRIBE_ROWS

        @pytest.mark.parametrize(
            "foreign_path",
            [
                pytest.param(CACHE_DIRECTORY / "Notes", id="a name outside the entries' grammar"),
                pytest.param(CACHE_DIRECTORY / "export" / "notes.txt", id="an unknown file"),
            ],
        )
        class TestWhenTheCacheHoldsAForeignName:
            @pytest.fixture
            def refused_pruning(self, foreign_path: PurePosixPath) -> RefusedPruning:
                cache = build_twice()
                cache.files.add_file(foreign_path, build_file_facts(0, 0o600), b"")
                return RefusedPruning(partial(prune_built_cache, cache, CHANGED_TREE), cache)

            def test_should_refuse_it_name_it_and_remove_nothing(
                self, refused_pruning: RefusedPruning, foreign_path: PurePosixPath
            ) -> None:
                entries_before = list_entries(refused_pruning.cache.files)
                with pytest.raises(
                    CacheEntryError, match=re.escape(f"holds {foreign_path.name!a}")
                ):
                    refused_pruning.prune()
                assert list_entries(refused_pruning.cache.files) == entries_before

        class TestWhenTheCurrentBuildIsIncomplete:
            @pytest.fixture
            def refused_pruning(self) -> RefusedPruning:
                cache = build_twice()
                resolved, _ = cache.resolve(tree_files=CHANGED_TREE)
                entry = PurePosixPath("cache/family/export", resolved.entries["export"].key)
                cache.files.remove_file_if_present(WORK_ROOT / entry / "record.json")
                return RefusedPruning(partial(prune_built_cache, cache, CHANGED_TREE), cache)

            def test_should_refuse_it_name_the_build_and_remove_nothing(
                self, refused_pruning: RefusedPruning
            ) -> None:
                entries_before = list_entries(refused_pruning.cache.files)
                with pytest.raises(
                    CacheEntryError,
                    match=re.escape("the current build lacks a complete entry of export")
                    + ".*"
                    + re.escape("--recipe <name> build"),
                ):
                    refused_pruning.prune()
                assert list_entries(refused_pruning.cache.files) == entries_before

        class TestWhenTheRecordOfAnEntryOfTheCurrentBuildIsRefused:
            @pytest.fixture
            def refused_pruning(self) -> RefusedPruning:
                cache = build_twice()
                resolved, _ = cache.resolve(tree_files=CHANGED_TREE)
                record_path = (
                    WORK_ROOT
                    / "cache/family/export"
                    / resolved.entries["export"].key
                    / "record.json"
                )
                cache.files.contents[record_path] = UNREADABLE_RECORD
                cache.files.entries[record_path] = build_file_facts(len(UNREADABLE_RECORD), 0o444)
                return RefusedPruning(partial(prune_built_cache, cache, CHANGED_TREE), cache)

            def test_should_refuse_name_the_cleaning_and_remove_nothing(
                self, refused_pruning: RefusedPruning
            ) -> None:
                entries_before = list_entries(refused_pruning.cache.files)
                with pytest.raises(
                    CacheEntryError,
                    match=re.escape("the entry cache/family/export/")
                    + ".*"
                    + re.escape("--recipe <name> clean export"),
                ):
                    refused_pruning.prune()
                assert list_entries(refused_pruning.cache.files) == entries_before

        class TestWhenNoGpuIsReadable:
            @pytest.fixture
            def refused_pruning(self) -> RefusedPruning:
                cache = build_twice()
                return RefusedPruning(
                    partial(prune_built_cache, cache, CHANGED_TREE, HARDWARE_WITHOUT_GPU), cache
                )

            def test_should_refuse_and_remove_nothing(
                self, refused_pruning: RefusedPruning
            ) -> None:
                entries_before = list_entries(refused_pruning.cache.files)
                with pytest.raises(
                    CacheEntryError, match=re.escape("/proc/driver/nvidia holds no GPU")
                ):
                    refused_pruning.prune()
                assert list_entries(refused_pruning.cache.files) == entries_before
