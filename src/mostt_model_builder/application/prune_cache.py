from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import PurePosixPath

from mostt_model_builder.application.cache_entries import (
    BUILD_COMMAND,
    CacheEntryError,
    EntryKeys,
    EntryOfRecords,
    EntryState,
    SkipCause,
    key_entries_from_records,
)
from mostt_model_builder.application.clean_run import (
    BuildCleaningFiles,
    EntryCleaning,
    clean_entry,
    is_judged_directory,
    remove_candidate_entry,
)
from mostt_model_builder.application.run_step import ContainerEngine, HostIdentity
from mostt_model_builder.domain.cache.build_units import BuildUnit
from mostt_model_builder.domain.cache.hardware_facts import NO_GPU_REASON, HardwareFacts
from mostt_model_builder.domain.cache.output_records import ENTRY_NAME_GRAMMAR, is_key
from mostt_model_builder.domain.containers.container_files import FileKind
from mostt_model_builder.domain.containers.work_directory import (
    CANDIDATE_ENTRY_NAME,
    CANDIDATE_FILES_DIRECTORY_NAME,
    ENTRY_OUTPUT_DIRECTORY_NAME,
    is_cleaning_log_name,
    locate_cache,
    locate_entry,
)
from mostt_model_builder.domain.contract.recipe_contract import JudgedRecipe


@dataclass(frozen=True, slots=True)
class CachePruning:
    recipe: JudgedRecipe
    units: Sequence[BuildUnit]
    work_root: PurePosixPath
    head_tree: str
    host: HostIdentity
    tree_files: Mapping[PurePosixPath, bytes]
    hardware: HardwareFacts | None


@dataclass(frozen=True, slots=True)
class PrunedEntry:
    name: str
    key: str
    size_bytes: int


# Only a complete build tells the entries it uses from the others: an entry that it lacks could
# come back with the same bytes, and its later entries with their old keys, once it runs again.
def prune_cache(
    request: CachePruning,
    engine: ContainerEngine,
    files: BuildCleaningFiles,
    report: Callable[[PrunedEntry], None],
) -> tuple[PrunedEntry, ...]:
    recipe = request.recipe
    user_id = request.host.user_id
    entries = key_entries_from_records(
        recipe,
        request.units,
        EntryKeys(recipe, request.tree_files, request.hardware),
        request.work_root,
        user_id,
        files,
    )
    refuse_incomplete_build(recipe, entries)
    kept = {(entry.unit.name, entry.key) for entry in entries if entry.key is not None}
    order = {entry.unit.name: position for position, entry in enumerate(entries)}
    cache_directory = locate_cache(recipe.name)
    if not is_judged_directory(request.work_root, cache_directory, user_id, files):
        return ()
    entry_names = sorted(
        files.list_entry_names(request.work_root / cache_directory),
        key=lambda name: (order.get(name, len(order)), name),
    )
    stale_entries = [
        (entry_name, key)
        for entry_name in entry_names
        for key in list_stale_keys(request, entry_name, kept, files)
    ]
    pruned = []
    for entry_name, key in stale_entries:
        size_bytes = measure_entry(request, entry_name, key, files)
        remove_entry(request, entry_name, key, engine, files)
        pruned_entry = PrunedEntry(entry_name, key, size_bytes)
        report(pruned_entry)
        pruned.append(pruned_entry)
    return tuple(pruned)


def refuse_incomplete_build(recipe: JudgedRecipe, entries: Sequence[EntryOfRecords]) -> None:
    for entry in entries:
        if entry.unit.step not in recipe.build_order:
            continue
        if entry.skip is not None and entry.skip.cause is SkipCause.NO_GPU:
            raise CacheEntryError(
                f"prune cannot tell the entries of {entry.unit.name} from old ones: "
                f"{NO_GPU_REASON}; load the NVIDIA module, then prune"
            )
        if entry.refusal is not None:
            raise CacheEntryError(f"prune cannot go on: {entry.refusal}, then prune")
        if entry.state is not EntryState.COMPLETE:
            raise CacheEntryError(
                f"the current build lacks a complete entry of {entry.unit.name}, whose old "
                f"entries prune would remove: make it with {BUILD_COMMAND}, then prune"
            )


def list_stale_keys(
    request: CachePruning,
    entry_name: str,
    kept: set[tuple[str, str]],
    files: BuildCleaningFiles,
) -> list[str]:
    if not ENTRY_NAME_GRAMMAR.fullmatch(entry_name):
        raise CacheEntryError(f"the cache of {request.recipe.name} holds {entry_name!a}, no row's")
    row_directory = locate_cache(request.recipe.name) / entry_name
    if not is_judged_directory(request.work_root, row_directory, request.host.user_id, files):
        return []
    stale = []
    for name in files.list_entry_names(request.work_root / row_directory):
        if is_key(name):
            if (entry_name, name) not in kept:
                stale.append(name)
        elif not is_cleaning_log_name(name):
            raise CacheEntryError(
                f"{row_directory} holds {name!a}, neither an entry nor the log of a cleaning"
            )
    return stale


# The size of an entry is that of the files its row or the host left in it, read from their
# facts alone, since a pruned entry is neither judged again nor hashed.
def measure_entry(
    request: CachePruning, entry_name: str, key: str, files: BuildCleaningFiles
) -> int:
    content_name = (
        CANDIDATE_FILES_DIRECTORY_NAME
        if entry_name == CANDIDATE_ENTRY_NAME
        else ENTRY_OUTPUT_DIRECTORY_NAME
    )
    directory = locate_entry(request.recipe.name, entry_name, key) / content_name
    if not is_judged_directory(request.work_root, directory, request.host.user_id, files):
        return 0
    size_bytes = 0
    for name in files.list_entry_names(request.work_root / directory):
        facts = files.inspect_components(request.work_root, directory / name)[-1].facts
        if facts.kind is FileKind.REGULAR:
            size_bytes += facts.size_bytes
    return size_bytes


def remove_entry(
    request: CachePruning,
    entry_name: str,
    key: str,
    engine: ContainerEngine,
    files: BuildCleaningFiles,
) -> None:
    if entry_name == CANDIDATE_ENTRY_NAME:
        remove_candidate_entry(
            request.recipe.name, key, request.work_root, request.host.user_id, files
        )
        return
    clean_entry(
        EntryCleaning(
            recipe_name=request.recipe.name,
            entry_name=entry_name,
            key=key,
            work_root=request.work_root,
            head_tree=request.head_tree,
            host=request.host,
        ),
        engine,
        files,
    )
