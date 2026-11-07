from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import PurePosixPath
from typing import Protocol

from mostt_model_builder.application.cache_entries import (
    EntryKeys,
    EntryOfRecords,
    RecordFiles,
    key_entries_from_records,
)
from mostt_model_builder.application.run_step import (
    ContainerEngine,
    HostIdentity,
    StepFiles,
    prepare_container,
    run_judged_container,
)
from mostt_model_builder.domain.cache.build_units import BuildUnit
from mostt_model_builder.domain.cache.hardware_facts import HardwareFacts
from mostt_model_builder.domain.cache.output_records import RECORD_FILE_NAME
from mostt_model_builder.domain.containers.container_files import FileKind, judge_mount
from mostt_model_builder.domain.containers.container_steps import (
    ContainerImage,
    build_clean_row,
    plan_container_run,
)
from mostt_model_builder.domain.containers.work_directory import (
    CANDIDATE_ENTRY_NAME,
    CANDIDATE_FILES_DIRECTORY_NAME,
    locate_entry,
    locate_model_card,
    locate_verification_report,
    place_entry,
    place_entry_clean,
)
from mostt_model_builder.domain.contract.recipe_contract import JudgedRecipe


class CleanFiles(StepFiles, Protocol):
    def remove_empty_directory(self, path: PurePosixPath) -> None: ...

    def remove_request_directory(self, path: PurePosixPath) -> None: ...

    def remove_file_if_present(self, path: PurePosixPath) -> bool: ...

    def remove_replacement_if_present(self, path: PurePosixPath) -> bool: ...


class CleanOutcome(Enum):
    CLEANED = auto()
    NOTHING_TO_CLEAN = auto()


class BuildFile(Enum):
    VERIFICATION_REPORT = auto()
    MODEL_CARD = auto()


@dataclass(frozen=True, slots=True)
class EntryCleaning:
    recipe_name: str
    entry_name: str
    key: str
    image: ContainerImage
    work_root: PurePosixPath
    head_tree: str
    host: HostIdentity


# The entry and the files beside its outputs are the host's own, and only the outputs, which a
# container wrote and may have filled with directories of its user's, go through a container.
def clean_entry(request: EntryCleaning, engine: ContainerEngine, files: CleanFiles) -> CleanOutcome:
    entry = locate_entry(request.recipe_name, request.entry_name, request.key)
    if not is_judged_directory(request.work_root, entry, request.host.user_id, files):
        return CleanOutcome.NOTHING_TO_CLEAN
    placement = place_entry(request.recipe_name, request.entry_name, request.key)
    output_directory = placement.output_directory
    has_outputs = is_judged_directory(
        request.work_root, output_directory, request.host.user_id, files
    )
    request_directory = placement.request_directory
    has_request = request_directory is not None and is_judged_directory(
        request.work_root, request_directory, request.host.user_id, files
    )
    if has_outputs:
        empty_output_directory(request, engine, files)
        files.remove_empty_directory(request.work_root / output_directory)
    if has_request and request_directory is not None:
        files.remove_request_directory(request.work_root / request_directory)
    files.remove_file_if_present(request.work_root / placement.log_path)
    remove_record(request.work_root / entry, files)
    files.remove_empty_directory(request.work_root / entry)
    return CleanOutcome.CLEANED


def empty_output_directory(
    request: EntryCleaning, engine: ContainerEngine, files: CleanFiles
) -> None:
    clean_row = build_clean_row(request.image)
    image_id = prepare_container(clean_row, request.head_tree, request.host, engine)
    clean_placement = place_entry_clean(request.recipe_name, request.entry_name, request.key)
    run = plan_container_run(
        clean_row, image_id, clean_placement, {}, request.work_root, request.host.group_id, {}
    )
    run_judged_container(run, clean_placement.log_path, clean_row.name, request.host, engine, files)
    files.remove_file_if_present(request.work_root / clean_placement.log_path)


# The candidate holds the host's own copies, in a directory the host made, so the host removes
# them itself, without a container.
def remove_candidate_entry(
    recipe_name: str, key: str, work_root: PurePosixPath, user_id: int, files: CleanFiles
) -> CleanOutcome:
    entry = locate_entry(recipe_name, CANDIDATE_ENTRY_NAME, key)
    if not is_judged_directory(work_root, entry, user_id, files):
        return CleanOutcome.NOTHING_TO_CLEAN
    candidate_directory = entry / CANDIDATE_FILES_DIRECTORY_NAME
    if is_judged_directory(work_root, candidate_directory, user_id, files):
        files.remove_request_directory(work_root / candidate_directory)
    remove_record(work_root / entry, files)
    files.remove_empty_directory(work_root / entry)
    return CleanOutcome.CLEANED


# A write of the record that was cut short leaves its replacement beside it.
def remove_record(entry: PurePosixPath, files: CleanFiles) -> None:
    files.remove_file_if_present(entry / RECORD_FILE_NAME)
    files.remove_replacement_if_present(entry / RECORD_FILE_NAME)


def is_judged_directory(
    work_root: PurePosixPath, relative_path: PurePosixPath, user_id: int, files: CleanFiles
) -> bool:
    components = files.inspect_components(work_root, relative_path)
    if components[-1].facts.kind is FileKind.MISSING:
        return False
    judge_mount(components, is_file=False, user_id=user_id)
    return True


# The host writes the verification's report itself, from the outputs of every step, so it removes
# it itself, at any cleaning, since a report of outputs that are gone would vouch for what no longer
# exists.
def remove_verification_report(
    recipe_name: str, work_root: PurePosixPath, user_id: int, files: CleanFiles
) -> CleanOutcome:
    return remove_build_file(locate_verification_report(recipe_name), work_root, user_id, files)


# The card is rendered from the verification, so it goes with it.
def remove_model_card(
    recipe_name: str, work_root: PurePosixPath, user_id: int, files: CleanFiles
) -> CleanOutcome:
    return remove_build_file(locate_model_card(recipe_name), work_root, user_id, files)


def remove_build_file(
    path: PurePosixPath, work_root: PurePosixPath, user_id: int, files: CleanFiles
) -> CleanOutcome:
    if not is_judged_directory(work_root, path.parent, user_id, files):
        return CleanOutcome.NOTHING_TO_CLEAN
    if files.remove_file_if_present(work_root / path):
        return CleanOutcome.CLEANED
    return CleanOutcome.NOTHING_TO_CLEAN


@dataclass(frozen=True, slots=True)
class BuildFileCleaned:
    file: BuildFile
    outcome: CleanOutcome


@dataclass(frozen=True, slots=True)
class EntryCleaned:
    name: str
    key: str
    outcome: CleanOutcome


@dataclass(frozen=True, slots=True)
class EntrySkipped:
    name: str
    reason: str


type CleanEvent = BuildFileCleaned | EntryCleaned | EntrySkipped


@dataclass(frozen=True, slots=True)
class BuildCleaning:
    recipe: JudgedRecipe
    units: Sequence[BuildUnit]
    steps: frozenset[str] | None
    work_root: PurePosixPath
    head_tree: str
    host: HostIdentity
    tree_files: Mapping[PurePosixPath, bytes]
    hardware: HardwareFacts | None


class BuildCleaningFiles(CleanFiles, RecordFiles, Protocol):
    pass


# Every key is computed before anything is removed, since a key draws on the records of the entries
# before it.
def clean_build(
    request: BuildCleaning,
    engine: ContainerEngine,
    files: BuildCleaningFiles,
    report: Callable[[CleanEvent], None],
) -> None:
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
    report(
        BuildFileCleaned(
            BuildFile.VERIFICATION_REPORT,
            remove_verification_report(recipe.name, request.work_root, user_id, files),
        )
    )
    report(
        BuildFileCleaned(
            BuildFile.MODEL_CARD,
            remove_model_card(recipe.name, request.work_root, user_id, files),
        )
    )
    for entry in entries:
        if request.steps is not None and entry.unit.step not in request.steps:
            continue
        if entry.key is None:
            report(EntrySkipped(entry.unit.name, entry.skip.describe() if entry.skip else ""))
            continue
        report(
            EntryCleaned(entry.unit.name, entry.key, remove_entry(request, entry, engine, files))
        )


def remove_entry(
    request: BuildCleaning,
    entry: EntryOfRecords,
    engine: ContainerEngine,
    files: BuildCleaningFiles,
) -> CleanOutcome:
    key = entry.key
    if key is None:
        return CleanOutcome.NOTHING_TO_CLEAN
    row = entry.unit.row
    if row is None:
        return remove_candidate_entry(
            request.recipe.name, key, request.work_root, request.host.user_id, files
        )
    return clean_entry(
        EntryCleaning(
            recipe_name=request.recipe.name,
            entry_name=entry.unit.name,
            key=key,
            image=row.image,
            work_root=request.work_root,
            head_tree=request.head_tree,
            host=request.host,
        ),
        engine,
        files,
    )
