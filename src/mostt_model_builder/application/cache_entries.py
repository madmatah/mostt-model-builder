from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import PurePosixPath
from typing import Protocol

from mostt_model_builder.application.run_step import WrittenFile
from mostt_model_builder.domain.cache.build_units import (
    BuildUnit,
    UnitKind,
    list_key_dependencies,
)
from mostt_model_builder.domain.cache.hardware_facts import NO_GPU_REASON, HardwareFacts
from mostt_model_builder.domain.cache.import_closure import BOOTSTRAP_PATH, collect_code_files
from mostt_model_builder.domain.cache.output_keys import (
    PlainData,
    RowKeyInputs,
    compute_key,
    document_candidate_key,
    document_row_key,
    lacks_gpu_for_key,
    name_earlier_output,
    name_image_extra,
    plan_keyed_run,
    select_image_files,
)
from mostt_model_builder.domain.cache.output_records import (
    RECORD_FILE_NAME,
    RECORD_MAX_SIZE_BYTES,
    OutputRecord,
    OutputRecordError,
    judge_record_file,
    judge_record_identity,
    parse_output_record,
    show_key,
)
from mostt_model_builder.domain.containers.container_files import (
    FileKind,
    PathComponent,
    judge_mount,
)
from mostt_model_builder.domain.containers.container_steps import ContainerImage, StepRow
from mostt_model_builder.domain.containers.work_directory import (
    CANDIDATE_ENTRY_NAME,
    locate_entry,
)
from mostt_model_builder.domain.contract.recipe_contract import JudgedRecipe
from mostt_model_builder.domain.lock.extra_closure import LockedEntry, close_extra
from mostt_model_builder.domain.lock.lock_check import LOCK_FILE_NAME

BUILD_COMMAND = "./model-build.sh --work <dir> --recipe <name> build"
CLEAN_COMMAND = "./model-build.sh --work <dir> --recipe <name> clean"
IMAGES_LOCK_PATH = PurePosixPath("images", LOCK_FILE_NAME)
TREE_ENCODING = "utf-8"


class CacheEntryError(Exception):
    pass


class RecordFiles(Protocol):
    def inspect_components(
        self, root: PurePosixPath, relative_path: PurePosixPath
    ) -> tuple[PathComponent, ...]: ...

    def read_host_file(
        self, directory: PurePosixPath, name: str, max_size_bytes: int
    ) -> WrittenFile: ...


class EntryState(Enum):
    MISSING = auto()
    UNFINISHED = auto()
    COMPLETE = auto()


class SkipCause(Enum):
    MISSING_RECORD = auto()
    REFUSED_RECORD = auto()
    NO_GPU = auto()


@dataclass(frozen=True, slots=True)
class KeySkip:
    cause: SkipCause
    dependency: str = ""

    def describe(self) -> str:
        match self.cause:
            case SkipCause.MISSING_RECORD:
                return f"the record of {self.dependency}, which its key needs, is missing"
            case SkipCause.REFUSED_RECORD:
                return f"the record of {self.dependency}, which its key needs, is refused"
            case SkipCause.NO_GPU:
                return NO_GPU_REASON


def flatten_digests(digests: Mapping[str, Mapping[str, str]]) -> dict[str, str]:
    return {
        name_earlier_output(entry_name, output): sha256
        for entry_name, outputs in digests.items()
        for output, sha256 in outputs.items()
    }


def describe_entry(name: str, key: str) -> str:
    return f"{name}, key {show_key(key)}"


def find_entry_state(
    work_root: PurePosixPath, entry: PurePosixPath, user_id: int, files: RecordFiles
) -> EntryState:
    components = files.inspect_components(work_root, entry)
    if components[-1].facts.kind is FileKind.MISSING:
        return EntryState.MISSING
    judge_mount(components, is_file=False, user_id=user_id)
    if files.inspect_components(work_root, entry / RECORD_FILE_NAME)[-1].facts.kind is (
        FileKind.MISSING
    ):
        return EntryState.UNFINISHED
    return EntryState.COMPLETE


# A record is the host's own file, read strictly, and a malformed one points to a change that the
# user must look at, so the refusal names the entry and the cleaning that removes it.
def read_entry_record(
    work_root: PurePosixPath,
    entry: PurePosixPath,
    unit: BuildUnit,
    key: str,
    user_id: int,
    files: RecordFiles,
) -> OutputRecord:
    written = files.read_host_file(work_root / entry, RECORD_FILE_NAME, RECORD_MAX_SIZE_BYTES)
    try:
        judge_record_file(written.facts, user_id)
        if written.content is None:
            raise OutputRecordError("the record was not read")
        record = parse_output_record(written.content)
        judge_record_identity(record, key, unit.name)
    except OutputRecordError as error:
        raise refuse_entry(entry, unit, str(error)) from error
    return record


def refuse_entry(entry: PurePosixPath, unit: BuildUnit, reason: str) -> CacheEntryError:
    return CacheEntryError(
        f"the entry {entry} of {unit.name} is refused: {reason}; look at it, then remove it "
        f"with {CLEAN_COMMAND} {unit.step}"
    )


def refuse_missing_entry(unit: BuildUnit, key: str) -> CacheEntryError:
    return CacheEntryError(
        f"{unit.name} has no entry at its key {show_key(key)}: make it with {BUILD_COMMAND}"
    )


class EntryKeys:
    def __init__(
        self,
        recipe: JudgedRecipe,
        tree_files: Mapping[PurePosixPath, bytes],
        hardware: HardwareFacts | None,
    ) -> None:
        self.recipe = recipe
        self.tree_files = tree_files
        self.hardware = hardware
        self.code_files_by_module: dict[str, dict[PurePosixPath, bytes]] = {}
        self.packages_by_image: dict[str, tuple[LockedEntry, ...]] = {}

    def find_code_files(self, entry_module: str) -> dict[PurePosixPath, bytes]:
        if entry_module not in self.code_files_by_module:
            self.code_files_by_module[entry_module] = collect_code_files(
                entry_module, self.tree_files
            )
        return self.code_files_by_module[entry_module]

    def find_image_packages(self, image: ContainerImage) -> tuple[LockedEntry, ...]:
        if image.tag not in self.packages_by_image:
            extra = name_image_extra(image)
            lock_text = (
                self.require_tree_file(IMAGES_LOCK_PATH).decode(TREE_ENCODING)
                if extra is not None
                else ""
            )
            self.packages_by_image[image.tag] = close_extra(lock_text, extra)
        return self.packages_by_image[image.tag]

    def require_tree_file(self, path: PurePosixPath) -> bytes:
        if path not in self.tree_files:
            raise CacheEntryError(f"HEAD's tree holds no {path}, which every key reads")
        return self.tree_files[path]

    def find_key_skip(
        self,
        unit: BuildUnit,
        recorded: Mapping[str, Mapping[str, str]],
        refused: Collection[str],
    ) -> KeySkip | None:
        missing = [
            dependency for dependency in list_key_dependencies(unit) if dependency not in recorded
        ]
        if missing:
            cause = SkipCause.REFUSED_RECORD if missing[0] in refused else SkipCause.MISSING_RECORD
            return KeySkip(cause, missing[0])
        if unit.row is not None and lacks_gpu_for_key(unit.row, self.hardware):
            return KeySkip(SkipCause.NO_GPU)
        return None

    def key_unit(
        self, unit: BuildUnit, digests: Mapping[str, Mapping[str, str]]
    ) -> tuple[str, dict[str, PlainData]]:
        if unit.kind is UnitKind.CANDIDATE:
            document = document_candidate_key(self.recipe.candidate, flatten_digests(digests))
            return compute_key(document), document
        row = require_row(unit)
        document = document_row_key(
            RowKeyInputs(
                row=row,
                planned_run=plan_keyed_run(row, self.recipe.pins),
                request_files=unit.request_files,
                image_files=select_image_files(row.image, self.tree_files),
                image_packages=self.find_image_packages(row.image),
                bootstrap=self.require_tree_file(BOOTSTRAP_PATH),
                code_files=self.find_code_files(row.entry_module),
                pins=self.recipe.pins,
                earlier_output_digests=flatten_digests(digests),
                candidate_digests=digests.get(CANDIDATE_ENTRY_NAME, {}),
                hardware=self.hardware if unit.kind is UnitKind.RUN or row.uses_gpu else None,
            )
        )
        return compute_key(document), document


def require_row(unit: BuildUnit) -> StepRow:
    if unit.row is None:
        raise CacheEntryError(f"{unit.name} plans no row")
    return unit.row


@dataclass(frozen=True, slots=True)
class EntryOfRecords:
    unit: BuildUnit
    key: str | None
    skip: KeySkip | None
    state: EntryState
    refusal: str | None = None


# Cleaning and pruning key each entry from the records of the entries before it, never from their
# outputs, so that a corrupted entry, and every entry after it, stays removable without being judged
# again. A refused record leaves its entry removable under its key, and the entries whose keys need
# its digests skipped.
def key_entries_from_records(
    recipe: JudgedRecipe,
    units: Sequence[BuildUnit],
    keys: EntryKeys,
    work_root: PurePosixPath,
    user_id: int,
    files: RecordFiles,
) -> tuple[EntryOfRecords, ...]:
    recorded: dict[str, Mapping[str, str]] = {}
    refused: set[str] = set()
    entries = []
    for unit in units:
        skip = keys.find_key_skip(unit, recorded, refused)
        if skip is not None:
            entries.append(EntryOfRecords(unit, None, skip, EntryState.MISSING))
            continue
        key, _ = keys.key_unit(unit, recorded)
        entry = locate_entry(recipe.name, unit.name, key)
        state = find_entry_state(work_root, entry, user_id, files)
        refusal = None
        if state is EntryState.COMPLETE:
            try:
                recorded[unit.name] = read_entry_record(
                    work_root, entry, unit, key, user_id, files
                ).digests
            except CacheEntryError as error:
                refused.add(unit.name)
                refusal = str(error)
        entries.append(EntryOfRecords(unit, key, None, state, refusal))
    return tuple(entries)
