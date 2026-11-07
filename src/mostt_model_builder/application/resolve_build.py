from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Protocol

from mostt_model_builder.application.assemble_candidate import (
    INSPECTION_PARSED_NAMES,
    CandidateAssembly,
    CandidateCopy,
    CandidateError,
    CandidateFiles,
    CandidateRequest,
    DigestFiles,
    InspectionOutcome,
    check_candidate_copies,
    copy_candidate,
    inspect_candidate,
    read_candidate_texts,
    read_inspection,
)
from mostt_model_builder.application.cache_entries import (
    CacheEntryError,
    EntryKeys,
    EntryState,
    find_entry_state,
    read_entry_record,
    refuse_entry,
    refuse_missing_entry,
)
from mostt_model_builder.application.check_fleurs import (
    FLEURS_CHECK_PARSED_NAMES,
    FleursCheckOutcome,
    FleursCheckRun,
    check_fleurs_bench,
    read_fleurs_check,
)
from mostt_model_builder.application.clean_run import (
    CleanFiles,
    EntryCleaning,
    clean_entry,
    remove_candidate_entry,
)
from mostt_model_builder.application.convert_variants import (
    CONVERSION_PARSED_NAMES,
    ConversionOutcome,
    ConversionRun,
    convert_variants,
    read_conversion,
)
from mostt_model_builder.application.run_recipe_step import (
    RecipeStepOutcome,
    RecipeStepRun,
    RunShard,
    list_parsed_names,
    read_recipe_step,
    run_recipe_step,
)
from mostt_model_builder.application.run_step import (
    ContainerEngine,
    HostIdentity,
    StepError,
    StepOutputs,
    find_parsed_names,
    read_step_outputs,
)
from mostt_model_builder.domain.bench.hypotheses import Hypothesis
from mostt_model_builder.domain.cache.build_units import BuildUnit, UnitKind
from mostt_model_builder.domain.cache.hardware_facts import HardwareFacts
from mostt_model_builder.domain.cache.output_keys import PlainData
from mostt_model_builder.domain.cache.output_records import (
    RECORD_FILE_MODE,
    RECORD_FILE_NAME,
    OutputRecord,
    OutputRecordError,
    RecordedOutput,
    judge_recorded_outputs,
    render_output_record,
)
from mostt_model_builder.domain.containers.container_files import (
    ContainerFileError,
    judge_written_file,
)
from mostt_model_builder.domain.containers.container_steps import (
    CONTAINER_USER_ID,
    INSPECT_ROW,
    StepPlacement,
    StepRow,
)
from mostt_model_builder.domain.containers.step_peaks import StepPeaksError
from mostt_model_builder.domain.containers.work_directory import (
    CANDIDATE_ENTRY_NAME,
    locate_candidate_files,
    locate_entry,
    place_entry,
)
from mostt_model_builder.domain.contract.recipe_contract import JudgedRecipe, JudgedRun
from mostt_model_builder.domain.onnx.conversion_request import ConversionRequest


class CacheFiles(CandidateFiles, DigestFiles, CleanFiles, Protocol):
    def write_new_file_atomically(self, path: PurePosixPath, content: bytes, mode: int) -> None: ...


@dataclass(frozen=True, slots=True)
class StepBegun:
    step: str


@dataclass(frozen=True, slots=True)
class EntryReused:
    name: str
    key: str


@dataclass(frozen=True, slots=True)
class UnfinishedEntryRemoved:
    name: str
    key: str


@dataclass(frozen=True, slots=True)
class FleursChecked:
    outcome: FleursCheckOutcome


@dataclass(frozen=True, slots=True)
class VariantConverted:
    row: StepRow
    outcome: ConversionOutcome


@dataclass(frozen=True, slots=True)
class CandidateAssembled:
    copies: tuple[CandidateCopy, ...]


@dataclass(frozen=True, slots=True)
class CandidateInspected:
    outcome: InspectionOutcome


@dataclass(frozen=True, slots=True)
class RecipeStepRan:
    row: StepRow
    outcome: RecipeStepOutcome


@dataclass(frozen=True, slots=True)
class RunLanguageTranscribed:
    run: str
    hypotheses: tuple[Hypothesis, ...]


type BuildEvent = (
    StepBegun
    | EntryReused
    | UnfinishedEntryRemoved
    | FleursChecked
    | VariantConverted
    | CandidateAssembled
    | CandidateInspected
    | RecipeStepRan
    | RunLanguageTranscribed
)


@dataclass(frozen=True, slots=True)
class BuildResolutionRequest:
    recipe: JudgedRecipe
    units: Sequence[BuildUnit]
    running_steps: frozenset[str]
    work_root: PurePosixPath
    head_tree: str
    host: HostIdentity
    tree_files: Mapping[PurePosixPath, bytes]
    hardware: HardwareFacts | None


@dataclass(frozen=True, slots=True)
class ResolvedEntry:
    name: str
    key: str
    output_directory: PurePosixPath
    digests: Mapping[str, str]


@dataclass(frozen=True, slots=True)
class ResolvedBuild:
    entries: Mapping[str, ResolvedEntry]

    @property
    def output_directories(self) -> dict[str, PurePosixPath]:
        return {name: entry.output_directory for name, entry in self.entries.items()}


@dataclass(frozen=True, slots=True)
class KeyedRow:
    unit: BuildUnit
    row: StepRow
    key: str
    document: Mapping[str, PlainData]
    placement: StepPlacement

    @property
    def entry(self) -> PurePosixPath:
        return self.placement.output_directory.parent


# An entry reused under its key goes through the judgments, the hashes and the parsers of an
# output just written, since today's checks may be stricter than those it was made under, and its
# digests feed the later keys as its run's would.
def resolve_build(
    request: BuildResolutionRequest,
    engine: ContainerEngine,
    files: CacheFiles,
    report: Callable[[BuildEvent], None],
) -> ResolvedBuild:
    resolver = BuildResolver(request, engine, files, report)
    units = list(request.units)
    position = 0
    step = None
    while position < len(units):
        unit = units[position]
        if unit.step != step:
            step = unit.step
            report(StepBegun(step))
        if unit.kind is UnitKind.RUN:
            batch = take_run_language(units, position)
            resolver.resolve_run_language(batch)
            position += len(batch)
            continue
        if unit.kind is UnitKind.CANDIDATE:
            resolver.resolve_candidate(unit)
        else:
            resolver.resolve_row(unit)
        position += 1
    return ResolvedBuild(dict(resolver.resolved))


def take_run_language(units: Sequence[BuildUnit], start: int) -> list[BuildUnit]:
    first = units[start].derived_row
    batch = []
    for unit in units[start:]:
        derived_row = unit.derived_row
        if (
            unit.kind is not UnitKind.RUN
            or first is None
            or derived_row is None
            or (derived_row.run, derived_row.language) != (first.run, first.language)
        ):
            break
        batch.append(unit)
    return batch


def build_candidate_request(
    recipe: JudgedRecipe,
    candidate_directory: PurePosixPath,
    inspection_directory: PurePosixPath,
    work_root: PurePosixPath,
    head_tree: str,
    host: HostIdentity,
) -> CandidateRequest:
    return CandidateRequest(
        work_root=work_root,
        head_tree=head_tree,
        host=host,
        candidate=recipe.candidate,
        rows=recipe.rows,
        conversion_input=recipe.conversion_input,
        candidate_directory=candidate_directory,
        inspection_directory=inspection_directory,
        conversions=recipe.conversions,
    )


def locate_candidate(
    recipe: JudgedRecipe,
    resolved: ResolvedBuild,
    work_root: PurePosixPath,
    head_tree: str,
    host: HostIdentity,
) -> CandidateRequest:
    candidate = resolved.entries.get(CANDIDATE_ENTRY_NAME)
    inspection = resolved.entries.get(INSPECT_ROW.name)
    if candidate is None or inspection is None:
        raise CandidateError(f"the build of {recipe.name} resolved no inspected candidate")
    return build_candidate_request(
        recipe,
        candidate.output_directory,
        inspection.output_directory,
        work_root,
        head_tree,
        host,
    )


class BuildResolver:
    def __init__(
        self,
        request: BuildResolutionRequest,
        engine: ContainerEngine,
        files: CacheFiles,
        report: Callable[[BuildEvent], None],
    ) -> None:
        self.request = request
        self.recipe = request.recipe
        self.engine = engine
        self.files = files
        self.report = report
        self.keys = EntryKeys(request.recipe, request.tree_files, request.hardware)
        self.resolved: dict[str, ResolvedEntry] = {}
        self.candidate_sizes_bytes: dict[str, int] = {}
        self.candidate_texts: Mapping[str, bytes] | None = None

    @property
    def digests(self) -> dict[str, Mapping[str, str]]:
        return {name: entry.digests for name, entry in self.resolved.items()}

    def key_row(self, unit: BuildUnit) -> KeyedRow:
        row = unit.row
        if row is None:
            raise CacheEntryError(f"{unit.name} plans no row")
        key, document = self.keys.key_unit(unit, self.digests)
        candidate = self.resolved.get(CANDIDATE_ENTRY_NAME)
        placement = place_entry(
            self.recipe.name,
            row.name,
            key,
            candidate.output_directory if candidate is not None else None,
        )
        return KeyedRow(unit, row, key, document, placement)

    def find_state(self, entry: PurePosixPath) -> EntryState:
        return find_entry_state(
            self.request.work_root, entry, self.request.host.user_id, self.files
        )

    def prepare_run(self, keyed: KeyedRow, state: EntryState) -> None:
        if keyed.unit.step not in self.request.running_steps:
            raise refuse_missing_entry(keyed.unit, keyed.key)
        if state is EntryState.UNFINISHED:
            clean_entry(
                EntryCleaning(
                    recipe_name=self.recipe.name,
                    entry_name=keyed.row.name,
                    key=keyed.key,
                    work_root=self.request.work_root,
                    head_tree=self.request.head_tree,
                    host=self.request.host,
                ),
                self.engine,
                self.files,
            )
            self.report(UnfinishedEntryRemoved(keyed.row.name, keyed.key))

    def resolve_row(self, unit: BuildUnit) -> None:
        keyed = self.key_row(unit)
        state = self.find_state(keyed.entry)
        if state is EntryState.COMPLETE:
            outputs = self.reuse_row(keyed, self.list_parsed_names(keyed))
            self.read_reused_row(keyed, outputs)
            self.report(EntryReused(keyed.row.name, keyed.key))
            return
        self.prepare_run(keyed, state)
        self.report(self.run_row(keyed))
        self.record_row(keyed)

    def list_parsed_names(self, keyed: KeyedRow) -> frozenset[str]:
        match keyed.unit.kind:
            case UnitKind.FLEURS_CHECK:
                return FLEURS_CHECK_PARSED_NAMES
            case UnitKind.CONVERSION:
                return CONVERSION_PARSED_NAMES
            case UnitKind.INSPECTION:
                return INSPECTION_PARSED_NAMES
            case _:
                return list_parsed_names(self.build_recipe_step_run(keyed))

    def read_row(self, keyed: KeyedRow, outputs: StepOutputs) -> RecipeStepOutcome | None:
        unit = keyed.unit
        match unit.kind:
            case UnitKind.FLEURS_CHECK:
                read_fleurs_check(outputs)
            case UnitKind.CONVERSION:
                read_conversion(self.require_conversion_request(unit), outputs)
            case UnitKind.INSPECTION:
                read_inspection(
                    self.locate_candidate(keyed.placement.output_directory),
                    outputs,
                    self.candidate_sizes_bytes,
                )
            case _:
                return read_recipe_step(self.build_recipe_step_run(keyed), outputs)
        return None

    # Every parser and judgment of the host raises a ValueError of its own, so a reused output that
    # today's checks refuse names its entry and the cleaning that removes it.
    def read_reused_row(self, keyed: KeyedRow, outputs: StepOutputs) -> RecipeStepOutcome | None:
        try:
            return self.read_row(keyed, outputs)
        except (ValueError, CandidateError, StepError) as error:
            raise refuse_entry(keyed.entry, keyed.unit, str(error)) from error

    def run_row(self, keyed: KeyedRow) -> BuildEvent:
        unit = keyed.unit
        request = self.request
        match unit.kind:
            case UnitKind.FLEURS_CHECK:
                return FleursChecked(
                    check_fleurs_bench(
                        FleursCheckRun(
                            pins=self.recipe.pins,
                            placement=keyed.placement,
                            work_root=request.work_root,
                            head_tree=request.head_tree,
                            host=request.host,
                        ),
                        self.engine,
                        self.files,
                    )
                )
            case UnitKind.CONVERSION:
                return VariantConverted(
                    keyed.row,
                    convert_variants(
                        ConversionRun(
                            row=keyed.row,
                            request=self.require_conversion_request(unit),
                            placement=keyed.placement,
                            earlier_output_directories=self.find_earlier_directories(keyed.row),
                            work_root=request.work_root,
                            head_tree=request.head_tree,
                            host=request.host,
                        ),
                        self.engine,
                        self.files,
                    ),
                )
            case UnitKind.INSPECTION:
                return CandidateInspected(
                    inspect_candidate(
                        self.locate_candidate(keyed.placement.output_directory),
                        keyed.placement,
                        self.candidate_sizes_bytes,
                        self.engine,
                        self.files,
                    )
                )
            case _:
                return RecipeStepRan(
                    keyed.row,
                    run_recipe_step(self.build_recipe_step_run(keyed), self.engine, self.files),
                )

    def resolve_run_language(self, units: Sequence[BuildUnit]) -> None:
        hypotheses_by_row: dict[str, tuple[Hypothesis, ...]] = {}
        missing = []
        for unit in units:
            keyed = self.key_row(unit)
            state = self.find_state(keyed.entry)
            if state is EntryState.COMPLETE:
                outputs = self.reuse_row(keyed, self.list_parsed_names(keyed))
                outcome = self.read_reused_row(keyed, outputs)
                hypotheses_by_row[keyed.row.name] = outcome.hypotheses if outcome else ()
                self.report(EntryReused(keyed.row.name, keyed.key))
            else:
                self.prepare_run(keyed, state)
                missing.append(keyed)
        if not missing:
            return
        for keyed, ran in zip(missing, self.run_at_once(missing), strict=True):
            hypotheses_by_row[keyed.row.name] = ran.hypotheses
        derived_row = units[0].derived_row
        if derived_row is not None:
            self.report(
                RunLanguageTranscribed(
                    derived_row.run,
                    tuple(
                        hypothesis for unit in units for hypothesis in hypotheses_by_row[unit.name]
                    ),
                )
            )

    # The decoding times are read against the trial build's, which ran the shards of a run at the
    # same time, each on CPUs of its own.
    def run_at_once(self, missing: Sequence[KeyedRow]) -> list[RecipeStepOutcome]:
        with ThreadPoolExecutor(max_workers=len(missing)) as pool:
            futures: list[Future[RecipeStepOutcome]] = [
                pool.submit(
                    run_recipe_step, self.build_recipe_step_run(keyed), self.engine, self.files
                )
                for keyed in missing
            ]
        outcomes = []
        first_error: BaseException | None = None
        for keyed, future in zip(missing, futures, strict=True):
            error = future.exception()
            if error is not None:
                first_error = first_error or error
                continue
            outcome = future.result()
            self.report(RecipeStepRan(keyed.row, outcome))
            self.record_row(keyed)
            outcomes.append(outcome)
        if first_error is not None:
            raise first_error
        return outcomes

    def reuse_row(self, keyed: KeyedRow, parsed_names: frozenset[str]) -> StepOutputs:
        request = self.request
        record = read_entry_record(
            request.work_root,
            keyed.entry,
            keyed.unit,
            keyed.key,
            request.host.user_id,
            self.files,
        )
        try:
            outputs = read_step_outputs(
                keyed.row,
                keyed.placement.output_directory,
                request.work_root,
                request.host.user_id,
                find_parsed_names(keyed.row, parsed_names),
                self.files,
            )
            judge_recorded_outputs(record, self.digest_outputs(keyed))
        except (ContainerFileError, OutputRecordError, StepError, StepPeaksError) as error:
            raise refuse_entry(keyed.entry, keyed.unit, str(error)) from error
        self.resolve(keyed.row.name, keyed.key, keyed.placement.output_directory, record)
        return outputs

    def digest_outputs(self, keyed: KeyedRow) -> tuple[RecordedOutput, ...]:
        directory = self.request.work_root / keyed.placement.output_directory
        recorded = []
        for output in keyed.row.outputs:
            hashed = self.files.digest_host_file(directory, output.name, output.max_size_bytes)
            judge_written_file(output.name, hashed.facts, output.max_size_bytes, CONTAINER_USER_ID)
            if hashed.sha256 is None:
                raise StepError(f"the output {output.name} was not hashed")
            recorded.append(RecordedOutput(output.name, hashed.facts.size_bytes, hashed.sha256))
        return tuple(recorded)

    def record_row(self, keyed: KeyedRow) -> None:
        record = OutputRecord(
            keyed.key, keyed.row.name, dict(keyed.document), self.digest_outputs(keyed)
        )
        self.write_record(keyed.entry, record)
        self.resolve(keyed.row.name, keyed.key, keyed.placement.output_directory, record)

    def write_record(self, entry: PurePosixPath, record: OutputRecord) -> None:
        self.files.write_new_file_atomically(
            self.request.work_root / entry / RECORD_FILE_NAME,
            render_output_record(record),
            RECORD_FILE_MODE,
        )

    def resolve(
        self, name: str, key: str, output_directory: PurePosixPath, record: OutputRecord
    ) -> None:
        self.resolved[name] = ResolvedEntry(name, key, output_directory, record.digests)

    def resolve_candidate(self, unit: BuildUnit) -> None:
        request = self.request
        key, document = self.keys.key_unit(unit, self.digests)
        entry = locate_entry(self.recipe.name, CANDIDATE_ENTRY_NAME, key)
        assembly = CandidateAssembly(
            work_root=request.work_root,
            host=request.host,
            candidate=self.recipe.candidate,
            rows=self.recipe.rows,
            candidate_directory=locate_candidate_files(self.recipe.name, key),
            origin_directories={
                name: found.output_directory for name, found in self.resolved.items()
            },
        )
        state = self.find_state(entry)
        if state is EntryState.COMPLETE:
            record = read_entry_record(
                request.work_root, entry, unit, key, request.host.user_id, self.files
            )
            try:
                copies = check_candidate_copies(assembly, self.files)
                judge_recorded_outputs(record, describe_copies(copies))
            except (ContainerFileError, OutputRecordError, CandidateError) as error:
                raise refuse_entry(entry, unit, str(error)) from error
            self.report(EntryReused(CANDIDATE_ENTRY_NAME, key))
        else:
            if unit.step not in request.running_steps:
                raise refuse_missing_entry(unit, key)
            if state is EntryState.UNFINISHED:
                remove_candidate_entry(
                    self.recipe.name, key, request.work_root, request.host.user_id, self.files
                )
                self.report(UnfinishedEntryRemoved(CANDIDATE_ENTRY_NAME, key))
            copies = copy_candidate(assembly, self.files)
            record = OutputRecord(key, CANDIDATE_ENTRY_NAME, document, describe_copies(copies))
            self.write_record(entry, record)
            self.report(CandidateAssembled(copies))
        self.candidate_sizes_bytes = {copy.name: copy.size_bytes for copy in copies}
        self.resolve(CANDIDATE_ENTRY_NAME, key, assembly.candidate_directory, record)

    def locate_candidate(
        self, inspection_directory: PurePosixPath | None = None
    ) -> CandidateRequest:
        request = self.request
        if inspection_directory is None:
            return locate_candidate(
                self.recipe,
                ResolvedBuild(self.resolved),
                request.work_root,
                request.head_tree,
                request.host,
            )
        candidate = self.resolved.get(CANDIDATE_ENTRY_NAME)
        if candidate is None:
            raise CandidateError(
                "the inspection reads the candidate, which the build has not resolved"
            )
        return build_candidate_request(
            self.recipe,
            candidate.output_directory,
            inspection_directory,
            request.work_root,
            request.head_tree,
            request.host,
        )

    def find_earlier_directories(self, row: StepRow) -> dict[str, PurePosixPath]:
        return {
            earlier_output.step: self.resolved[earlier_output.step].output_directory
            for earlier_output in row.earlier_outputs
        }

    def require_conversion_request(self, unit: BuildUnit) -> ConversionRequest:
        if unit.conversion_row is None:
            raise CacheEntryError(f"{unit.name} plans no conversion")
        return unit.conversion_row.request

    def find_candidate_texts(self) -> Mapping[str, bytes]:
        if self.candidate_texts is None:
            self.candidate_texts = read_candidate_texts(self.locate_candidate(), self.files)
        return self.candidate_texts

    def find_run(self, name: str) -> JudgedRun:
        for run in self.recipe.runs:
            if run.name == name:
                return run
        raise CacheEntryError(f"the recipe {self.recipe.name} has no run {name}")

    def build_recipe_step_run(self, keyed: KeyedRow) -> RecipeStepRun:
        request = self.request
        row = keyed.row
        derived_row = keyed.unit.derived_row
        shard = None
        if derived_row is not None:
            shard = RunShard(
                derived_row.request,
                self.find_run(derived_row.run).reader,
                self.find_candidate_texts(),
            )
        reads_candidate = row.binds_candidate or bool(row.candidate_files)
        return RecipeStepRun(
            row=row,
            pins=self.recipe.pins,
            placement=keyed.placement,
            earlier_output_directories=self.find_earlier_directories(row),
            work_root=request.work_root,
            head_tree=request.head_tree,
            host=request.host,
            parities=self.recipe.parities,
            shard=shard,
            candidate=self.locate_candidate() if reads_candidate else None,
        )


def describe_copies(copies: Sequence[CandidateCopy]) -> tuple[RecordedOutput, ...]:
    return tuple(RecordedOutput(copy.name, copy.size_bytes, copy.sha256) for copy in copies)
