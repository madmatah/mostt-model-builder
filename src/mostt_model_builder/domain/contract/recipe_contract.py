import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import PurePosixPath

from mostt_model_builder.domain.acceptance.acceptance import (
    AcceptanceError,
    Comparison,
    judge_comparison_rule,
)
from mostt_model_builder.domain.acceptance.parity_reports import PARITY_NAME_GRAMMAR
from mostt_model_builder.domain.bench.fleurs_index import BENCH_LANGUAGES
from mostt_model_builder.domain.bench.hypotheses import RUN_LINES_FILE_NAME, RunReader
from mostt_model_builder.domain.containers.base_images import BASE_PYTHON, ENVIRONMENT_PYTHON
from mostt_model_builder.domain.containers.container_steps import (
    CLEAN_STEP_NAME,
    CONVERT_STEP_NAME,
    CORE_IMAGES,
    CORE_OWNER,
    CORE_PACKAGE,
    ENTRY_MODULE_GRAMMAR,
    FLEURS_CHECK_STEP_NAME,
    INSPECT_STEP_NAME,
    LOCK_STEP_NAME,
    PEAKS_FILE_NAME,
    PUBLISH_STEP_NAME,
    RUNNABLE_CORE_ROWS,
    STEP_NAME_GRAMMAR,
    ContainerImage,
    DeclaredOutput,
    Device,
    EarlierOutput,
    SourceReference,
    StepLimits,
    StepRow,
    StepTableError,
    build_convert_row,
    find_package_of_owner,
    judge_row,
)
from mostt_model_builder.domain.containers.work_directory import (
    CANDIDATE_DIRECTORY_NAME,
    LOGS_DIRECTORY_NAME,
    MODEL_CARD_FILE_NAME,
    REQUESTS_DIRECTORY_NAME,
)
from mostt_model_builder.domain.contract.derived_rows import (
    MAX_SHARD_COUNT,
    MIN_SHARD_COUNT,
    RUN_NAME_GRAMMAR,
    DerivedRow,
    DeviceRow,
    ShardedRun,
    StepTemplate,
    derive_rows,
)
from mostt_model_builder.domain.contract.model_description import ModelDescription
from mostt_model_builder.domain.formats.file_formats import FileFormat
from mostt_model_builder.domain.formats.file_names import (
    REFUSED_SUFFIXES,
    find_refused_suffixes,
    is_file_name,
)
from mostt_model_builder.domain.onnx.conversion_request import (
    Conversion,
    ConversionRequest,
    ConversionRequestError,
    EightBitBlocksConversion,
    judge_conversions,
)
from mostt_model_builder.domain.onnx.graph_rules import EightBitBlocksOutput, GraphSource
from mostt_model_builder.domain.onnx.node_placement import PLACEMENT_FILE_NAME
from mostt_model_builder.domain.sources.source_pins import (
    PATH_SEPARATOR,
    SourcePin,
    SourcePinError,
    refuse_overlapping_pins,
)

# The bootstrap appends a recipe's sources only under this grammar, and the recipe's directory,
# its images' tags and its runs' directories are named after it.
RECIPE_NAME_GRAMMAR = re.compile(r"[a-z0-9][a-z0-9-]{0,31}")
VARIANT_GRAMMAR = re.compile(r"[a-z0-9][a-z0-9-]{0,31}")
SOURCE_DIRECTORY_NAME = "src"
RECIPES_DIRECTORY_NAME = "recipes"
MODULE_SEPARATOR = "."
PYTHON_SOURCE_SUFFIX = ".py"
CORE_STEP_NAMES = frozenset(
    {
        FLEURS_CHECK_STEP_NAME,
        CLEAN_STEP_NAME,
        LOCK_STEP_NAME,
        INSPECT_STEP_NAME,
        CONVERT_STEP_NAME,
        PUBLISH_STEP_NAME,
    }
)
# A step's outputs live beside the run's requests, logs and candidate, under the same commit.
RESERVED_STEP_NAMES = frozenset(
    {*CORE_STEP_NAMES, REQUESTS_DIRECTORY_NAME, LOGS_DIRECTORY_NAME, CANDIDATE_DIRECTORY_NAME}
)
BUILT_CORE_STEP_NAMES = (FLEURS_CHECK_STEP_NAME, INSPECT_STEP_NAME)
REFERENCE_VARIANT = "reference"
CORE_IMAGES_BY_NAME = {image.name: image for image in CORE_IMAGES}
MAX_HOST_CONTENT_SIZE_BYTES = 2**20


class RecipeContractError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class RecipeImage:
    name: str
    device: str
    extra: str | None
    reads_code_running_sources: bool = False


@dataclass(frozen=True, slots=True)
class SourceMount:
    repository: str
    path: str


@dataclass(frozen=True, slots=True)
class EarlierOutputMount:
    step: str
    output: str


@dataclass(frozen=True, slots=True)
class CandidateMount:
    pass


CANDIDATE_MOUNT = CandidateMount()
type Mount = SourceMount | EarlierOutputMount | CandidateMount


@dataclass(frozen=True, slots=True)
class DeviceTemplate:
    device: str
    image: str
    limits: StepLimits
    outputs: tuple[DeclaredOutput, ...]


@dataclass(frozen=True, slots=True)
class RecipeStep:
    name: str
    image: str | None
    entry_module: str
    mounts: tuple[Mount, ...]
    outputs: tuple[DeclaredOutput, ...]
    limits: StepLimits | None
    uses_gpu: bool = False
    devices: tuple[DeviceTemplate, ...] = ()


@dataclass(frozen=True, slots=True)
class StepOutputOrigin:
    step: str
    output: str


@dataclass(frozen=True, slots=True)
class HostContentOrigin:
    content: bytes


@dataclass(frozen=True, slots=True)
class CandidateFile:
    name: str
    file_format: str
    variants: tuple[str, ...]
    origin: StepOutputOrigin | HostContentOrigin
    external_data_of: str | None = None


@dataclass(frozen=True, slots=True)
class Parity:
    name: str
    step: str
    output: str
    absolute_difference_tolerance: float


@dataclass(frozen=True, slots=True)
class Run:
    name: str
    step: str
    device: str
    variant: str
    languages: tuple[str, ...]
    shard_count: int
    reader: RunReader | None


@dataclass(frozen=True, slots=True)
class VariantConversion:
    variant: str
    conversion: Conversion
    max_size_bytes: int


@dataclass(frozen=True, slots=True)
class RecipeContract:
    name: str
    images: tuple[RecipeImage, ...]
    steps: tuple[RecipeStep, ...]
    candidate: tuple[CandidateFile, ...]
    parities: tuple[Parity, ...] = ()
    runs: tuple[Run, ...] = ()
    conversion_input: StepOutputOrigin | None = None
    build_order: tuple[str, ...] = ()
    conversions: tuple[VariantConversion, ...] = ()
    comparisons: tuple[Comparison, ...] = ()
    model: ModelDescription | None = None


@dataclass(frozen=True, slots=True)
class RecipeFacts:
    core_pins: tuple[SourcePin, ...]
    recipe_pins: tuple[SourcePin, ...]
    present_entry_modules: frozenset[str]
    host_cpu_count: int


@dataclass(frozen=True, slots=True)
class JudgedCandidateFile:
    name: str
    file_format: FileFormat
    variants: tuple[str, ...]
    origin: StepOutputOrigin | HostContentOrigin
    external_data_of: str | None = None


@dataclass(frozen=True, slots=True)
class JudgedRun:
    name: str
    step: str
    device: Device
    variant: str
    languages: tuple[str, ...]
    shard_count: int
    reader: RunReader


@dataclass(frozen=True, slots=True)
class JudgedRecipe:
    name: str
    pins: tuple[SourcePin, ...]
    images: tuple[ContainerImage, ...]
    rows: tuple[StepRow, ...]
    candidate: tuple[JudgedCandidateFile, ...]
    parities: tuple[Parity, ...]
    runs: tuple[JudgedRun, ...] = ()
    derived_rows: tuple[DerivedRow, ...] = ()
    conversion_input: StepOutputOrigin | None = None
    build_order: tuple[str, ...] = ()
    conversions: tuple[VariantConversion, ...] = ()
    comparisons: tuple[Comparison, ...] = ()
    model: ModelDescription | None = None


@dataclass(frozen=True, slots=True)
class JudgedImages:
    by_name: Mapping[str, ContainerImage]
    code_reader: str | None


def locate_entry_module_file(entry_module: str, recipe_name: str) -> PurePosixPath | None:
    package, *module_path = entry_module.split(MODULE_SEPARATOR)
    if not module_path:
        return None
    if package == CORE_PACKAGE:
        source_directory = PurePosixPath(SOURCE_DIRECTORY_NAME)
    elif package == find_package_of_owner(recipe_name):
        source_directory = PurePosixPath(RECIPES_DIRECTORY_NAME, recipe_name, SOURCE_DIRECTORY_NAME)
    else:
        return None
    *parent_packages, module_name = (package, *module_path)
    return source_directory.joinpath(*parent_packages, module_name + PYTHON_SOURCE_SUFFIX)


def list_runnable_rows(recipe: JudgedRecipe) -> tuple[StepRow, ...]:
    return (*RUNNABLE_CORE_ROWS, *recipe.rows)


def find_runnable_row(recipe: JudgedRecipe, step_name: str) -> StepRow:
    for row in list_runnable_rows(recipe):
        if row.name == step_name:
            return row
    raise RecipeContractError(
        f"neither the core nor the recipe {recipe.name} has a step named {step_name!a}"
    )


def list_rows_of_step(recipe: JudgedRecipe, step_name: str) -> tuple[StepRow, ...]:
    derived = tuple(
        derived_row.row for derived_row in recipe.derived_rows if derived_row.step == step_name
    )
    if derived:
        return derived
    return (find_runnable_row(recipe, step_name),)


def find_candidate_file_max_size_bytes(
    origin: StepOutputOrigin | HostContentOrigin, rows: Sequence[StepRow]
) -> int:
    if isinstance(origin, HostContentOrigin):
        return len(origin.content)
    for row in rows:
        if row.name != origin.step:
            continue
        for output in row.outputs:
            if output.name == origin.output:
                return output.max_size_bytes
    raise RecipeContractError(
        f"the candidate copies {origin.output} of {origin.step}, which no step declares"
    )


def build_conversion_request(recipe: JudgedRecipe) -> ConversionRequest:
    if recipe.conversion_input is None or not recipe.conversions:
        raise RecipeContractError(f"the recipe {recipe.name} converts none of its variants")
    return ConversionRequest(
        input_step=recipe.conversion_input.step,
        input_name=recipe.conversion_input.output,
        conversions=tuple(variant.conversion for variant in recipe.conversions),
    )


def find_graph_source(
    origin: StepOutputOrigin | HostContentOrigin,
    conversion_input: StepOutputOrigin | None,
    conversions: Sequence[VariantConversion],
) -> GraphSource | EightBitBlocksOutput:
    if origin == conversion_input:
        return GraphSource.CONVERSION_INPUT
    for variant_conversion in conversions:
        conversion = variant_conversion.conversion
        if isinstance(conversion, EightBitBlocksConversion) and origin == StepOutputOrigin(
            CONVERT_STEP_NAME, conversion.output
        ):
            return EightBitBlocksOutput(conversion.quantized_product_count)
    return GraphSource.OTHER


def list_entry_modules(contract: RecipeContract) -> tuple[str, ...]:
    return tuple(step.entry_module for step in contract.steps)


def judge_recipe(contract: RecipeContract, facts: RecipeFacts) -> JudgedRecipe:
    if not RECIPE_NAME_GRAMMAR.fullmatch(contract.name):
        raise RecipeContractError(
            f"the recipe {contract.name!a} is not named under {RECIPE_NAME_GRAMMAR.pattern}"
        )
    where = f"the recipe {contract.name}"
    pins = combine_pins(facts.core_pins, facts.recipe_pins, where)
    images = judge_images(contract, where)
    runs = judge_runs(contract, where)
    rows, derived_rows = judge_steps(contract, images, pins, facts, runs, where)
    convert_row = judge_conversions_of_recipe(contract, rows, facts.host_cpu_count, where)
    if convert_row is not None:
        rows = (*rows, convert_row)
    candidate = judge_candidate(contract.candidate, rows, where)
    judge_converted_files(contract.conversions, candidate, where)
    return JudgedRecipe(
        name=contract.name,
        pins=pins,
        images=tuple(images.by_name[image.name] for image in contract.images),
        rows=rows,
        candidate=candidate,
        parities=judge_parities(contract.parities, rows, where),
        runs=runs,
        derived_rows=derived_rows,
        conversion_input=judge_conversion_input(contract.conversion_input, candidate, where),
        build_order=judge_build_order(
            contract.build_order, contract.steps, bool(contract.conversions), where
        ),
        conversions=contract.conversions,
        comparisons=judge_comparisons(contract.comparisons, runs, where),
        model=contract.model,
    )


def combine_pins(
    core_pins: Sequence[SourcePin], recipe_pins: Sequence[SourcePin], where: str
) -> tuple[SourcePin, ...]:
    pins = (*core_pins, *recipe_pins)
    try:
        refuse_overlapping_pins(pins)
    except SourcePinError as error:
        raise RecipeContractError(f"{where}: its sources and the core's: {error}") from error
    return pins


def judge_images(contract: RecipeContract, where: str) -> JudgedImages:
    by_name: dict[str, ContainerImage] = {}
    code_readers: list[str] = []
    for image in contract.images:
        image_where = f"{where}: its image {image.name!a}"
        if not STEP_NAME_GRAMMAR.fullmatch(image.name):
            raise RecipeContractError(
                f"{image_where} is not named under {STEP_NAME_GRAMMAR.pattern}"
            )
        if image.name in CORE_IMAGES_BY_NAME:
            raise RecipeContractError(f"{image_where} bears the name of an image of the core")
        if image.name in by_name:
            raise RecipeContractError(f"{image_where} is declared twice")
        if image.device not in set(Device):
            devices = ", ".join(device.value for device in Device)
            raise RecipeContractError(f"{image_where} has no device among {devices}")
        expected_extra = f"{contract.name}-{image.name}"
        if image.extra is not None and image.extra != expected_extra:
            raise RecipeContractError(
                f"{image_where} installs the extra {image.extra!a}, not {expected_extra}, "
                "the name of its extra in the images' environment"
            )
        if image.reads_code_running_sources:
            code_readers.append(image.name)
        by_name[image.name] = ContainerImage(
            owner=contract.name,
            name=image.name,
            device=Device(image.device),
            python=BASE_PYTHON if image.extra is None else ENVIRONMENT_PYTHON,
        )
    if len(code_readers) > 1:
        raise RecipeContractError(
            f"{where}: the images {', '.join(code_readers)} read the sources that run code when "
            "they load, which one image at most may"
        )
    return JudgedImages(by_name, code_readers[0] if code_readers else None)


def judge_runs(contract: RecipeContract, where: str) -> tuple[JudgedRun, ...]:
    candidate_variants = {
        variant for candidate_file in contract.candidate for variant in candidate_file.variants
    }
    step_names = {step.name for step in contract.steps}
    judged: dict[str, JudgedRun] = {}
    for run in contract.runs:
        run_where = f"{where}: its run {run.name!a}"
        if not RUN_NAME_GRAMMAR.fullmatch(run.name):
            raise RecipeContractError(f"{run_where} is not named under {RUN_NAME_GRAMMAR.pattern}")
        if run.name in judged:
            raise RecipeContractError(f"{run_where} is declared twice")
        if run.step not in step_names:
            raise RecipeContractError(
                f"{run_where} comes from {run.step!a}, which is not a step of the recipe"
            )
        if run.device not in set(Device):
            raise RecipeContractError(f"{run_where} runs on {run.device!a}, neither cpu nor gpu")
        if run.variant != REFERENCE_VARIANT and run.variant not in candidate_variants:
            raise RecipeContractError(
                f"{run_where} reads the variant {run.variant!a}, which no file of the candidate "
                "serves"
            )
        judge_languages(run.languages, run_where)
        if (
            isinstance(run.shard_count, bool)
            or not isinstance(run.shard_count, int)
            or not MIN_SHARD_COUNT <= run.shard_count <= MAX_SHARD_COUNT
        ):
            raise RecipeContractError(
                f"{run_where} has {run.shard_count!a} shards, outside {MIN_SHARD_COUNT} to "
                f"{MAX_SHARD_COUNT}"
            )
        if run.reader is None or not callable(run.reader):
            raise RecipeContractError(f"{run_where} has no reader of its lines")
        judged[run.name] = JudgedRun(
            name=run.name,
            step=run.step,
            device=Device(run.device),
            variant=run.variant,
            languages=run.languages,
            shard_count=run.shard_count,
            reader=run.reader,
        )
    return tuple(judged.values())


def judge_languages(languages: Sequence[str], where: str) -> None:
    if not languages:
        raise RecipeContractError(f"{where} reads no language")
    for language in languages:
        if language not in BENCH_LANGUAGES:
            raise RecipeContractError(
                f"{where} reads {language!a}, which is not a language of the bench"
            )
    if len(set(languages)) != len(languages):
        raise RecipeContractError(f"{where} names a language twice")


def judge_steps(
    contract: RecipeContract,
    images: JudgedImages,
    pins: Sequence[SourcePin],
    facts: RecipeFacts,
    runs: Sequence[JudgedRun],
    where: str,
) -> tuple[tuple[StepRow, ...], tuple[DerivedRow, ...]]:
    rows: dict[str, StepRow] = {}
    derived_rows: list[DerivedRow] = []
    step_names: set[str] = set()
    for step in contract.steps:
        step_where = f"{where}: its step {step.name!a}"
        if not STEP_NAME_GRAMMAR.fullmatch(step.name):
            raise RecipeContractError(
                f"{step_where} is not named under {STEP_NAME_GRAMMAR.pattern}"
            )
        if step.name in RESERVED_STEP_NAMES:
            raise RecipeContractError(
                f"{step_where} bears the name of a step of the core or of a directory of its run"
            )
        if step.name in step_names:
            raise RecipeContractError(f"{step_where} is declared twice")
        step_names.add(step.name)
        judge_entry_module(step, contract.name, facts.present_entry_modules, step_where)
        step_runs = [run for run in runs if run.step == step.name]
        if step.devices:
            step_rows = judge_template_step(step, images, pins, rows, step_runs, step_where)
            derived_rows.extend(step_rows)
            new_rows = [derived_row.row for derived_row in step_rows]
        else:
            if step_runs:
                raise RecipeContractError(
                    f"{step_where} produces the run {step_runs[0].name}, and has no template of "
                    "its rows"
                )
            new_rows = [judge_plain_step(step, images, pins, rows, step_where)]
        for row in new_rows:
            if row.name in rows:
                raise RecipeContractError(
                    f"{step_where} derives the row {row.name}, which another row already holds"
                )
            try:
                judge_row(row, facts.host_cpu_count)
            except StepTableError as error:
                raise RecipeContractError(f"{where}: {error}") from error
            rows[row.name] = row
    return tuple(rows.values()), tuple(derived_rows)


def judge_plain_step(
    step: RecipeStep,
    images: JudgedImages,
    pins: Sequence[SourcePin],
    rows: Mapping[str, StepRow],
    where: str,
) -> StepRow:
    if step.image is None or step.limits is None:
        raise RecipeContractError(f"{where} has neither an image and limits nor a template")
    image = find_image(step.image, images, where)
    mounts = judge_mounts(step, image, images, pins, rows, where)
    return StepRow(
        name=step.name,
        image=image,
        entry_module=step.entry_module,
        sources=mounts.sources,
        has_request=False,
        outputs=step.outputs,
        limits=step.limits,
        uses_gpu=step.uses_gpu,
        earlier_outputs=mounts.earlier_outputs,
        binds_candidate=mounts.binds_candidate,
    )


def judge_template_step(
    step: RecipeStep,
    images: JudgedImages,
    pins: Sequence[SourcePin],
    rows: Mapping[str, StepRow],
    runs: Sequence[JudgedRun],
    where: str,
) -> tuple[DerivedRow, ...]:
    if step.image is not None or step.limits is not None or step.uses_gpu:
        raise RecipeContractError(
            f"{where} has a template of its rows, which gives each device its image, its limits "
            "and the GPU"
        )
    if not runs:
        raise RecipeContractError(f"{where} has a template of its rows, and no run")
    device_rows: dict[Device, DeviceRow] = {}
    mounts: JudgedMounts | None = None
    for template in step.devices:
        device_where = f"{where}: its template for {template.device!a}"
        if template.device not in set(Device):
            raise RecipeContractError(f"{device_where} names no device among cpu, gpu")
        device = Device(template.device)
        if device in device_rows:
            raise RecipeContractError(f"{device_where} is declared twice")
        image = find_image(template.image, images, device_where)
        if image.device is not device:
            raise RecipeContractError(
                f"{device_where} runs in the {image.device} image {image.tag}"
            )
        judge_shard_outputs(template, device, device_where)
        mounts = judge_mounts(step, image, images, pins, rows, where)
        device_rows[device] = DeviceRow(image, template.limits, template.outputs)
    for run in runs:
        if run.device not in device_rows:
            raise RecipeContractError(
                f"{where}: its run {run.name} runs on the {run.device}, which its template does "
                "not give"
            )
    if mounts is None:
        raise RecipeContractError(f"{where} has a template that names no device")
    return derive_rows(
        StepTemplate(
            name=step.name,
            entry_module=step.entry_module,
            sources=mounts.sources,
            earlier_outputs=mounts.earlier_outputs,
            binds_candidate=mounts.binds_candidate,
            outputs=step.outputs,
            devices=device_rows,
        ),
        [
            ShardedRun(run.name, run.variant, run.device, run.languages, run.shard_count)
            for run in runs
        ],
    )


def judge_shard_outputs(template: DeviceTemplate, device: Device, where: str) -> None:
    names = {output.name for output in template.outputs}
    if RUN_LINES_FILE_NAME not in names:
        raise RecipeContractError(f"{where} writes no {RUN_LINES_FILE_NAME}, the lines of a run")
    if device is Device.GPU and PLACEMENT_FILE_NAME not in names:
        raise RecipeContractError(
            f"{where} writes no {PLACEMENT_FILE_NAME}, the placement of its nodes on the GPU"
        )
    if device is not Device.GPU and PLACEMENT_FILE_NAME in names:
        raise RecipeContractError(
            f"{where} writes {PLACEMENT_FILE_NAME}, which only a run on the GPU writes"
        )


def find_image(image_name: str, images: JudgedImages, where: str) -> ContainerImage:
    image = images.by_name.get(image_name) or CORE_IMAGES_BY_NAME.get(image_name)
    if image is None:
        raise RecipeContractError(
            f"{where} runs in the image {image_name!a}, which neither the recipe nor the core "
            "declares"
        )
    return image


def judge_entry_module(
    step: RecipeStep, recipe_name: str, present_entry_modules: frozenset[str], where: str
) -> None:
    if not ENTRY_MODULE_GRAMMAR.fullmatch(step.entry_module):
        raise RecipeContractError(
            f"{where}: its entry module {step.entry_module!a} is not a dotted module name"
        )
    if step.entry_module not in present_entry_modules:
        module_file = locate_entry_module_file(step.entry_module, recipe_name)
        place = "outside the core and the recipe" if module_file is None else f"at {module_file}"
        raise RecipeContractError(
            f"{where}: its entry module {step.entry_module} has no regular file {place}"
        )


@dataclass(frozen=True, slots=True)
class JudgedMounts:
    sources: tuple[SourceReference, ...]
    earlier_outputs: tuple[EarlierOutput, ...]
    binds_candidate: bool


def judge_mounts(
    step: RecipeStep,
    image: ContainerImage,
    images: JudgedImages,
    pins: Sequence[SourcePin],
    earlier_rows: Mapping[str, StepRow],
    where: str,
) -> JudgedMounts:
    sources: list[SourceReference] = []
    earlier_outputs: list[EarlierOutput] = []
    binds_candidate = False
    for mount in step.mounts:
        if isinstance(mount, CandidateMount):
            if binds_candidate:
                raise RecipeContractError(f"{where} binds the candidate twice")
            binds_candidate = True
        elif isinstance(mount, SourceMount):
            pin = find_pinned_file(mount, pins, where)
            if pin.runs_code_when_loaded and not reads_code_running_sources(image, images):
                raise RecipeContractError(
                    f"{where} binds {mount.path} of {mount.repository}, which runs code when it "
                    f"loads, in the image {image.tag}, which the recipe does not declare to read it"
                )
            reference = SourceReference(mount.repository, mount.path)
            if reference in sources:
                raise RecipeContractError(f"{where} binds {mount.path} of {mount.repository} twice")
            sources.append(reference)
        else:
            earlier_output = find_earlier_output(mount, earlier_rows, where)
            if earlier_output in earlier_outputs:
                raise RecipeContractError(f"{where} binds {mount.output} of {mount.step} twice")
            earlier_outputs.append(earlier_output)
    return JudgedMounts(tuple(sources), tuple(earlier_outputs), binds_candidate)


def reads_code_running_sources(image: ContainerImage, images: JudgedImages) -> bool:
    return image.owner != CORE_OWNER and image.name == images.code_reader


def find_pinned_file(mount: SourceMount, pins: Sequence[SourcePin], where: str) -> SourcePin:
    for pin in pins:
        if (pin.repository, pin.path) == (mount.repository, mount.path):
            return pin
    directory_prefix = mount.path.rstrip(PATH_SEPARATOR) + PATH_SEPARATOR
    if any(
        pin.repository == mount.repository and pin.path.startswith(directory_prefix) for pin in pins
    ):
        raise RecipeContractError(
            f"{where} binds the directory {mount.path!a} of {mount.repository}, where only a "
            "pinned file may be bound"
        )
    raise RecipeContractError(
        f"{where} binds {mount.path!a} of {mount.repository!a}, which no source pins"
    )


def find_earlier_output(
    mount: EarlierOutputMount, earlier_rows: Mapping[str, StepRow], where: str
) -> EarlierOutput:
    earlier_row = earlier_rows.get(mount.step)
    if earlier_row is None:
        raise RecipeContractError(
            f"{where} binds an output of {mount.step!a}, which is not an earlier step of the recipe"
        )
    if not mount.output:
        raise RecipeContractError(
            f"{where} binds the outputs of {mount.step} as a directory, where only a declared "
            "output file may be bound"
        )
    output_names = tuple(output.name for output in earlier_row.outputs)
    for output in earlier_row.outputs:
        if output.name == mount.output:
            return EarlierOutput(mount.step, output, output_names)
    raise RecipeContractError(
        f"{where} binds {mount.output!a} of {mount.step}, which that step does not declare"
    )


def judge_candidate(
    candidate: Sequence[CandidateFile], rows: Sequence[StepRow], where: str
) -> tuple[JudgedCandidateFile, ...]:
    outputs_by_step = {row.name: {output.name for output in row.outputs} for row in rows}
    judged: dict[str, JudgedCandidateFile] = {}
    for candidate_file in candidate:
        file_where = f"{where}: its candidate file {candidate_file.name!a}"
        judge_candidate_name(candidate_file.name, file_where)
        if candidate_file.name in judged:
            raise RecipeContractError(f"{file_where} is declared twice")
        if candidate_file.file_format not in set(FileFormat):
            formats = ", ".join(file_format.value for file_format in FileFormat)
            raise RecipeContractError(
                f"{file_where} has the format {candidate_file.file_format!a}, not one of {formats}"
            )
        judge_variants(candidate_file.variants, file_where)
        judge_origin(candidate_file.origin, outputs_by_step, file_where)
        judged[candidate_file.name] = JudgedCandidateFile(
            name=candidate_file.name,
            file_format=FileFormat(candidate_file.file_format),
            variants=candidate_file.variants,
            origin=candidate_file.origin,
            external_data_of=candidate_file.external_data_of,
        )
    for judged_file in judged.values():
        judge_external_data_of(
            judged_file, judged, f"{where}: its candidate file {judged_file.name}"
        )
    return tuple(judged.values())


def judge_external_data_of(
    candidate_file: JudgedCandidateFile,
    candidate: Mapping[str, JudgedCandidateFile],
    where: str,
) -> None:
    owner_name = candidate_file.external_data_of
    if candidate_file.file_format is not FileFormat.ONNX_EXTERNAL_DATA:
        if owner_name is not None:
            raise RecipeContractError(
                f"{where} is {candidate_file.file_format.value}, and names the onnx file whose "
                "tensors it holds"
            )
        return
    if owner_name is None:
        raise RecipeContractError(f"{where} holds external data, and names no onnx file it serves")
    owner = candidate.get(owner_name)
    if owner is None or owner.file_format is not FileFormat.ONNX:
        raise RecipeContractError(
            f"{where} holds the tensors of {owner_name!a}, which is not an onnx file of the "
            "candidate"
        )
    if set(owner.variants) != set(candidate_file.variants):
        raise RecipeContractError(
            f"{where} serves {', '.join(candidate_file.variants)}, and its onnx file "
            f"{owner_name} serves {', '.join(owner.variants)}"
        )


def judge_conversion_input(
    conversion_input: StepOutputOrigin | None,
    candidate: Sequence[JudgedCandidateFile],
    where: str,
) -> StepOutputOrigin | None:
    if conversion_input is None:
        return None
    if not any(
        candidate_file.file_format is FileFormat.ONNX and candidate_file.origin == conversion_input
        for candidate_file in candidate
    ):
        raise RecipeContractError(
            f"{where}: its conversions read {conversion_input.output!a} of "
            f"{conversion_input.step!a}, which no onnx file of the candidate comes from"
        )
    return conversion_input


def judge_conversions_of_recipe(
    contract: RecipeContract, rows: Sequence[StepRow], host_cpu_count: int, where: str
) -> StepRow | None:
    if not contract.conversions:
        return None
    conversion_input = contract.conversion_input
    if conversion_input is None:
        raise RecipeContractError(f"{where} converts its variants, and names no conversions' input")
    variants: set[str] = set()
    for variant_conversion in contract.conversions:
        variant = variant_conversion.variant
        if not isinstance(variant, str) or not VARIANT_GRAMMAR.fullmatch(variant):
            raise RecipeContractError(
                f"{where}: its conversion to the variant {variant!a} is not named under "
                f"{VARIANT_GRAMMAR.pattern}"
            )
        if variant in variants:
            raise RecipeContractError(
                f"{where}: its conversion to the variant {variant} is declared twice"
            )
        variants.add(variant)
    try:
        judge_conversions([variant.conversion for variant in contract.conversions])
    except ConversionRequestError as error:
        raise RecipeContractError(f"{where}: {error}") from error
    for variant_conversion in contract.conversions:
        judge_candidate_name(
            variant_conversion.conversion.output,
            f"{where}: the output of its conversion {variant_conversion.variant}",
        )
    earlier_rows = {row.name: row for row in rows}
    inputs = [
        find_earlier_output(
            EarlierOutputMount(conversion_input.step, conversion_input.output),
            earlier_rows,
            f"{where}: its conversions",
        )
    ]
    external_data = find_external_data_of_origin(contract.candidate, conversion_input)
    if external_data is not None:
        if (
            not isinstance(external_data.origin, StepOutputOrigin)
            or external_data.origin.step != conversion_input.step
        ):
            raise RecipeContractError(
                f"{where}: the external data {external_data.name} of the conversions' input does "
                f"not come from {conversion_input.step}, beside its onnx file"
            )
        inputs.append(
            find_earlier_output(
                EarlierOutputMount(conversion_input.step, external_data.origin.output),
                earlier_rows,
                f"{where}: its conversions",
            )
        )
    row = build_convert_row(
        tuple(inputs),
        tuple(
            DeclaredOutput(variant.conversion.output, variant.max_size_bytes)
            for variant in contract.conversions
        ),
    )
    try:
        judge_row(row, host_cpu_count)
    except StepTableError as error:
        raise RecipeContractError(f"{where}: {error}") from error
    return row


def find_external_data_of_origin(
    candidate: Sequence[CandidateFile], origin: StepOutputOrigin
) -> CandidateFile | None:
    onnx_names = {
        candidate_file.name
        for candidate_file in candidate
        if candidate_file.origin == origin and candidate_file.file_format == FileFormat.ONNX
    }
    for candidate_file in candidate:
        if candidate_file.external_data_of in onnx_names:
            return candidate_file
    return None


def judge_converted_files(
    conversions: Sequence[VariantConversion],
    candidate: Sequence[JudgedCandidateFile],
    where: str,
) -> None:
    for variant_conversion in conversions:
        origin = StepOutputOrigin(CONVERT_STEP_NAME, variant_conversion.conversion.output)
        if not any(
            candidate_file.origin == origin
            and candidate_file.file_format is FileFormat.ONNX
            and candidate_file.variants == (variant_conversion.variant,)
            for candidate_file in candidate
        ):
            raise RecipeContractError(
                f"{where}: its conversion to the variant {variant_conversion.variant} writes "
                f"{variant_conversion.conversion.output}, which no onnx file of the candidate "
                "serving that variant alone comes from"
            )


def judge_build_order(
    build_order: Sequence[str], steps: Sequence[RecipeStep], converts: bool, where: str
) -> tuple[str, ...]:
    known = {*BUILT_CORE_STEP_NAMES, *(step.name for step in steps)}
    if converts:
        known.add(CONVERT_STEP_NAME)
    for step_name in build_order:
        if step_name not in known:
            raise RecipeContractError(
                f"{where}: its build runs {step_name!a}, which neither the core nor the recipe "
                "builds with"
            )
    if len(set(build_order)) != len(build_order):
        raise RecipeContractError(f"{where}: its build runs a step twice")
    return tuple(build_order)


def judge_candidate_name(name: str, where: str) -> None:
    if not is_file_name(name):
        raise RecipeContractError(f"{where} is not a file name")
    if name.casefold() == MODEL_CARD_FILE_NAME.casefold():
        raise RecipeContractError(
            f"{where} bears the name of the model card, {MODEL_CARD_FILE_NAME}, which the core "
            "writes"
        )
    refused_suffixes = find_refused_suffixes(name, REFUSED_SUFFIXES)
    if refused_suffixes:
        raise RecipeContractError(
            f"{where} has the refused suffix {refused_suffixes[0]}, which marks code, an archive "
            "or a format that runs code when it loads"
        )


def judge_variants(variants: Sequence[str], where: str) -> None:
    if not variants:
        raise RecipeContractError(f"{where} serves no variant")
    for variant in variants:
        if not VARIANT_GRAMMAR.fullmatch(variant):
            raise RecipeContractError(
                f"{where} serves the variant {variant!a}, not named under {VARIANT_GRAMMAR.pattern}"
            )
    if len(set(variants)) != len(variants):
        raise RecipeContractError(f"{where} names a variant twice")


def judge_origin(
    origin: StepOutputOrigin | HostContentOrigin,
    outputs_by_step: Mapping[str, set[str]],
    where: str,
) -> None:
    if isinstance(origin, HostContentOrigin):
        if not 0 < len(origin.content) <= MAX_HOST_CONTENT_SIZE_BYTES:
            raise RecipeContractError(
                f"{where} has a content of {len(origin.content)} bytes, outside 1 to "
                f"{MAX_HOST_CONTENT_SIZE_BYTES}"
            )
        return
    if origin.output not in outputs_by_step.get(origin.step, set()):
        raise RecipeContractError(
            f"{where} comes from {origin.output!a} of {origin.step!a}, which no step of the "
            "recipe declares"
        )


def judge_parities(
    parities: Sequence[Parity], rows: Sequence[StepRow], where: str
) -> tuple[Parity, ...]:
    outputs_by_step = {row.name: {output.name for output in row.outputs} for row in rows}
    names: set[str] = set()
    for parity in parities:
        parity_where = f"{where}: its parity {parity.name!a}"
        if not PARITY_NAME_GRAMMAR.fullmatch(parity.name):
            raise RecipeContractError(
                f"{parity_where} is not named under {PARITY_NAME_GRAMMAR.pattern}"
            )
        if parity.name in names:
            raise RecipeContractError(f"{parity_where} is declared twice")
        names.add(parity.name)
        if parity.output not in outputs_by_step.get(parity.step, set()):
            raise RecipeContractError(
                f"{parity_where} reads its report from {parity.output!a} of {parity.step!a}, "
                "which no step of the recipe declares"
            )
        if parity.output == PEAKS_FILE_NAME:
            raise RecipeContractError(
                f"{parity_where} reads its report from {PEAKS_FILE_NAME}, the peaks of its step"
            )
        judge_tolerance(parity.absolute_difference_tolerance, parity_where)
    return tuple(parities)


def judge_tolerance(tolerance: float, where: str) -> None:
    if (
        not isinstance(tolerance, int | float)
        or isinstance(tolerance, bool)
        or not math.isfinite(tolerance)
        or tolerance <= 0
    ):
        raise RecipeContractError(
            f"{where} has the tolerance {tolerance!a}, not a finite number above 0"
        )


def judge_comparisons(
    comparisons: Sequence[Comparison], runs: Sequence[JudgedRun], where: str
) -> tuple[Comparison, ...]:
    languages_by_run = {run.name: run.languages for run in runs}
    pairs: set[tuple[str, str]] = set()
    for comparison in comparisons:
        comparison_where = (
            f"{where}: its comparison of {comparison.run!a} against {comparison.reference_run!a}"
        )
        for run_name in (comparison.run, comparison.reference_run):
            if run_name not in languages_by_run:
                raise RecipeContractError(
                    f"{comparison_where} names {run_name!a}, which is not a run of the recipe"
                )
        if comparison.run == comparison.reference_run:
            raise RecipeContractError(f"{comparison_where} compares a run with itself")
        pair = (comparison.run, comparison.reference_run)
        if pair in pairs:
            raise RecipeContractError(f"{comparison_where} is declared twice")
        pairs.add(pair)
        if not set(languages_by_run[comparison.run]) & set(
            languages_by_run[comparison.reference_run]
        ):
            raise RecipeContractError(f"{comparison_where} compares runs that share no language")
        try:
            judge_comparison_rule(comparison.rule)
        except AcceptanceError as error:
            raise RecipeContractError(f"{comparison_where}: {error}") from error
    return tuple(comparisons)
