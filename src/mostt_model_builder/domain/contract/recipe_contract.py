import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import PurePosixPath

from mostt_model_builder.domain.acceptance.parity_reports import PARITY_NAME_GRAMMAR
from mostt_model_builder.domain.containers.base_images import BASE_PYTHON, ENVIRONMENT_PYTHON
from mostt_model_builder.domain.containers.container_steps import (
    CLEAN_STEP_NAME,
    CORE_IMAGES,
    CORE_OWNER,
    CORE_PACKAGE,
    ENTRY_MODULE_GRAMMAR,
    FLEURS_CHECK_STEP_NAME,
    LOCK_STEP_NAME,
    PEAKS_FILE_NAME,
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
    find_package_of_owner,
    judge_row,
)
from mostt_model_builder.domain.formats.file_formats import FileFormat
from mostt_model_builder.domain.formats.file_names import (
    REFUSED_SUFFIXES,
    find_refused_suffixes,
    is_file_name,
)
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
CORE_STEP_NAMES = frozenset({FLEURS_CHECK_STEP_NAME, CLEAN_STEP_NAME, LOCK_STEP_NAME})
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


type Mount = SourceMount | EarlierOutputMount


@dataclass(frozen=True, slots=True)
class RecipeStep:
    name: str
    image: str
    entry_module: str
    mounts: tuple[Mount, ...]
    outputs: tuple[DeclaredOutput, ...]
    limits: StepLimits
    uses_gpu: bool = False


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


@dataclass(frozen=True, slots=True)
class Parity:
    name: str
    step: str
    output: str
    absolute_difference_tolerance: float


@dataclass(frozen=True, slots=True)
class RecipeContract:
    name: str
    images: tuple[RecipeImage, ...]
    steps: tuple[RecipeStep, ...]
    candidate: tuple[CandidateFile, ...]
    parities: tuple[Parity, ...] = ()


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


@dataclass(frozen=True, slots=True)
class JudgedRecipe:
    name: str
    pins: tuple[SourcePin, ...]
    images: tuple[ContainerImage, ...]
    rows: tuple[StepRow, ...]
    candidate: tuple[JudgedCandidateFile, ...]
    parities: tuple[Parity, ...]


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
    rows = judge_steps(contract, images, pins, facts, where)
    return JudgedRecipe(
        name=contract.name,
        pins=pins,
        images=tuple(images.by_name[image.name] for image in contract.images),
        rows=rows,
        candidate=judge_candidate(contract.candidate, rows, where),
        parities=judge_parities(contract.parities, rows, where),
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


def judge_steps(
    contract: RecipeContract,
    images: JudgedImages,
    pins: Sequence[SourcePin],
    facts: RecipeFacts,
    where: str,
) -> tuple[StepRow, ...]:
    rows: dict[str, StepRow] = {}
    for step in contract.steps:
        step_where = f"{where}: its step {step.name!a}"
        if not STEP_NAME_GRAMMAR.fullmatch(step.name):
            raise RecipeContractError(
                f"{step_where} is not named under {STEP_NAME_GRAMMAR.pattern}"
            )
        if step.name in CORE_STEP_NAMES:
            raise RecipeContractError(f"{step_where} bears the name of a step of the core")
        if step.name in rows:
            raise RecipeContractError(f"{step_where} is declared twice")
        image = find_image(step, images, step_where)
        judge_entry_module(step, contract.name, facts.present_entry_modules, step_where)
        sources, earlier_outputs = judge_mounts(step, image, images, pins, rows, step_where)
        row = StepRow(
            name=step.name,
            image=image,
            entry_module=step.entry_module,
            sources=sources,
            has_request=False,
            outputs=step.outputs,
            limits=step.limits,
            uses_gpu=step.uses_gpu,
            earlier_outputs=earlier_outputs,
        )
        try:
            judge_row(row, facts.host_cpu_count)
        except StepTableError as error:
            raise RecipeContractError(f"{where}: {error}") from error
        rows[step.name] = row
    return tuple(rows.values())


def find_image(step: RecipeStep, images: JudgedImages, where: str) -> ContainerImage:
    image = images.by_name.get(step.image) or CORE_IMAGES_BY_NAME.get(step.image)
    if image is None:
        raise RecipeContractError(
            f"{where} runs in the image {step.image!a}, which neither the recipe nor the core "
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


def judge_mounts(
    step: RecipeStep,
    image: ContainerImage,
    images: JudgedImages,
    pins: Sequence[SourcePin],
    earlier_rows: Mapping[str, StepRow],
    where: str,
) -> tuple[tuple[SourceReference, ...], tuple[EarlierOutput, ...]]:
    sources: list[SourceReference] = []
    earlier_outputs: list[EarlierOutput] = []
    for mount in step.mounts:
        if isinstance(mount, SourceMount):
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
    return tuple(sources), tuple(earlier_outputs)


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
        )
    return tuple(judged.values())


def judge_candidate_name(name: str, where: str) -> None:
    if not is_file_name(name):
        raise RecipeContractError(f"{where} is not a file name")
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
