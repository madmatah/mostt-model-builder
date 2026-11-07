import argparse
import os
import pwd
import stat
import sys
import tomllib
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from mostt_model_builder.adapters.curl_downloader import CurlDownloader
from mostt_model_builder.adapters.docker_engine import ContainerEngineError, DockerEngine
from mostt_model_builder.adapters.git_tree import GitError, GitTree
from mostt_model_builder.adapters.terminal_prompt import TerminalError, TerminalPrompt
from mostt_model_builder.adapters.work_files import WorkFiles
from mostt_model_builder.application.assemble_candidate import (
    CandidateError,
    CandidateRequest,
    assemble_candidate,
    read_candidate_texts,
)
from mostt_model_builder.application.build_images import ImageBuildError, build_images
from mostt_model_builder.application.check_fleurs import FleursCheckRun, check_fleurs_bench
from mostt_model_builder.application.check_published import (
    PublishedCheckError,
    PublishedCheckRequest,
    check_published,
)
from mostt_model_builder.application.clean_run import (
    CleanOutcome,
    CleanRequest,
    clean_step,
    remove_candidate,
    remove_model_card,
    remove_verification_report,
)
from mostt_model_builder.application.convert_variants import (
    ConversionOutcome,
    ConversionRun,
    convert_variants,
)
from mostt_model_builder.application.fetch_sources import (
    DownloadError,
    FetchError,
    FetchOutcome,
    SourceFileError,
    fetch_source,
)
from mostt_model_builder.application.open_work_directory import (
    WorkDirectoryRequest,
    open_work_directory,
)
from mostt_model_builder.application.prepare_publication import (
    CardOutcome,
    CardRequest,
    PublicationError,
    prepare_publication,
    write_model_card,
)
from mostt_model_builder.application.publish_candidate import (
    CandidatePublication,
    publish_candidate,
)
from mostt_model_builder.application.resolve_lock import (
    LockRequest,
    LockResolution,
    resolve_lock,
)
from mostt_model_builder.application.run_recipe_step import (
    MeasuredParity,
    RecipeStepOutcome,
    RecipeStepRun,
    RunShard,
    run_recipe_step,
)
from mostt_model_builder.application.run_step import HostIdentity, StepError
from mostt_model_builder.application.verify_build import (
    BuildRejectedError,
    VerificationRequest,
    refuse_rejected_build,
    verify_build,
)
from mostt_model_builder.domain.acceptance.acceptance import AcceptanceError, name_rule
from mostt_model_builder.domain.acceptance.paired_comparison import PairedComparisonError
from mostt_model_builder.domain.acceptance.parity_reports import ParityReportError
from mostt_model_builder.domain.acceptance.verification import (
    ComparisonMeasure,
    ParityMeasure,
    RunMeasure,
    Verification,
    VerificationError,
    compute_run_word_error_rate_percent,
    compute_run_word_error_rate_with_unknown_tokens_percent,
)
from mostt_model_builder.domain.bench.fleurs_check_report import FleursCheckReportError
from mostt_model_builder.domain.bench.hypotheses import Hypothesis, HypothesisError
from mostt_model_builder.domain.bench.run_coverage import RunCoverageError
from mostt_model_builder.domain.bench.word_error_rate import WordErrorRateError
from mostt_model_builder.domain.containers.account_ranges import (
    AccountRangeError,
    HostAccounts,
    parse_subordinate_ranges,
)
from mostt_model_builder.domain.containers.container_files import ContainerFileError
from mostt_model_builder.domain.containers.container_steps import (
    CLEAN_STEP_NAME,
    CONTAINER_USER_ID,
    CONVERT_STEP_NAME,
    CORE_IMAGES,
    FLEURS_CHECK_STEP_NAME,
    INSPECT_STEP_NAME,
    LOCK_STEP_NAME,
    RUNNABLE_CORE_ROWS,
    StepRow,
    StepTableError,
)
from mostt_model_builder.domain.containers.provenance import ProvenanceError
from mostt_model_builder.domain.containers.step_peaks import StepPeaks, StepPeaksError
from mostt_model_builder.domain.containers.work_directory import (
    WorkDirectory,
    WorkDirectoryError,
    locate_source,
)
from mostt_model_builder.domain.contract.derived_rows import DerivedRow
from mostt_model_builder.domain.contract.recipe_contract import (
    RECIPES_DIRECTORY_NAME,
    JudgedRecipe,
    JudgedRun,
    RecipeContract,
    RecipeContractError,
    RecipeFacts,
    find_runnable_row,
    judge_recipe,
    list_entry_modules,
    list_rows_of_step,
    list_runnable_rows,
    locate_card_template,
    locate_entry_module_file,
)
from mostt_model_builder.domain.formats.external_data import ExternalDataError
from mostt_model_builder.domain.lock.declared_requirements import (
    DeclaredRequirement,
    read_declared_requirements,
)
from mostt_model_builder.domain.lock.lock_check import (
    LOCK_FILE_NAME,
    MANIFEST_FILE_NAME,
    LockError,
)
from mostt_model_builder.domain.lock.lock_freshness import LockFreshnessError
from mostt_model_builder.domain.onnx.conversion_report import ConversionReportError
from mostt_model_builder.domain.onnx.conversion_request import ConversionRequestError
from mostt_model_builder.domain.onnx.graph_rules import (
    GraphRuleError,
    count_eight_bit_block_products,
    count_products,
)
from mostt_model_builder.domain.onnx.inspection_report import InspectionReportError
from mostt_model_builder.domain.onnx.node_placement import NodePlacementError, PlacementSummary
from mostt_model_builder.domain.publication.hub_token import HubTokenError
from mostt_model_builder.domain.publication.publication_report import PublicationReportError
from mostt_model_builder.domain.publication.repository_files import RepositoryFilesError
from mostt_model_builder.domain.sources.source_pins import (
    SourcePin,
    SourcePinError,
    parse_source_pins,
)
from mostt_recipe_parakeet_tdt.recipe import RECIPE as PARAKEET_TDT_RECIPE

PROGRAM_NAME = "model-build.sh"
DESCRIPTION = (
    "Builds speech recognition models from their original weights, in confined containers, "
    "and verifies them on FLEURS."
)
CLONE_ROOT = Path(__file__).resolve().parents[2]
SOURCES_FILE_NAME = "sources.toml"
IMAGES_ENVIRONMENT_DIRECTORY = CLONE_ROOT / "images"
SUBORDINATE_USER_IDS_PATH = Path("/etc/subuid")
SUBORDINATE_GROUP_IDS_PATH = Path("/etc/subgid")
FETCH_COMMAND = "fetch"
IMAGES_COMMAND = "images"
STEP_COMMAND = "step"
BUILD_COMMAND = "build"
VERIFY_COMMAND = "verify"
CARD_COMMAND = "card"
CHECK_PUBLISHED_COMMAND = "check-published"
PUBLISH_COMMAND = "publish"
STANDARD_INPUT_DESCRIPTOR = 0
CPU_PROVIDER = "CPUExecutionProvider"
CUDA_PROVIDER = "CUDAExecutionProvider"
RECIPES = {contract.name: contract for contract in (PARAKEET_TDT_RECIPE,)}
STEP_NAMES = sorted(
    {row.name for row in RUNNABLE_CORE_ROWS}
    | {CONVERT_STEP_NAME}
    | {step.name for contract in RECIPES.values() for step in contract.steps}
)
TEXT_ENCODING = "utf-8"
FAILURE_EXIT_STATUS = 1
NO_ACCOUNT_HOME = ""
CARD_OUTCOME_WORDS = {
    CardOutcome.WRITTEN: "wrote",
    CardOutcome.UNCHANGED: "unchanged:",
    CardOutcome.REPLACED: "replaced",
}
OUTCOME_WORDS = {FetchOutcome.FETCHED: "fetched", FetchOutcome.ALREADY_FETCHED: "already fetched"}
CLEAN_OUTCOME_WORDS = {
    CleanOutcome.CLEANED: "cleaned",
    CleanOutcome.NOTHING_TO_CLEAN: "nothing to clean for",
}
COMMAND_ERRORS = (
    SourcePinError,
    WorkDirectoryError,
    FetchError,
    DownloadError,
    SourceFileError,
    StepError,
    StepTableError,
    StepPeaksError,
    ContainerFileError,
    ContainerEngineError,
    AccountRangeError,
    ProvenanceError,
    GitError,
    ImageBuildError,
    FleursCheckReportError,
    ParityReportError,
    RecipeContractError,
    LockError,
    LockFreshnessError,
    CandidateError,
    InspectionReportError,
    GraphRuleError,
    ExternalDataError,
    NodePlacementError,
    HypothesisError,
    ConversionRequestError,
    ConversionReportError,
    VerificationError,
    RunCoverageError,
    PairedComparisonError,
    AcceptanceError,
    WordErrorRateError,
    BuildRejectedError,
    PublicationError,
    PublishedCheckError,
    TerminalError,
    HubTokenError,
    PublicationReportError,
    RepositoryFilesError,
    ValueError,
    OSError,
)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=PROGRAM_NAME, description=DESCRIPTION, add_help=False, allow_abbrev=False
    )
    parser.add_argument("--help", action="help", help="show this message and exit")
    parser.add_argument(
        "--work",
        required=True,
        metavar="<dir>",
        help="the work directory: an existing directory of yours, in mode 0700, outside the "
        "home directory and the clone",
    )
    parser.add_argument(
        "--recipe",
        choices=sorted(RECIPES),
        metavar="<name>",
        help=f"the recipe of the model family to build, which every command but {LOCK_STEP_NAME} "
        f"needs: {', '.join(sorted(RECIPES))}",
    )
    commands = parser.add_subparsers(dest="command", required=True, metavar="<command>")
    commands.add_parser(
        FETCH_COMMAND,
        add_help=False,
        help="download the pinned sources into the work directory, each checked by its size "
        "and SHA-256",
    )
    commands.add_parser(
        LOCK_STEP_NAME,
        add_help=False,
        help="resolve the lock of the images' environment in its container, check it, print "
        "its change, and write it to images/environment.lock",
    )
    commands.add_parser(
        IMAGES_COMMAND,
        add_help=False,
        help="build the images of the containers from a clean clone, labeled with its tree",
    )
    step_parser = commands.add_parser(
        STEP_COMMAND,
        add_help=False,
        help="run a step of the core or of the recipe in its container",
    )
    step_parser.add_argument(
        "step", choices=STEP_NAMES, metavar="<step>", help=", ".join(STEP_NAMES)
    )
    commands.add_parser(
        VERIFY_COMMAND,
        add_help=False,
        help="measure the runs on FLEURS, hold the parities, the runs and their comparisons to "
        "the recipe's rules, and write verification.json",
    )
    commands.add_parser(
        BUILD_COMMAND,
        add_help=False,
        help="run every step of the recipe's build, in its order, each row of a run in its own "
        "container, then verify it",
    )
    commands.add_parser(
        CARD_COMMAND,
        add_help=False,
        help="verify the build again from a clean clone, render its model card, the README.md of "
        "its Hugging Face repository, and write it beside verification.json for review",
    )
    publish_parser = commands.add_parser(
        PUBLISH_COMMAND,
        add_help=False,
        help="publish the verified candidate and its reviewed card into a Hugging Face "
        "repository you created, with a token you type at the terminal; the maintainer's "
        "alone",
    )
    publish_parser.add_argument(
        "--repository",
        required=True,
        metavar="<namespace>/<name>",
        help="the Hugging Face repository to publish into, which must exist",
    )
    check_parser = commands.add_parser(
        CHECK_PUBLISHED_COMMAND,
        add_help=False,
        help="download each file of a published revision, without a token, check it against "
        "the build's own bytes and its card, and print the revision that mostt pins",
    )
    check_parser.add_argument(
        "--repository",
        required=True,
        metavar="<namespace>/<name>",
        help="the Hugging Face repository the files were published in",
    )
    check_parser.add_argument(
        "--revision",
        required=True,
        metavar="<commit>",
        help="the published commit, in its 40 hexadecimal digits",
    )
    clean_parser = commands.add_parser(
        CLEAN_STEP_NAME,
        add_help=False,
        help="remove the outputs, the request and the log of a step, or of every step",
    )
    clean_parser.add_argument(
        "step", nargs="?", choices=STEP_NAMES, metavar="<step>", help=", ".join(STEP_NAMES)
    )
    return parser


def find_account_home() -> str:
    try:
        return pwd.getpwuid(os.getuid()).pw_dir
    except KeyError:
        return NO_ACCOUNT_HOME


def has_account(user_id: int) -> bool:
    try:
        pwd.getpwuid(user_id)
    except KeyError:
        return False
    return True


def read_optional_text(path: Path) -> str:
    try:
        return path.read_text(encoding=TEXT_ENCODING)
    except FileNotFoundError:
        return ""


def read_host_identity() -> HostIdentity:
    return HostIdentity(
        user_id=os.getuid(),
        group_id=os.getgid(),
        cpu_count=len(os.sched_getaffinity(0)),
        accounts=HostAccounts(
            has_container_user_account=has_account(CONTAINER_USER_ID),
            subordinate_user_ranges=parse_subordinate_ranges(
                read_optional_text(SUBORDINATE_USER_IDS_PATH), str(SUBORDINATE_USER_IDS_PATH)
            ),
            subordinate_group_ranges=parse_subordinate_ranges(
                read_optional_text(SUBORDINATE_GROUP_IDS_PATH), str(SUBORDINATE_GROUP_IDS_PATH)
            ),
        ),
    )


def open_requested_work_directory(path: str, files: WorkFiles) -> WorkDirectory:
    request = WorkDirectoryRequest(
        path=path,
        home_variable=os.environ.get("HOME"),
        account_home=find_account_home(),
        clone_root=str(CLONE_ROOT),
        user_id=os.getuid(),
    )
    return open_work_directory(request, files)


def read_pins(directory: Path) -> tuple[SourcePin, ...]:
    return parse_source_pins((directory / SOURCES_FILE_NAME).read_text(encoding=TEXT_ENCODING))


def is_regular_file(path: Path) -> bool:
    try:
        return stat.S_ISREG(path.lstat().st_mode)
    except FileNotFoundError:
        return False


def find_present_entry_modules(contract: RecipeContract) -> frozenset[str]:
    present_entry_modules = set()
    for entry_module in list_entry_modules(contract):
        module_file = locate_entry_module_file(entry_module, contract.name)
        if module_file is not None and is_regular_file(CLONE_ROOT / module_file):
            present_entry_modules.add(entry_module)
    return frozenset(present_entry_modules)


def is_card_template_present(contract: RecipeContract, clone_root: Path) -> bool:
    card_template = locate_card_template(contract.name, contract.model)
    return card_template is not None and is_regular_file(clone_root / card_template)


def load_recipe(name: str) -> JudgedRecipe:
    contract = RECIPES[name]
    return judge_recipe(
        contract,
        RecipeFacts(
            core_pins=read_pins(CLONE_ROOT),
            recipe_pins=read_pins(CLONE_ROOT / RECIPES_DIRECTORY_NAME / contract.name),
            present_entry_modules=find_present_entry_modules(contract),
            host_cpu_count=len(os.sched_getaffinity(0)),
            has_card_template=is_card_template_present(contract, CLONE_ROOT),
        ),
    )


def describe_peaks(peaks: StepPeaks) -> str:
    return f"peaks: {peaks.memory_peak_mib:.0f} MiB of memory, {peaks.process_count_peak} processes"


def describe_lock_resolution(resolution: LockResolution) -> str:
    return (
        f"wrote images/{LOCK_FILE_NAME}, {len(resolution.checked_lock.packages)} packages, "
        f"{len(resolution.recorded_extras)} extras, from {resolution.lock_run}; "
        "review its change in the diff"
    )


def describe_parity(parity: MeasuredParity) -> str:
    report = parity.report
    return (
        f"parity {report.name}: a maximum absolute difference of "
        f"{report.maximum_absolute_difference:.3g} over {report.compared_count} compared, "
        f"against a tolerance of {parity.absolute_difference_tolerance:.3g}"
    )


def fetch(recipe: JudgedRecipe, work_directory: WorkDirectory, files: WorkFiles) -> None:
    downloader = CurlDownloader()
    for pin in recipe.pins:
        source_path = locate_source(pin)
        print(f"checking {source_path}, {pin.size_bytes} bytes", flush=True)
        outcome = fetch_source(pin, work_directory, downloader, files)
        print(f"{OUTCOME_WORDS[outcome]} {source_path}", flush=True)


def build_container_images(
    recipe: JudgedRecipe, work_directory: WorkDirectory, files: WorkFiles
) -> None:
    clone_root = PurePosixPath(CLONE_ROOT)
    built_images = build_images(
        (*CORE_IMAGES, *recipe.images),
        PurePosixPath(work_directory.path),
        GitTree(clone_root),
        DockerEngine(),
        files,
    )
    for image in built_images:
        print(f"built {image.tag}, {image.image_id}", flush=True)


def check_fleurs(recipe: JudgedRecipe, work_directory: WorkDirectory, files: WorkFiles) -> None:
    clone_root = PurePosixPath(CLONE_ROOT)
    git_tree = GitTree(clone_root)
    outcome = check_fleurs_bench(
        FleursCheckRun(
            pins=recipe.pins,
            recipe_name=recipe.name,
            commit=git_tree.find_head_commit(),
            work_root=PurePosixPath(work_directory.path),
            head_tree=git_tree.find_head_tree(),
            host=read_host_identity(),
        ),
        DockerEngine(),
        files,
    )
    for language in outcome.report.languages:
        print(
            f"{language.language}: {len(language.recordings)} recordings, "
            f"{language.duration_s:.0f} s",
            flush=True,
        )
    print(describe_peaks(outcome.peaks), flush=True)


@dataclass(frozen=True, slots=True)
class BuildContext:
    commit: str
    head_tree: str
    work_root: PurePosixPath
    host: HostIdentity


def read_build_context(work_directory: WorkDirectory) -> BuildContext:
    git_tree = GitTree(PurePosixPath(CLONE_ROOT))
    return BuildContext(
        commit=git_tree.find_head_commit(),
        head_tree=git_tree.find_head_tree(),
        work_root=PurePosixPath(work_directory.path),
        host=read_host_identity(),
    )


def build_recipe_step_run(
    recipe: JudgedRecipe, row: StepRow, context: BuildContext, shard: RunShard | None = None
) -> RecipeStepRun:
    return RecipeStepRun(
        row=row,
        recipe_name=recipe.name,
        pins=recipe.pins,
        commit=context.commit,
        work_root=context.work_root,
        head_tree=context.head_tree,
        host=context.host,
        parities=recipe.parities,
        shard=shard,
        candidate=build_candidate_request(recipe, context),
    )


def describe_outcome(row: StepRow, outcome: RecipeStepOutcome) -> None:
    print(f"{row.name}:", flush=True)
    for name, size_bytes in sorted(outcome.outputs.sizes_bytes.items()):
        print(f"  wrote {name}, {size_bytes} bytes", flush=True)
    for parity in outcome.parities:
        print(f"  {describe_parity(parity)}", flush=True)
    if outcome.placement is not None:
        print(f"  {describe_placement(outcome.placement)}", flush=True)
    print(f"  {describe_peaks(outcome.outputs.peaks)}", flush=True)


def describe_placement(summary: PlacementSummary) -> str:
    return (
        f"placement: {summary.count_nodes_on(CUDA_PROVIDER)} nodes on CUDA, "
        f"{summary.count_nodes_on(CPU_PROVIDER)} on the CPU, each of integers or booleans"
    )


def describe_hypotheses(run_name: str, hypotheses: Sequence[Hypothesis]) -> list[str]:
    lines = []
    for language in sorted({hypothesis.language for hypothesis in hypotheses}):
        of_language = [hypothesis for hypothesis in hypotheses if hypothesis.language == language]
        lines.append(
            f"run {run_name}, {language}: {len(of_language)} hypotheses, "
            f"{sum(hypothesis.unknown_token_count for hypothesis in of_language)} unknown tokens, "
            f"{sum(hypothesis.decoding_time_ms for hypothesis in of_language) / 1000:.1f} s of "
            "decoding"
        )
    return lines


def run_plain_step(
    recipe: JudgedRecipe, row: StepRow, context: BuildContext, files: WorkFiles
) -> None:
    outcome = run_recipe_step(build_recipe_step_run(recipe, row, context), DockerEngine(), files)
    describe_outcome(row, outcome)


# The decoding times are read against the trial build's, which ran the shards of a run at the
# same time, each on CPUs of its own; the languages of a run run one after the other.
def run_sharded_rows(
    recipe: JudgedRecipe,
    run: JudgedRun,
    derived_rows: Sequence[DerivedRow],
    candidate_texts: Mapping[str, bytes],
    context: BuildContext,
    files: WorkFiles,
) -> None:
    engine = DockerEngine()
    step_runs = [
        build_recipe_step_run(
            recipe,
            derived_row.row,
            context,
            RunShard(derived_row.request, run.reader, candidate_texts),
        )
        for derived_row in derived_rows
    ]
    with ThreadPoolExecutor(max_workers=len(step_runs)) as pool:
        outcomes = list(
            pool.map(lambda step_run: run_recipe_step(step_run, engine, files), step_runs)
        )
    for step_run, outcome in zip(step_runs, outcomes, strict=True):
        describe_outcome(step_run.row, outcome)
    hypotheses = [hypothesis for outcome in outcomes for hypothesis in outcome.hypotheses]
    for line in describe_hypotheses(run.name, hypotheses):
        print(line, flush=True)


def build_candidate_request(recipe: JudgedRecipe, context: BuildContext) -> CandidateRequest:
    return CandidateRequest(
        recipe_name=recipe.name,
        commit=context.commit,
        work_root=context.work_root,
        head_tree=context.head_tree,
        host=context.host,
        candidate=recipe.candidate,
        rows=recipe.rows,
        conversion_input=recipe.conversion_input,
        conversions=recipe.conversions,
    )


def inspect_candidate(recipe: JudgedRecipe, context: BuildContext, files: WorkFiles) -> None:
    assembled = assemble_candidate(build_candidate_request(recipe, context), DockerEngine(), files)
    for copy in assembled.files:
        print(
            f"candidate {copy.name}, {copy.file_format.value}, {copy.size_bytes} bytes, "
            f"sha256 {copy.sha256}",
            flush=True,
        )
    for inspection in assembled.report.files:
        counts = count_products(inspection)
        print(
            f"inspected {inspection.name}: {sum(node.count for node in inspection.nodes)} nodes, "
            f"{counts.constant_weight_count} products of a constant weight, "
            f"{counts.plain_count} of them plain, "
            f"{count_eight_bit_block_products(inspection)} in 8-bit blocks, "
            f"{len(inspection.external_tensors)} external tensors",
            flush=True,
        )
    print(describe_peaks(assembled.peaks), flush=True)


def convert(recipe: JudgedRecipe, context: BuildContext, files: WorkFiles) -> None:
    if not recipe.conversion_rows:
        raise RecipeContractError(f"the recipe {recipe.name} converts none of its variants")
    engine = DockerEngine()
    for conversion_row in recipe.conversion_rows:
        outcome = convert_variants(
            ConversionRun(
                row=conversion_row.row,
                request=conversion_row.request,
                recipe_name=recipe.name,
                commit=context.commit,
                work_root=context.work_root,
                head_tree=context.head_tree,
                host=context.host,
            ),
            engine,
            files,
        )
        describe_conversion(conversion_row.row, outcome)


def describe_conversion(row: StepRow, outcome: ConversionOutcome) -> None:
    print(f"{row.name}:", flush=True)
    for converted in outcome.report.files:
        print(
            f"  wrote {converted.output}, {outcome.outputs.sizes_bytes[converted.output]} bytes, "
            f"{converted.kind.value}: {converted.node_count} nodes, "
            f"{converted.quantized_product_count} products in 8-bit blocks, "
            f"{', '.join(converted.kept_float32_operators) or 'no operator'} kept in float32, "
            f"{converted.below_min_positive_count} values raised to the minimum and "
            f"{converted.above_max_finite_count} lowered to the maximum in "
            f"{converted.clamped_initializer_count} initializers, "
            f"{converted.realigned_cast_count} Cast nodes realigned",
            flush=True,
        )
    print(f"  {describe_peaks(outcome.outputs.peaks)}", flush=True)


def run_named_step(
    recipe: JudgedRecipe, step_name: str, work_directory: WorkDirectory, files: WorkFiles
) -> None:
    if step_name == FLEURS_CHECK_STEP_NAME:
        check_fleurs(recipe, work_directory, files)
        return
    context = read_build_context(work_directory)
    if step_name == INSPECT_STEP_NAME:
        inspect_candidate(recipe, context, files)
        return
    if step_name == CONVERT_STEP_NAME:
        convert(recipe, context, files)
        return
    runs = [run for run in recipe.runs if run.step == step_name]
    if not runs:
        run_plain_step(recipe, find_runnable_row(recipe, step_name), context, files)
        return
    candidate_texts = read_candidate_texts(build_candidate_request(recipe, context), files)
    for run in runs:
        for language in run.languages:
            derived_rows = [
                derived_row
                for derived_row in recipe.derived_rows
                if derived_row.run == run.name and derived_row.language == language
            ]
            run_sharded_rows(recipe, run, derived_rows, candidate_texts, context, files)


def build(recipe: JudgedRecipe, work_directory: WorkDirectory, files: WorkFiles) -> None:
    for step_name in recipe.build_order:
        print(f"== {step_name}", flush=True)
        run_named_step(recipe, step_name, work_directory, files)
    print(f"== {VERIFY_COMMAND}", flush=True)
    verify(recipe, work_directory, files)


def build_verification_request(recipe: JudgedRecipe, context: BuildContext) -> VerificationRequest:
    return VerificationRequest(
        candidate=build_candidate_request(recipe, context),
        pins=recipe.pins,
        runs=recipe.runs,
        derived_rows=recipe.derived_rows,
        parities=recipe.parities,
        comparisons=recipe.comparisons,
    )


def verify(recipe: JudgedRecipe, work_directory: WorkDirectory, files: WorkFiles) -> None:
    context = read_build_context(work_directory)
    verified = verify_build(build_verification_request(recipe, context), files)
    for line in describe_verification(verified.verification):
        print(line, flush=True)
    for digest in verified.verification.candidate:
        print(
            f"verified {digest.name}, {digest.size_bytes} bytes, sha256 {digest.sha256}",
            flush=True,
        )
    print(f"wrote {verified.report_path}", flush=True)
    refuse_rejected_build(verified)


def read_images_requirements() -> tuple[DeclaredRequirement, ...]:
    manifest = tomllib.loads(
        (IMAGES_ENVIRONMENT_DIRECTORY / MANIFEST_FILE_NAME).read_text(encoding=TEXT_ENCODING)
    )
    project = manifest.get("project")
    if not isinstance(project, dict):
        raise ValueError(f"images/{MANIFEST_FILE_NAME} holds no project table")
    return read_declared_requirements(project.get("optional-dependencies"))


def read_card_template(recipe: JudgedRecipe) -> str | None:
    if recipe.card_template is None:
        return None
    return (CLONE_ROOT / recipe.card_template).read_text(encoding=TEXT_ENCODING)


def build_card_request(recipe: JudgedRecipe, context: BuildContext) -> CardRequest:
    return CardRequest(
        verification=build_verification_request(recipe, context),
        model=recipe.model,
        template=read_card_template(recipe),
        conversions=recipe.conversions,
        requirements=read_images_requirements(),
    )


def write_card(recipe: JudgedRecipe, work_directory: WorkDirectory, files: WorkFiles) -> None:
    context = read_build_context(work_directory)
    prepared = prepare_publication(
        build_card_request(recipe, context), GitTree(PurePosixPath(CLONE_ROOT)), files
    )
    outcome = write_model_card(prepared, context.work_root, files)
    print(
        f"{CARD_OUTCOME_WORDS[outcome]} {prepared.card_path}; review it before publishing",
        flush=True,
    )


def publish(
    recipe: JudgedRecipe, repository: str, work_directory: WorkDirectory, files: WorkFiles
) -> None:
    context = read_build_context(work_directory)
    published = publish_candidate(
        CandidatePublication(build_card_request(recipe, context), repository),
        GitTree(PurePosixPath(CLONE_ROOT)),
        TerminalPrompt(STANDARD_INPUT_DESCRIPTOR, sys.stdout),
        DockerEngine(),
        files,
    )
    report = published.report
    visibility = "private" if report.is_private else "public"
    print(
        f"published {report.repository} at the revision {report.revision}, on "
        f"{report.parent_revision}; the repository is {visibility}",
        flush=True,
    )
    print(
        "check it once read on the site: ./model-build.sh --work <dir> --recipe "
        f"{recipe.name} {CHECK_PUBLISHED_COMMAND} --repository {report.repository} "
        f"--revision {report.revision}",
        flush=True,
    )


def check_published_revision(
    recipe: JudgedRecipe,
    repository: str,
    revision: str,
    work_directory: WorkDirectory,
    files: WorkFiles,
) -> None:
    context = read_build_context(work_directory)
    downloader = CurlDownloader()
    checked_files = check_published(
        PublishedCheckRequest(
            repository=repository,
            revision=revision,
            candidate=build_candidate_request(recipe, context),
            model=recipe.model,
        ),
        downloader,
        downloader,
        files,
        files,
    )
    for checked_file in checked_files:
        print(
            f"published {checked_file.name}, {checked_file.size_bytes} bytes, sha256 "
            f"{checked_file.sha256}, the build's own bytes",
            flush=True,
        )
    print(f"pin {repository} at the revision {revision}", flush=True)


def describe_verification(verification: Verification) -> list[str]:
    return [
        *(describe_parity_measure(measure) for measure in verification.parities),
        *(describe_run_measure(measure) for measure in verification.runs),
        *(describe_comparison_measure(measure) for measure in verification.comparisons),
    ]


def describe_parity_measure(measure: ParityMeasure) -> str:
    verdict = "accepted" if measure.failure is None else "REFUSED"
    return (
        f"parity {measure.report.name}: {measure.report.maximum_absolute_difference:.3g} against "
        f"{measure.absolute_difference_tolerance:.3g}, {verdict}"
    )


def describe_run_measure(measure: RunMeasure) -> str:
    return (
        f"run {measure.run}, {measure.language}: {measure.utterance_count} utterances, word "
        f"error rate {compute_run_word_error_rate_percent(measure):.2f} % without its unknown "
        f"tokens, {compute_run_word_error_rate_with_unknown_tokens_percent(measure):.2f} % with "
        f"them; {measure.unknown_token_count} unknown tokens; "
        f"{measure.decoding_time_ms / 1000:.1f} s of decoding"
    )


def describe_comparison_measure(measure: ComparisonMeasure) -> str:
    paired = measure.paired
    verdict = "accepted" if not measure.failures else "REFUSED: " + "; ".join(measure.failures)
    return (
        f"{measure.run} against {measure.reference_run}, {measure.language}: identical token ids "
        f"{measure.identical_count}/{measure.compared_count} "
        f"({measure.identical_count / measure.compared_count:.1%}), difference of word error "
        f"rates {paired.difference_points:+.2f} points "
        f"[{paired.interval_low_points:+.2f}, {paired.interval_high_points:+.2f}], "
        f"{name_rule(measure.rule)}, {verdict}"
    )


def clean(
    recipe: JudgedRecipe, step_name: str | None, work_directory: WorkDirectory, files: WorkFiles
) -> None:
    clone_root = PurePosixPath(CLONE_ROOT)
    git_tree = GitTree(clone_root)
    commit = git_tree.find_head_commit()
    head_tree = git_tree.find_head_tree()
    host = read_host_identity()
    rows = list_runnable_rows(recipe) if step_name is None else list_rows_of_step(recipe, step_name)
    requests = [
        CleanRequest(
            row=row,
            recipe_name=recipe.name,
            commit=commit,
            work_root=PurePosixPath(work_directory.path),
            head_tree=head_tree,
            host=host,
        )
        for row in rows
    ]
    report_outcome = remove_verification_report(requests[0], files)
    print(f"{CLEAN_OUTCOME_WORDS[report_outcome]} the verification's report", flush=True)
    card_outcome = remove_model_card(requests[0], files)
    print(f"{CLEAN_OUTCOME_WORDS[card_outcome]} the model card", flush=True)
    for request in requests:
        outcome = clean_step(request, DockerEngine(), files)
        print(f"{CLEAN_OUTCOME_WORDS[outcome]} {request.row.name}", flush=True)
        if request.row.name == INSPECT_STEP_NAME:
            candidate_outcome = remove_candidate(request, files)
            print(f"{CLEAN_OUTCOME_WORDS[candidate_outcome]} the candidate", flush=True)


def lock(work_directory: WorkDirectory, files: WorkFiles) -> None:
    clone_root = PurePosixPath(CLONE_ROOT)
    images_lock_path = IMAGES_ENVIRONMENT_DIRECTORY / LOCK_FILE_NAME
    resolution = resolve_lock(
        LockRequest(
            manifest_text=(IMAGES_ENVIRONMENT_DIRECTORY / MANIFEST_FILE_NAME).read_text(
                encoding=TEXT_ENCODING
            ),
            committed_lock_text=read_optional_text(images_lock_path) or None,
            images_lock_path=PurePosixPath(images_lock_path),
            work_root=PurePosixPath(work_directory.path),
            head_tree=GitTree(clone_root).find_head_tree(),
            host=read_host_identity(),
        ),
        DockerEngine(),
        files,
        lambda line: print(line, flush=True),
    )
    print(describe_lock_resolution(resolution), flush=True)
    print(describe_peaks(resolution.peaks), flush=True)


def run_command(arguments: argparse.Namespace) -> None:
    files = WorkFiles()
    work_directory = open_requested_work_directory(arguments.work, files)
    if arguments.command == LOCK_STEP_NAME:
        lock(work_directory, files)
        return
    recipe = load_recipe(arguments.recipe)
    if arguments.command == FETCH_COMMAND:
        fetch(recipe, work_directory, files)
    elif arguments.command == IMAGES_COMMAND:
        build_container_images(recipe, work_directory, files)
    elif arguments.command == STEP_COMMAND:
        run_named_step(recipe, arguments.step, work_directory, files)
    elif arguments.command == CLEAN_STEP_NAME:
        clean(recipe, arguments.step, work_directory, files)
    elif arguments.command == VERIFY_COMMAND:
        verify(recipe, work_directory, files)
    elif arguments.command == BUILD_COMMAND:
        build(recipe, work_directory, files)
    elif arguments.command == CARD_COMMAND:
        write_card(recipe, work_directory, files)
    elif arguments.command == PUBLISH_COMMAND:
        publish(recipe, arguments.repository, work_directory, files)
    elif arguments.command == CHECK_PUBLISHED_COMMAND:
        check_published_revision(
            recipe, arguments.repository, arguments.revision, work_directory, files
        )


def main(arguments: Sequence[str]) -> None:
    parser = build_argument_parser()
    parsed_arguments = parser.parse_args(arguments)
    if parsed_arguments.recipe is None and parsed_arguments.command != LOCK_STEP_NAME:
        parser.error(f"the command {parsed_arguments.command} needs --recipe <name>")
    if parsed_arguments.command == PUBLISH_COMMAND and not os.isatty(STANDARD_INPUT_DESCRIPTOR):
        parser.error(
            f"{PUBLISH_COMMAND} reads the token at a terminal, and the standard input is none: "
            "run it yourself, in a terminal"
        )
    try:
        run_command(parsed_arguments)
    except COMMAND_ERRORS as error:
        print(f"model-build: {error}", file=sys.stderr)
        raise SystemExit(FAILURE_EXIT_STATUS) from error


if __name__ == "__main__":
    main(sys.argv[1:])
