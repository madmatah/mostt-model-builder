import re
from collections.abc import Callable, Mapping
from dataclasses import replace
from functools import partial
from pathlib import PurePosixPath

import pytest

from mostt_model_builder.domain.bench.hypotheses import Hypothesis
from mostt_model_builder.domain.containers.base_images import BASE_PYTHON, ENVIRONMENT_PYTHON
from mostt_model_builder.domain.containers.container_steps import (
    FLEURS_CHECK_ROW,
    PEAKS_OUTPUT,
    DeclaredOutput,
    EarlierOutput,
    SourceReference,
    StepLimits,
    StepRow,
)
from mostt_model_builder.domain.contract.recipe_contract import (
    CANDIDATE_MOUNT,
    REFERENCE_VARIANT,
    CandidateFile,
    DeviceTemplate,
    EarlierOutputMount,
    HostContentOrigin,
    JudgedRecipe,
    Parity,
    RecipeContract,
    RecipeContractError,
    RecipeFacts,
    RecipeImage,
    RecipeStep,
    Run,
    SourceMount,
    StepOutputOrigin,
    find_candidate_file_max_size_bytes,
    find_runnable_row,
    judge_recipe,
    locate_entry_module_file,
)
from mostt_model_builder.domain.formats.file_formats import FileFormat
from mostt_model_builder.domain.sources.source_pins import RepositoryType, SourcePin

RECIPE_NAME = "family"
REVISION = "0123456789abcdef0123456789abcdef01234567"
SHA256 = "0" * 64
WEIGHTS_REPOSITORY = "maker/weights"
ARCHIVE_REPOSITORY = "maker/archive"
BENCH_PIN = SourcePin(RepositoryType.DATASET, "google/fleurs", REVISION, "data/test.tsv", 1, SHA256)
WEIGHTS_PIN = SourcePin(
    RepositoryType.MODEL, WEIGHTS_REPOSITORY, REVISION, "model.safetensors", 1, SHA256
)
NESTED_PIN = SourcePin(
    RepositoryType.MODEL, WEIGHTS_REPOSITORY, REVISION, "configs/model.json", 1, SHA256
)
ARCHIVE_PIN = SourcePin(
    RepositoryType.MODEL, ARCHIVE_REPOSITORY, REVISION, "model.nemo", 1, SHA256, True
)
DESCRIBE_MODULE = "mostt_recipe_family.steps.description"
TRANSCRIBE_MODULE = "mostt_recipe_family.steps.transcription"
RECIPE_MODULE = "mostt_recipe_family.recipe"
FLEURS_CHECK_MODULE = FLEURS_CHECK_ROW.entry_module
LIMITS = StepLimits(memory_mib=1024, process_count=64, cpu_count=2, temporary_directory_mib=64)
NAMES_OUTPUT = DeclaredOutput("names", 1024)
EXPORT_IMAGE = RecipeImage("export", "cpu", "family-export", reads_code_running_sources=True)
TRANSCRIBE_IMAGE = RecipeImage("transcribe", "gpu", "family-transcribe")
DESCRIBE_STEP = RecipeStep(
    name="describe",
    image="export",
    entry_module=DESCRIBE_MODULE,
    mounts=(
        SourceMount(WEIGHTS_REPOSITORY, "model.safetensors"),
        SourceMount(ARCHIVE_REPOSITORY, "model.nemo"),
    ),
    outputs=(NAMES_OUTPUT, PEAKS_OUTPUT),
    limits=LIMITS,
)
TRANSCRIBE_STEP = RecipeStep(
    name="transcribe",
    image="transcribe",
    entry_module=TRANSCRIBE_MODULE,
    mounts=(EarlierOutputMount("describe", "names"), SourceMount("google/fleurs", "data/test.tsv")),
    outputs=(PEAKS_OUTPUT,),
    limits=LIMITS,
    uses_gpu=True,
)
NAMES_FILE = CandidateFile("names.txt", "text", ("float32",), StepOutputOrigin("describe", "names"))
CONFIGURATION = b"{}"
CONFIGURATION_FILE = CandidateFile(
    "config.json", "text", ("float32", "float16"), HostContentOrigin(CONFIGURATION)
)
NAMES_PARITY = Parity("names-against-reference", "describe", "names", 1e-3)
CONTRACT = RecipeContract(
    name=RECIPE_NAME,
    images=(EXPORT_IMAGE, TRANSCRIBE_IMAGE),
    steps=(DESCRIBE_STEP, TRANSCRIBE_STEP),
    candidate=(NAMES_FILE, CONFIGURATION_FILE),
    parities=(NAMES_PARITY,),
)
FACTS = RecipeFacts(
    core_pins=(BENCH_PIN,),
    recipe_pins=(WEIGHTS_PIN, NESTED_PIN, ARCHIVE_PIN),
    present_entry_modules=frozenset(
        {DESCRIBE_MODULE, TRANSCRIBE_MODULE, RECIPE_MODULE, FLEURS_CHECK_MODULE}
    ),
    host_cpu_count=20,
)


# A reader takes a shard's lines and the candidate's texts; the contract only checks that a run
# has one, so this one reads nothing.
def read_no_lines(
    lines: bytes,  # noqa: ARG001
    candidate_texts: Mapping[str, bytes],  # noqa: ARG001
) -> tuple[Hypothesis, ...]:
    return ()


LINES_OUTPUT = DeclaredOutput("lines.jsonl", 2**20)
PLACEMENT_OUTPUT = DeclaredOutput("placement.json", 2**20)
SHARD_LIMITS = replace(LIMITS, cpu_count=1)
TRANSCRIBE_CPU_IMAGE = RecipeImage("transcribe-cpu", "cpu", "family-transcribe-cpu")
SHARDED_STEP = RecipeStep(
    name="transcribe-variants",
    image=None,
    entry_module=TRANSCRIBE_MODULE,
    mounts=(CANDIDATE_MOUNT, SourceMount("google/fleurs", "data/test.tsv")),
    outputs=(PEAKS_OUTPUT,),
    limits=None,
    devices=(
        DeviceTemplate("cpu", "transcribe-cpu", SHARD_LIMITS, (LINES_OUTPUT,)),
        DeviceTemplate("gpu", "transcribe", LIMITS, (LINES_OUTPUT, PLACEMENT_OUTPUT)),
    ),
)
ENCODER_FILE = CandidateFile(
    "encoder.onnx", "onnx", ("float32",), StepOutputOrigin("describe", "names")
)
ENCODER_DATA_FILE = CandidateFile(
    "encoder.onnx.data",
    "onnx-external-data",
    ("float32",),
    HostContentOrigin(b"\x00"),
    external_data_of="encoder.onnx",
)
CPU_RUN = Run(
    "float32-cpu", "transcribe-variants", "cpu", "float32", ("fr_fr", "en_us"), 3, read_no_lines
)
GPU_RUN = Run("float32-gpu", "transcribe-variants", "gpu", "float32", ("fr_fr",), 1, read_no_lines)
RUN_CONTRACT = replace(
    CONTRACT,
    images=(EXPORT_IMAGE, TRANSCRIBE_IMAGE, TRANSCRIBE_CPU_IMAGE),
    steps=(DESCRIBE_STEP, SHARDED_STEP),
    candidate=(NAMES_FILE, ENCODER_FILE, ENCODER_DATA_FILE),
    runs=(CPU_RUN, GPU_RUN),
    conversion_input=StepOutputOrigin("describe", "names"),
    build_order=("fleurs-check", "describe", "inspect", "transcribe-variants"),
)
RUN_FACTS = replace(FACTS, present_entry_modules=FACTS.present_entry_modules | {TRANSCRIBE_MODULE})


def change_run(run_index: int, **fields: object) -> RecipeContract:
    runs = list(RUN_CONTRACT.runs)
    runs[run_index] = replace(runs[run_index], **fields)  # type: ignore[arg-type]  # the test feeds a value outside the contract's types, on purpose
    return replace(RUN_CONTRACT, runs=tuple(runs))


def change_sharded_step(**fields: object) -> RecipeContract:
    return replace(
        RUN_CONTRACT,
        steps=(DESCRIBE_STEP, replace(SHARDED_STEP, **fields)),  # type: ignore[arg-type]  # the test feeds a value outside the contract's types, on purpose
    )


def change_device_template(device_index: int, **fields: object) -> RecipeContract:
    devices = list(SHARDED_STEP.devices)
    devices[device_index] = replace(devices[device_index], **fields)  # type: ignore[arg-type]  # the test feeds a value outside the contract's types, on purpose
    return change_sharded_step(devices=tuple(devices))


def change_external_data(**fields: object) -> RecipeContract:
    return replace(
        RUN_CONTRACT,
        candidate=(NAMES_FILE, ENCODER_FILE, replace(ENCODER_DATA_FILE, **fields)),  # type: ignore[arg-type]  # the test feeds a value outside the contract's types, on purpose
    )


def change_step(step_index: int, **fields: object) -> RecipeContract:
    steps = list(CONTRACT.steps)
    steps[step_index] = replace(steps[step_index], **fields)  # type: ignore[arg-type]  # the test feeds a value outside the contract's types, on purpose
    return replace(CONTRACT, steps=tuple(steps))


def change_image(image_index: int, **fields: object) -> RecipeContract:
    images = list(CONTRACT.images)
    images[image_index] = replace(images[image_index], **fields)  # type: ignore[arg-type]  # the test feeds a value outside the contract's types, on purpose
    return replace(CONTRACT, images=tuple(images))


def change_candidate(**fields: object) -> RecipeContract:
    return replace(CONTRACT, candidate=(replace(NAMES_FILE, **fields),))  # type: ignore[arg-type]  # the test feeds a value outside the contract's types, on purpose


def change_parity(**fields: object) -> RecipeContract:
    return replace(CONTRACT, parities=(replace(NAMES_PARITY, **fields),))  # type: ignore[arg-type]  # the test feeds a value outside the contract's types, on purpose


REFUSED_CONTRACTS = {
    "a recipe name outside the grammar": (
        replace(CONTRACT, name="Family"),
        "the recipe 'Family' is not named under",
    ),
    "an image name outside the grammar": (
        change_image(0, name="Export"),
        "its image 'Export' is not named under",
    ),
    "a recipe image named like an image of the core": (
        change_image(0, name="bench"),
        "its image 'bench' bears the name of an image of the core",
    ),
    "an image declared twice": (
        replace(CONTRACT, images=(EXPORT_IMAGE, EXPORT_IMAGE)),
        "its image 'export' is declared twice",
    ),
    "an image without its device": (
        change_image(0, device=""),
        "its image 'export' has no device among cpu, gpu",
    ),
    "an image on an unknown device": (
        change_image(0, device="tpu"),
        "its image 'export' has no device among cpu, gpu",
    ),
    "an image that installs another image's extra": (
        change_image(0, extra="family-transcribe"),
        "installs the extra 'family-transcribe', not family-export",
    ),
    "two images that read the sources that run code": (
        change_image(1, reads_code_running_sources=True),
        "the images export, transcribe read the sources that run code when they load",
    ),
    "a step name outside the grammar": (
        change_step(0, name="Describe"),
        "its step 'Describe' is not named under",
    ),
    "a step named like a step of the core": (
        change_step(0, name="fleurs-check"),
        "its step 'fleurs-check' bears the name of a step of the core",
    ),
    "a step declared twice": (
        replace(CONTRACT, steps=(DESCRIBE_STEP, DESCRIBE_STEP)),
        "its step 'describe' is declared twice",
    ),
    "an unknown image": (
        change_step(0, image="convert"),
        "runs in the image 'convert', which neither the recipe nor the core declares",
    ),
    "an entry module that is not a dotted name": (
        change_step(0, entry_module="mostt_recipe_family.steps.de-scription"),
        "its entry module 'mostt_recipe_family.steps.de-scription' is not a dotted module name",
    ),
    "an entry module whose file does not exist": (
        change_step(0, entry_module="mostt_recipe_family.steps.missing"),
        "has no regular file at recipes/family/src/mostt_recipe_family/steps/missing.py",
    ),
    "an entry module of the core in a recipe image": (
        change_step(0, entry_module=FLEURS_CHECK_MODULE),
        "lies outside mostt_recipe_family.steps, the steps of the image",
    ),
    "an entry module of the recipe outside its steps": (
        change_step(0, entry_module=RECIPE_MODULE),
        "its entry module mostt_recipe_family.recipe lies outside mostt_recipe_family.steps, "
        "the steps of the image",
    ),
    "an entry module of the recipe in a core image": (
        change_step(0, image="bench", mounts=()),
        "lies outside mostt_model_builder.steps, the steps of the image",
    ),
    "an undeclared source": (
        change_step(0, mounts=(SourceMount(WEIGHTS_REPOSITORY, "other.safetensors"),)),
        "binds 'other.safetensors' of 'maker/weights', which no source pins",
    ),
    "a source bound as a directory": (
        change_step(0, mounts=(SourceMount(WEIGHTS_REPOSITORY, "configs"),)),
        "binds the directory 'configs' of maker/weights, where only a pinned file may be bound",
    ),
    "a source bound twice": (
        change_step(
            0,
            mounts=(
                SourceMount(WEIGHTS_REPOSITORY, "model.safetensors"),
                SourceMount(WEIGHTS_REPOSITORY, "model.safetensors"),
            ),
        ),
        "binds model.safetensors of maker/weights twice",
    ),
    "the source that runs code bound in another image": (
        change_step(1, mounts=(SourceMount(ARCHIVE_REPOSITORY, "model.nemo"),)),
        "binds model.nemo of maker/archive, which runs code when it loads, in the image "
        "mostt-model-builder-family-transcribe",
    ),
    "the source that runs code bound in a core image": (
        replace(
            CONTRACT,
            steps=(
                replace(
                    DESCRIBE_STEP,
                    image="bench",
                    entry_module=FLEURS_CHECK_MODULE,
                    mounts=(SourceMount(ARCHIVE_REPOSITORY, "model.nemo"),),
                ),
            ),
            candidate=(),
        ),
        "in the image mostt-model-builder-core-bench, which the recipe does not declare",
    ),
    "an output of a step that is not earlier": (
        replace(CONTRACT, steps=(TRANSCRIBE_STEP, DESCRIBE_STEP)),
        "binds an output of 'describe', which is not an earlier step of the recipe",
    ),
    "the outputs of an earlier step bound as a directory": (
        change_step(1, mounts=(EarlierOutputMount("describe", ""),)),
        "binds the outputs of describe as a directory, where only a declared output file",
    ),
    "an output that the earlier step does not declare": (
        change_step(1, mounts=(EarlierOutputMount("describe", "other"),)),
        "binds 'other' of describe, which that step does not declare",
    ),
    "an output without its size bound": (
        change_step(0, outputs=(DeclaredOutput("names", 0), PEAKS_OUTPUT)),
        "the size bound of names in bytes, 0, lies outside 1 to 8589934592",
    ),
    "an output beyond the ceiling": (
        change_step(0, outputs=(DeclaredOutput("names", 8 * 2**30 + 1), PEAKS_OUTPUT)),
        "the size bound of names in bytes, 8589934593, lies outside 1 to 8589934592",
    ),
    "outputs without the peaks": (
        change_step(0, outputs=(NAMES_OUTPUT,)),
        "its outputs lack peaks.json",
    ),
    "memory beyond its ceiling": (
        change_step(0, limits=replace(LIMITS, memory_mib=57_345)),
        "its memory in MiB, 57345, lies outside 6 to 57344",
    ),
    "more cpus than the host has": (
        change_step(0, limits=replace(LIMITS, cpu_count=21)),
        "its CPUs, 21, lies outside 1 to 20",
    ),
    "the gpu in a cpu image": (
        change_step(0, uses_gpu=True),
        "uses the GPU in the cpu image",
    ),
    "an unknown format": (
        change_candidate(file_format="gguf"),
        "its candidate file 'names.txt' has the format 'gguf', not one of onnx, text, "
        "onnx-external-data",
    ),
    "a candidate name outside the file-name grammar": (
        change_candidate(name="../names.txt"),
        "its candidate file '../names.txt' is not a file name",
    ),
    "a candidate name with a refused suffix": (
        change_candidate(name="weights.PT"),
        "its candidate file 'weights.PT' has the refused suffix .pt",
    ),
    "a candidate name with an inner refused suffix": (
        change_candidate(name="weights.pkl.txt"),
        "its candidate file 'weights.pkl.txt' has the refused suffix .pkl",
    ),
    "a candidate file declared twice": (
        replace(CONTRACT, candidate=(NAMES_FILE, NAMES_FILE)),
        "its candidate file 'names.txt' is declared twice",
    ),
    "a candidate file that serves no variant": (
        change_candidate(variants=()),
        "its candidate file 'names.txt' serves no variant",
    ),
    "a variant outside the grammar": (
        change_candidate(variants=("Float32",)),
        "serves the variant 'Float32', not named under",
    ),
    "a variant named twice": (
        change_candidate(variants=("float32", "float32")),
        "its candidate file 'names.txt' names a variant twice",
    ),
    "a candidate file from an undeclared output": (
        change_candidate(origin=StepOutputOrigin("describe", "other")),
        "comes from 'other' of 'describe', which no step of the recipe declares",
    ),
    "an empty content that the host writes": (
        change_candidate(origin=HostContentOrigin(b"")),
        "has a content of 0 bytes, outside 1 to 1048576",
    ),
    "a parity name outside the grammar": (
        change_parity(name="Names"),
        "its parity 'Names' is not named under",
    ),
    "a parity declared twice": (
        replace(CONTRACT, parities=(NAMES_PARITY, NAMES_PARITY)),
        "its parity 'names-against-reference' is declared twice",
    ),
    "a parity report that its step does not declare": (
        change_parity(output="parity.json"),
        "reads its report from 'parity.json' of 'describe', which no step of the recipe declares",
    ),
    "a parity report of a step the recipe does not declare": (
        change_parity(step="export"),
        "reads its report from 'names' of 'export', which no step of the recipe declares",
    ),
    "a parity report read from the peaks": (
        change_parity(output="peaks.json"),
        "reads its report from peaks.json, the peaks of its step",
    ),
    "a tolerance of zero": (
        change_parity(absolute_difference_tolerance=0.0),
        "has the tolerance 0.0, not a finite number above 0",
    ),
    "a negative tolerance": (
        change_parity(absolute_difference_tolerance=-5e-5),
        "has the tolerance -5e-05, not a finite number above 0",
    ),
    "a tolerance that is not a number": (
        change_parity(absolute_difference_tolerance=float("nan")),
        "has the tolerance nan, not a finite number above 0",
    ),
    "an infinite tolerance": (
        change_parity(absolute_difference_tolerance=float("inf")),
        "has the tolerance inf, not a finite number above 0",
    ),
    "a boolean tolerance": (
        change_parity(absolute_difference_tolerance=True),
        "has the tolerance True, not a finite number above 0",
    ),
    "a step named like a directory of its run": (
        change_step(0, name="candidate"),
        "its step 'candidate' bears the name of a step of the core or of a directory of its run",
    ),
    "a run outside the grammar": (
        change_run(0, name="Float32-cpu"),
        "its run 'Float32-cpu' is not named under",
    ),
    "a run declared twice": (
        replace(RUN_CONTRACT, runs=(CPU_RUN, CPU_RUN)),
        "its run 'float32-cpu' is declared twice",
    ),
    "a run without a producing step": (
        change_run(0, step="transcribe-other"),
        "its run 'float32-cpu' comes from 'transcribe-other', which is not a step of the recipe",
    ),
    "a run of a step without a template": (
        change_run(0, step="describe"),
        "its step 'describe' produces the run float32-cpu, and has no template of its rows",
    ),
    "a run without a reader": (
        change_run(0, reader=None),
        "its run 'float32-cpu' has no reader of its lines",
    ),
    "a run on an unknown device": (
        change_run(0, device="tpu"),
        "its run 'float32-cpu' runs on 'tpu', neither cpu nor gpu",
    ),
    "a run on a device that its template does not give": (
        change_sharded_step(devices=(SHARDED_STEP.devices[0],)),
        "its run float32-gpu runs on the gpu, which its template does not give",
    ),
    "a run that reads a variant the candidate lacks": (
        change_run(0, variant="float16"),
        "its run 'float32-cpu' reads the variant 'float16', which no file of the candidate serves",
    ),
    "two runs whose rows share a name": (
        replace(
            RUN_CONTRACT,
            steps=(
                DESCRIBE_STEP,
                replace(SHARDED_STEP, name="a"),
                replace(SHARDED_STEP, name="a-b"),
            ),
            runs=(replace(CPU_RUN, name="b-c", step="a"), replace(CPU_RUN, name="c", step="a-b")),
        ),
        "its step 'a-b' derives the row a-b-c-0, which another row already holds",
    ),
    "a run that reads no language": (
        change_run(0, languages=()),
        "its run 'float32-cpu' reads no language",
    ),
    "a run that reads a language outside the bench": (
        change_run(0, languages=("de_de",)),
        "its run 'float32-cpu' reads 'de_de', which is not a language of the bench",
    ),
    "a run that names a language twice": (
        change_run(0, languages=("fr_fr", "fr_fr")),
        "its run 'float32-cpu' names a language twice",
    ),
    "a run of no shard": (
        change_run(0, shard_count=0),
        "its run 'float32-cpu' has 0 shards, outside 1 to 16",
    ),
    "a run of 17 shards": (
        change_run(0, shard_count=17),
        "its run 'float32-cpu' has 17 shards, outside 1 to 16",
    ),
    "a template whose device does not match its image": (
        change_device_template(0, image="transcribe"),
        "its template for 'cpu' runs in the gpu image mostt-model-builder-family-transcribe",
    ),
    "a template on an unknown device": (
        change_device_template(0, device="tpu"),
        "its template for 'tpu' names no device among cpu, gpu",
    ),
    "a template whose shard writes no lines": (
        change_device_template(0, outputs=()),
        "its template for 'cpu' writes no lines.jsonl, the lines of a run",
    ),
    "a template of the gpu whose shard writes no placement": (
        change_device_template(1, outputs=(LINES_OUTPUT,)),
        "its template for 'gpu' writes no placement.json, the placement of its nodes on the GPU",
    ),
    "a template of the cpu whose shard writes a placement": (
        change_device_template(0, outputs=(LINES_OUTPUT, PLACEMENT_OUTPUT)),
        "its template for 'cpu' writes placement.json, which only a run on the GPU writes",
    ),
    "a step with a template and its own image": (
        change_sharded_step(image="transcribe"),
        "has a template of its rows, which gives each device its image, its limits and the GPU",
    ),
    "a step with a template and no run": (
        replace(RUN_CONTRACT, runs=()),
        "its step 'transcribe-variants' has a template of its rows, and no run",
    ),
    "a step with neither an image nor a template": (
        change_step(0, image=None),
        "its step 'describe' has neither an image and limits nor a template",
    ),
    "external data that names no onnx file": (
        change_external_data(external_data_of=None),
        "its candidate file encoder.onnx.data holds external data, and names no onnx file",
    ),
    "external data of a file that is not onnx": (
        change_external_data(external_data_of="names.txt"),
        "holds the tensors of 'names.txt', which is not an onnx file of the candidate",
    ),
    "external data whose onnx file serves another variant": (
        change_external_data(variants=("float16",)),
        "serves float16, and its onnx file encoder.onnx serves float32",
    ),
    "a file of another format that names an onnx file": (
        replace(RUN_CONTRACT, candidate=(replace(NAMES_FILE, external_data_of="encoder.onnx"),)),
        "is text, and names the onnx file whose tensors it holds",
    ),
    "conversions that read no onnx file of the candidate": (
        replace(RUN_CONTRACT, conversion_input=StepOutputOrigin("describe", "peaks.json")),
        "its conversions read 'peaks.json' of 'describe', which no onnx file of the candidate",
    ),
    "a build that runs an unknown step": (
        replace(RUN_CONTRACT, build_order=("fleurs-check", "publish")),
        "its build runs 'publish', which neither the core nor the recipe builds with",
    ),
    "a build that runs a step twice": (
        replace(RUN_CONTRACT, build_order=("describe", "describe")),
        "its build runs a step twice",
    ),
}


class TestTheRecipeContract:
    class TestTheJudgeRecipeFunction:
        class TestWhenTheContractDeclaresRuns:
            @pytest.fixture
            def judged_recipe(self) -> JudgedRecipe:
                return judge_recipe(RUN_CONTRACT, RUN_FACTS)

            def test_should_derive_a_row_per_shard_named_after_its_step_run_and_shard(
                self, judged_recipe: JudgedRecipe
            ) -> None:
                assert [derived_row.row.name for derived_row in judged_recipe.derived_rows] == [
                    "transcribe-variants-float32-cpu-0",
                    "transcribe-variants-float32-cpu-1",
                    "transcribe-variants-float32-cpu-2",
                    "transcribe-variants-float32-gpu-0",
                ]

            def test_should_put_the_derived_rows_among_the_rows_of_the_recipe(
                self, judged_recipe: JudgedRecipe
            ) -> None:
                assert [row.name for row in judged_recipe.rows][1:] == [
                    derived_row.row.name for derived_row in judged_recipe.derived_rows
                ]

            def test_should_run_each_row_in_the_image_of_its_device(
                self, judged_recipe: JudgedRecipe
            ) -> None:
                assert {
                    (derived_row.request.device.value, derived_row.row.image.name)
                    for derived_row in judged_recipe.derived_rows
                } == {("cpu", "transcribe-cpu"), ("gpu", "transcribe")}

            def test_should_give_the_gpu_to_the_rows_of_the_gpu_run_alone(
                self, judged_recipe: JudgedRecipe
            ) -> None:
                assert [
                    derived_row.row.name
                    for derived_row in judged_recipe.derived_rows
                    if derived_row.row.uses_gpu
                ] == ["transcribe-variants-float32-gpu-0"]

            def test_should_bind_the_candidate_in_every_row_of_the_step(
                self, judged_recipe: JudgedRecipe
            ) -> None:
                assert {
                    derived_row.row.binds_candidate for derived_row in judged_recipe.derived_rows
                } == {True}

            def test_should_keep_the_runs_with_their_readers(
                self, judged_recipe: JudgedRecipe
            ) -> None:
                assert [(run.name, run.reader) for run in judged_recipe.runs] == [
                    ("float32-cpu", read_no_lines),
                    ("float32-gpu", read_no_lines),
                ]

            def test_should_keep_the_external_data_with_the_onnx_file_it_serves(
                self, judged_recipe: JudgedRecipe
            ) -> None:
                assert [
                    (candidate_file.name, candidate_file.external_data_of)
                    for candidate_file in judged_recipe.candidate
                    if candidate_file.external_data_of is not None
                ] == [("encoder.onnx.data", "encoder.onnx")]

            def test_should_keep_the_order_of_the_build(self, judged_recipe: JudgedRecipe) -> None:
                assert judged_recipe.build_order == (
                    "fleurs-check",
                    "describe",
                    "inspect",
                    "transcribe-variants",
                )

        class TestWhenARunReadsTheReference:
            @pytest.fixture
            def judged_recipe(self) -> JudgedRecipe:
                return judge_recipe(change_run(0, variant=REFERENCE_VARIANT), RUN_FACTS)

            def test_should_accept_it_since_a_reference_framework_reads_no_variant_of_the_candidate(
                self, judged_recipe: JudgedRecipe
            ) -> None:
                assert [run.variant for run in judged_recipe.runs] == [
                    REFERENCE_VARIANT,
                    "float32",
                ]

        class TestWhenTheContractKeepsWithinItsBounds:
            @pytest.fixture
            def judged_recipe(self) -> JudgedRecipe:
                return judge_recipe(CONTRACT, FACTS)

            def test_should_tag_the_recipes_images_after_the_recipe(
                self, judged_recipe: JudgedRecipe
            ) -> None:
                assert [image.tag for image in judged_recipe.images] == [
                    "mostt-model-builder-family-export",
                    "mostt-model-builder-family-transcribe",
                ]

            def test_should_run_an_image_with_an_extra_on_its_environments_python(
                self, judged_recipe: JudgedRecipe
            ) -> None:
                assert {image.python for image in judged_recipe.images} == {ENVIRONMENT_PYTHON}

            def test_should_place_each_containerfile_under_the_recipes_images(
                self, judged_recipe: JudgedRecipe
            ) -> None:
                assert [image.containerfile for image in judged_recipe.images] == [
                    PurePosixPath("recipes/family/images/export/Containerfile"),
                    PurePosixPath("recipes/family/images/transcribe/Containerfile"),
                ]

            def test_should_keep_the_steps_in_their_order(
                self, judged_recipe: JudgedRecipe
            ) -> None:
                assert [row.name for row in judged_recipe.rows] == ["describe", "transcribe"]

            def test_should_bind_each_source_by_its_pin(self, judged_recipe: JudgedRecipe) -> None:
                assert judged_recipe.rows[0].sources == (
                    SourceReference(WEIGHTS_REPOSITORY, "model.safetensors"),
                    SourceReference(ARCHIVE_REPOSITORY, "model.nemo"),
                )

            def test_should_bind_an_earlier_output_with_every_name_its_step_declares(
                self, judged_recipe: JudgedRecipe
            ) -> None:
                assert judged_recipe.rows[1].earlier_outputs == (
                    EarlierOutput("describe", NAMES_OUTPUT, ("names", "peaks.json")),
                )

            def test_should_give_no_row_the_network(self, judged_recipe: JudgedRecipe) -> None:
                assert [row.name for row in judged_recipe.rows if row.has_network] == []

            def test_should_hold_the_cores_pins_and_the_recipes(
                self, judged_recipe: JudgedRecipe
            ) -> None:
                assert judged_recipe.pins == (BENCH_PIN, WEIGHTS_PIN, NESTED_PIN, ARCHIVE_PIN)

            def test_should_keep_each_parity_with_its_tolerance(
                self, judged_recipe: JudgedRecipe
            ) -> None:
                assert judged_recipe.parities == (NAMES_PARITY,)

            def test_should_read_the_format_of_each_candidate_file(
                self, judged_recipe: JudgedRecipe
            ) -> None:
                assert [file.file_format for file in judged_recipe.candidate] == [
                    FileFormat.TEXT,
                    FileFormat.TEXT,
                ]

        class TestWhenAnImageHasNoExtra:
            @pytest.fixture
            def judged_recipe(self) -> JudgedRecipe:
                return judge_recipe(change_image(1, extra=None), FACTS)

            def test_should_run_it_on_the_bases_python_since_it_has_no_environment(
                self, judged_recipe: JudgedRecipe
            ) -> None:
                assert judged_recipe.images[1].python == BASE_PYTHON

        class TestWhenAStepRunsInACoreImage:
            @pytest.fixture
            def judged_recipe(self) -> JudgedRecipe:
                return judge_recipe(
                    replace(
                        CONTRACT,
                        steps=(
                            replace(
                                DESCRIBE_STEP,
                                name="bench-probe",
                                image="bench",
                                entry_module=FLEURS_CHECK_MODULE,
                                mounts=(),
                            ),
                        ),
                        candidate=(),
                        parities=(),
                    ),
                    FACTS,
                )

            def test_should_accept_an_entry_module_of_the_cores_steps(
                self, judged_recipe: JudgedRecipe
            ) -> None:
                assert judged_recipe.rows[0].image.owner == "core"

        class TestWhenTheRecipePinsAFileOfTheCore:
            @pytest.fixture
            def judge_act(self) -> Callable[[], JudgedRecipe]:
                return partial(judge_recipe, CONTRACT, replace(FACTS, recipe_pins=(BENCH_PIN,)))

            def test_should_refuse_it_since_one_file_has_one_pin(
                self, judge_act: Callable[[], JudgedRecipe]
            ) -> None:
                with pytest.raises(
                    RecipeContractError,
                    match=re.escape("its sources and the core's: data/test.tsv of google/fleurs"),
                ):
                    judge_act()

        @pytest.mark.parametrize(
            ("contract", "reason"),
            list(REFUSED_CONTRACTS.values()),
            ids=list(REFUSED_CONTRACTS.keys()),
        )
        class TestWhenTheContractBreaksARule:
            @pytest.fixture
            def judge_act(self, contract: RecipeContract) -> Callable[[], JudgedRecipe]:
                return partial(judge_recipe, contract, FACTS)

            def test_should_refuse_it_and_say_why(
                self, judge_act: Callable[[], JudgedRecipe], reason: str
            ) -> None:
                with pytest.raises(RecipeContractError, match=re.escape(reason)):
                    judge_act()

    class TestTheLocateEntryModuleFileFunction:
        @pytest.mark.parametrize(
            ("entry_module", "module_file"),
            [
                pytest.param(
                    DESCRIBE_MODULE,
                    PurePosixPath("recipes/family/src/mostt_recipe_family/steps/description.py"),
                    id="a module of the recipe, under the recipe's sources",
                ),
                pytest.param(
                    FLEURS_CHECK_MODULE,
                    PurePosixPath("src/mostt_model_builder/steps/fleurs_check.py"),
                    id="a module of the core, under the core's sources",
                ),
                pytest.param(
                    "mostt_recipe_other.steps.description",
                    None,
                    id="a module of another recipe, nowhere",
                ),
                pytest.param("os.path", None, id="a module of the standard library, nowhere"),
                pytest.param("mostt_recipe_family", None, id="a package alone, nowhere"),
            ],
        )
        class TestWhenGivenAnEntryModule:
            @pytest.fixture
            def located_file(self, entry_module: str) -> PurePosixPath | None:
                return locate_entry_module_file(entry_module, RECIPE_NAME)

            def test_should_place_its_file_lexically_without_importing_it(
                self, located_file: PurePosixPath | None, module_file: PurePosixPath | None
            ) -> None:
                assert located_file == module_file

    class TestTheFindRunnableRowFunction:
        @pytest.fixture
        def judged_recipe(self) -> JudgedRecipe:
            return judge_recipe(CONTRACT, FACTS)

        class TestWhenTheStepIsTheRecipes:
            @pytest.fixture
            def row(self, judged_recipe: JudgedRecipe) -> StepRow:
                return find_runnable_row(judged_recipe, "transcribe")

            def test_should_find_its_row(self, row: StepRow) -> None:
                assert row.entry_module == TRANSCRIBE_MODULE

        class TestWhenTheStepIsTheCores:
            @pytest.fixture
            def row(self, judged_recipe: JudgedRecipe) -> StepRow:
                return find_runnable_row(judged_recipe, "fleurs-check")

            def test_should_find_its_row(self, row: StepRow) -> None:
                assert row == FLEURS_CHECK_ROW

        class TestWhenTheStepIsTheCleaning:
            @pytest.fixture
            def find_act(self, judged_recipe: JudgedRecipe) -> Callable[[], StepRow]:
                return partial(find_runnable_row, judged_recipe, "clean")

            def test_should_refuse_it_since_clean_has_its_own_command(
                self, find_act: Callable[[], StepRow]
            ) -> None:
                with pytest.raises(
                    RecipeContractError,
                    match="neither the core nor the recipe family has a step named 'clean'",
                ):
                    find_act()

    class TestTheFindCandidateFileMaxSizeBytesFunction:
        @pytest.fixture
        def judged_recipe(self) -> JudgedRecipe:
            return judge_recipe(CONTRACT, FACTS)

        class TestWhenTheFileIsCopiedFromTheOutputOfAStep:
            @pytest.fixture
            def max_size_bytes(self, judged_recipe: JudgedRecipe) -> int:
                return find_candidate_file_max_size_bytes(NAMES_FILE.origin, judged_recipe.rows)

            def test_should_give_the_bound_its_step_declares(self, max_size_bytes: int) -> None:
                assert max_size_bytes == NAMES_OUTPUT.max_size_bytes

        class TestWhenTheFileHoldsContentOfTheHost:
            @pytest.fixture
            def max_size_bytes(self, judged_recipe: JudgedRecipe) -> int:
                return find_candidate_file_max_size_bytes(
                    CONFIGURATION_FILE.origin, judged_recipe.rows
                )

            def test_should_give_the_length_of_that_content(self, max_size_bytes: int) -> None:
                assert max_size_bytes == len(CONFIGURATION)
