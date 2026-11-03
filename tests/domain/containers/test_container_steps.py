import re
from collections.abc import Callable
from dataclasses import replace
from functools import partial
from pathlib import PurePosixPath

import pytest

from mostt_model_builder.domain.containers.base_images import BASE_PYTHON
from mostt_model_builder.domain.containers.container_steps import (
    BOOTSTRAP_IN_THE_IMAGE,
    CLEAN_STEP_NAME,
    CONTAINER_USER_ID,
    CORE_IMAGES,
    FLEURS_CHECK_ROW,
    LOCK_ROW,
    LOCK_STEP_NAME,
    OUTPUT_IN_THE_CONTAINER,
    PEAKS_FILE_NAME,
    REQUEST_IN_THE_CONTAINER,
    TEMPORARY_DIRECTORY_IN_THE_CONTAINER,
    BindMount,
    ContainerImage,
    ContainerRun,
    DeclaredOutput,
    Device,
    SourceReference,
    StepLimits,
    StepPlacement,
    StepRow,
    StepTableError,
    TemporaryFilesystem,
    build_clean_row,
    judge_row,
    plan_container_run,
)
from mostt_model_builder.domain.contract.recipe_contract import (
    RECIPES_DIRECTORY_NAME,
    JudgedRecipe,
    RecipeFacts,
    judge_recipe,
)
from mostt_model_builder.host_main import (
    CLONE_ROOT,
    RECIPES,
    find_present_entry_modules,
    read_pins,
)

HOST_CPU_COUNT = 20
GROUP_ID = 1000
IMAGE_ID = "sha256:" + "a" * 64
WORK_ROOT = PurePosixPath("/data/builds/work")
REQUEST_DIRECTORY = PurePosixPath("runs/core/c0ffee/requests/step")
PLACEMENT = StepPlacement(
    output_directory=PurePosixPath("runs/core/c0ffee/step"),
    request_directory=REQUEST_DIRECTORY,
    log_path=PurePosixPath("runs/core/c0ffee/logs/step.log"),
)
BENCH_FILE_COUNT = 4
REVISION = "70bb2e84b976b7e960aa89f1c648e09c59f894dd"


def judge_every_recipe() -> tuple[JudgedRecipe, ...]:
    return tuple(
        judge_recipe(
            contract,
            RecipeFacts(
                core_pins=read_pins(CLONE_ROOT),
                recipe_pins=read_pins(CLONE_ROOT / RECIPES_DIRECTORY_NAME / contract.name),
                present_entry_modules=find_present_entry_modules(contract),
                host_cpu_count=HOST_CPU_COUNT,
            ),
        )
        for contract in RECIPES.values()
    )


JUDGED_RECIPES = judge_every_recipe()
CODE_RUNNING_SOURCES = {
    SourceReference(pin.repository, pin.path)
    for recipe in JUDGED_RECIPES
    for pin in recipe.pins
    if pin.runs_code_when_loaded
}
CODE_READING_IMAGES = {
    (contract.name, image.name)
    for contract in RECIPES.values()
    for image in contract.images
    if image.reads_code_running_sources
}
EVERY_ROW = (
    FLEURS_CHECK_ROW,
    LOCK_ROW,
    *(build_clean_row(image) for image in CORE_IMAGES),
    *(row for recipe in JUDGED_RECIPES for row in recipe.rows),
    *(build_clean_row(image) for recipe in JUDGED_RECIPES for image in recipe.images),
)
ROW_IDS = [f"{row.name} in {row.image.tag}" for row in EVERY_ROW]
GOOD_LIMITS = StepLimits(memory_mib=1024, process_count=64, cpu_count=2, temporary_directory_mib=64)
CLEAN_ROW = build_clean_row(CORE_IMAGES[0])
GPU_IMAGE = ContainerImage("core", "gpu-probe", Device.GPU, BASE_PYTHON)
RECIPE_IMAGE = ContainerImage("family", "export", Device.CPU, BASE_PYTHON)
SCRATCH_ROW = replace(FLEURS_CHECK_ROW, limits=replace(GOOD_LIMITS, scratch_mib=256))
REFUSED_ROWS = {
    "a name outside the grammar": (
        replace(FLEURS_CHECK_ROW, name="Fleurs-check"),
        "its name does not match",
    ),
    "an entry module outside the core's steps": (
        replace(FLEURS_CHECK_ROW, entry_module="mostt_model_builder.host_main"),
        "lies outside mostt_model_builder.steps",
    ),
    "a core entry module in a recipe's image, which runs only the recipe's steps": (
        replace(FLEURS_CHECK_ROW, image=RECIPE_IMAGE),
        "lies outside mostt_recipe_family.steps, the steps of the image "
        "mostt-model-builder-family-export",
    ),
    "a recipe's entry module named clean, which runs only the core's cleaning": (
        replace(
            build_clean_row(RECIPE_IMAGE), entry_module="mostt_recipe_family.steps.clean_output"
        ),
        "lies outside mostt_model_builder.steps",
    ),
    "an entry module that is not a dotted name": (
        replace(FLEURS_CHECK_ROW, entry_module="mostt_model_builder.steps.fleurs-check"),
        "is not a dotted module name",
    ),
    "memory under 6 MiB, which docker refuses": (
        replace(FLEURS_CHECK_ROW, limits=replace(GOOD_LIMITS, memory_mib=5)),
        "its memory in MiB, 5, lies outside 6 to 57344",
    ),
    "memory beyond its ceiling": (
        replace(FLEURS_CHECK_ROW, limits=replace(GOOD_LIMITS, memory_mib=57_345)),
        "its memory in MiB, 57345, lies outside 6 to 57344",
    ),
    "memory no larger than its temporary filesystems": (
        replace(FLEURS_CHECK_ROW, limits=replace(GOOD_LIMITS, memory_mib=64)),
        "does not exceed its temporary filesystems, 64 MiB",
    ),
    "no process, which docker reads as no limit": (
        replace(FLEURS_CHECK_ROW, limits=replace(GOOD_LIMITS, process_count=0)),
        "its count of processes, 0, lies outside 1 to 8192",
    ),
    "processes beyond the ceiling": (
        replace(FLEURS_CHECK_ROW, limits=replace(GOOD_LIMITS, process_count=8193)),
        "its count of processes, 8193, lies outside 1 to 8192",
    ),
    "a count of processes that is not a whole number": (
        replace(FLEURS_CHECK_ROW, limits=replace(GOOD_LIMITS, process_count=1.5)),  # type: ignore[arg-type]  # a recipe's limits come from plain data, which the types cannot hold
        "its count of processes, 1.5, lies outside 1 to 8192",
    ),
    "a count of cpus that is a boolean": (
        replace(FLEURS_CHECK_ROW, limits=replace(GOOD_LIMITS, cpu_count=True)),
        "its CPUs, True, lies outside 1 to 20",
    ),
    "no cpu": (
        replace(FLEURS_CHECK_ROW, limits=replace(GOOD_LIMITS, cpu_count=0)),
        "its CPUs, 0, lies outside 1 to 20",
    ),
    "more cpus than the host has": (
        replace(FLEURS_CHECK_ROW, limits=replace(GOOD_LIMITS, cpu_count=HOST_CPU_COUNT + 1)),
        "its CPUs, 21, lies outside 1 to 20",
    ),
    "an empty /tmp": (
        replace(FLEURS_CHECK_ROW, limits=replace(GOOD_LIMITS, temporary_directory_mib=0)),
        "its /tmp in MiB, 0, lies outside 1 to 16384",
    ),
    "a /tmp beyond its ceiling": (
        replace(
            FLEURS_CHECK_ROW,
            limits=replace(GOOD_LIMITS, temporary_directory_mib=16_385, memory_mib=20_000),
        ),
        "its /tmp in MiB, 16385, lies outside 1 to 16384",
    ),
    "an empty /scratch": (
        replace(FLEURS_CHECK_ROW, limits=replace(GOOD_LIMITS, scratch_mib=0)),
        "its /scratch in MiB, 0, lies outside 1 to 16384",
    ),
    "an output outside the file-name grammar": (
        replace(FLEURS_CHECK_ROW, outputs=(DeclaredOutput("../peaks.json", 10),)),
        "its output '../peaks.json' is not a file name",
    ),
    "an output with no byte to hold": (
        replace(
            FLEURS_CHECK_ROW,
            outputs=(DeclaredOutput("report.json", 0), DeclaredOutput(PEAKS_FILE_NAME, 10)),
        ),
        "the size bound of report.json in bytes, 0, lies outside 1 to 8589934592",
    ),
    "an output beyond the ceiling of 8 GiB": (
        replace(
            FLEURS_CHECK_ROW,
            outputs=(
                DeclaredOutput("report.json", 8 * 2**30 + 1),
                DeclaredOutput(PEAKS_FILE_NAME, 10),
            ),
        ),
        "the size bound of report.json in bytes, 8589934593, lies outside 1 to 8589934592",
    ),
    "an output declared twice": (
        replace(
            FLEURS_CHECK_ROW,
            outputs=(DeclaredOutput(PEAKS_FILE_NAME, 10), DeclaredOutput(PEAKS_FILE_NAME, 10)),
        ),
        "its output peaks.json is declared twice",
    ),
    "outputs without the peaks": (
        replace(FLEURS_CHECK_ROW, outputs=(DeclaredOutput("report.json", 10),)),
        "its outputs lack peaks.json",
    ),
    "the gpu in a cpu image": (
        replace(FLEURS_CHECK_ROW, uses_gpu=True),
        "uses the GPU in the cpu image",
    ),
    "the network on a step other than the lock": (
        replace(FLEURS_CHECK_ROW, has_network=True),
        "asks for the network, which only the core's lock has",
    ),
}


def plan_row(row: StepRow) -> ContainerRun:
    source_paths = {
        reference: PurePosixPath("sources", reference.repository, REVISION, reference.path)
        for reference in row.sources
    }
    earlier_output_directories = {
        earlier_output.step: PurePosixPath("runs/core/c0ffee", earlier_output.step)
        for earlier_output in row.earlier_outputs
    }
    return plan_container_run(
        row, IMAGE_ID, PLACEMENT, source_paths, WORK_ROOT, GROUP_ID, earlier_output_directories
    )


PROXY_VARIABLE_NAMES = (
    "HTTP_PROXY",
    "http_proxy",
    "HTTPS_PROXY",
    "https_proxy",
    "FTP_PROXY",
    "ftp_proxy",
    "NO_PROXY",
    "no_proxy",
    "ALL_PROXY",
    "all_proxy",
)


class TestTheContainerSteps:
    class TestTheJudgeRowFunction:
        @pytest.mark.parametrize("row", EVERY_ROW, ids=ROW_IDS)
        class TestWhenEachRowOfTheTableIsJudged:
            @pytest.fixture
            def judge_act(self, row: StepRow) -> Callable[[], object]:
                return partial(judge_row, row, HOST_CPU_COUNT)

            def test_should_find_its_limits_and_outputs_within_their_bounds(
                self, judge_act: Callable[[], object]
            ) -> None:
                assert judge_act() is None

        @pytest.mark.parametrize(
            ("row", "reason"), list(REFUSED_ROWS.values()), ids=list(REFUSED_ROWS.keys())
        )
        class TestWhenARowBreaksABound:
            @pytest.fixture
            def judge_act(self, row: StepRow) -> Callable[[], object]:
                return partial(judge_row, row, HOST_CPU_COUNT)

            def test_should_refuse_it_and_say_why(
                self, judge_act: Callable[[], object], reason: str
            ) -> None:
                with pytest.raises(StepTableError, match=re.escape(reason)):
                    judge_act()

        class TestWhenAGpuImageRunsAStepOnTheGpu:
            @pytest.fixture
            def judge_act(self) -> Callable[[], object]:
                return partial(
                    judge_row,
                    replace(FLEURS_CHECK_ROW, image=GPU_IMAGE, uses_gpu=True),
                    HOST_CPU_COUNT,
                )

            def test_should_accept_it(self, judge_act: Callable[[], object]) -> None:
                assert judge_act() is None

    class TestTheTableOfRows:
        class TestWhenEveryRowIsRead:
            def test_should_bind_a_source_that_runs_code_only_in_the_image_declared_to_read_it(
                self,
            ) -> None:
                bindings = [
                    (row.image.owner, row.image.name)
                    for row in EVERY_ROW
                    for reference in row.sources
                    if reference in CODE_RUNNING_SOURCES
                ]
                assert (len(bindings) > 0, set(bindings) - CODE_READING_IMAGES) == (True, set())

        @pytest.mark.parametrize("row", EVERY_ROW, ids=ROW_IDS)
        class TestWhenARowIsRead:
            def test_should_declare_the_peaks_among_its_outputs_unless_it_cleans(
                self, row: StepRow
            ) -> None:
                assert (PEAKS_FILE_NAME in [output.name for output in row.outputs]) is (
                    row.name != CLEAN_STEP_NAME
                )

            def test_should_use_the_gpu_only_in_an_image_of_the_gpu(self, row: StepRow) -> None:
                assert row.uses_gpu is False or row.image.device is Device.GPU

            def test_should_give_the_network_to_the_lock_alone(self, row: StepRow) -> None:
                assert row.has_network is (row.name == LOCK_STEP_NAME)

        def test_should_clean_with_the_limits_of_a_step_that_only_removes_files(self) -> None:
            assert CLEAN_ROW.limits == StepLimits(
                memory_mib=256, process_count=32, cpu_count=1, temporary_directory_mib=16
            )

        def test_should_check_fleurs_within_one_gib_eight_processes_and_two_cpus(self) -> None:
            assert FLEURS_CHECK_ROW.limits == StepLimits(
                memory_mib=1024, process_count=8, cpu_count=2, temporary_directory_mib=64
            )

        def test_should_lock_with_the_limits_of_the_trial_build(self) -> None:
            assert LOCK_ROW.limits == StepLimits(
                memory_mib=4096, process_count=256, cpu_count=8, temporary_directory_mib=3072
            )

        def test_should_tag_the_images_apart_from_the_trial_builds(self) -> None:
            assert [image.tag for image in CORE_IMAGES] == [
                "mostt-model-builder-core-bench",
                "mostt-model-builder-core-uv",
            ]

        def test_should_run_the_core_images_on_the_bases_python_without_a_venv(self) -> None:
            assert {image.python for image in CORE_IMAGES} == {BASE_PYTHON}

    class TestThePlanContainerRunFunction:
        @pytest.mark.parametrize("row", EVERY_ROW, ids=ROW_IDS)
        class TestWhenEachRowIsPlanned:
            @pytest.fixture
            def container_run(self, row: StepRow) -> ContainerRun:
                return plan_row(row)

            def test_should_start_the_images_python_isolated_on_the_bootstrap_and_the_entry(
                self, container_run: ContainerRun, row: StepRow
            ) -> None:
                assert (container_run.python, container_run.python_arguments) == (
                    row.image.python,
                    ("-I", "-B", str(BOOTSTRAP_IN_THE_IMAGE), row.entry_module),
                )

            def test_should_run_as_the_container_user_in_the_users_group(
                self, container_run: ContainerRun
            ) -> None:
                assert (container_run.user_id, container_run.group_id) == (
                    CONTAINER_USER_ID,
                    GROUP_ID,
                )

            def test_should_bind_only_the_output_directory_writable(
                self, container_run: ContainerRun
            ) -> None:
                assert [mount for mount in container_run.mounts if not mount.is_read_only] == [
                    BindMount(PLACEMENT.output_directory, OUTPUT_IN_THE_CONTAINER, False, False)
                ]

            def test_should_give_a_tmp_of_the_rows_size(
                self, container_run: ContainerRun, row: StepRow
            ) -> None:
                assert container_run.temporary_filesystems == (
                    TemporaryFilesystem(
                        TEMPORARY_DIRECTORY_IN_THE_CONTAINER, row.limits.temporary_directory_mib
                    ),
                )

            def test_should_set_every_proxy_variable_empty_so_that_dockers_own_never_reaches_it(
                self, container_run: ContainerRun
            ) -> None:
                assert {
                    name: setting
                    for name, setting in container_run.environment
                    if name in PROXY_VARIABLE_NAMES
                } == dict.fromkeys(PROXY_VARIABLE_NAMES, "")

            def test_should_bind_each_source_and_earlier_output_by_itself_read_only(
                self, container_run: ContainerRun, row: StepRow
            ) -> None:
                bound_files = [mount for mount in container_run.mounts if mount.is_file]
                assert len(bound_files) == len(row.sources) + len(row.earlier_outputs)
                assert [mount for mount in bound_files if not mount.is_read_only] == []

            def test_should_bind_each_source_at_the_layout_of_its_pin(
                self, container_run: ContainerRun, row: StepRow
            ) -> None:
                source_mounts = [
                    mount for mount in container_run.mounts if mount.host_path.parts[0] == "sources"
                ]
                assert (
                    len(source_mounts),
                    [
                        mount
                        for mount in source_mounts
                        if mount.container_path != PurePosixPath("/") / mount.host_path
                    ],
                ) == (len(row.sources), [])

            def test_should_keep_every_cache_and_home_under_tmp_and_the_hub_offline(
                self, container_run: ContainerRun
            ) -> None:
                assert {
                    name: setting
                    for name, setting in container_run.environment
                    if name not in PROXY_VARIABLE_NAMES
                } == {
                    "HOME": "/tmp/home",
                    "XDG_CACHE_HOME": "/tmp/cache",
                    "HF_HOME": "/tmp/huggingface",
                    "HF_HUB_OFFLINE": "1",
                    "HF_HUB_DISABLE_TELEMETRY": "1",
                }

        class TestWhenTheFleursCheckIsPlanned:
            @pytest.fixture
            def container_run(self) -> ContainerRun:
                return plan_row(FLEURS_CHECK_ROW)

            def test_should_bind_each_source_by_itself_read_only_at_its_layout(
                self, container_run: ContainerRun
            ) -> None:
                source_mounts = [
                    mount for mount in container_run.mounts if mount.host_path.parts[0] == "sources"
                ]
                assert len(source_mounts) == BENCH_FILE_COUNT
                assert [
                    mount
                    for mount in source_mounts
                    if not mount.is_file
                    or not mount.is_read_only
                    or mount.container_path != PurePosixPath("/") / mount.host_path
                ] == []

            def test_should_bind_the_index_and_the_audio_of_both_languages(
                self, container_run: ContainerRun
            ) -> None:
                assert sorted(
                    str(mount.host_path).split("/", 4)[4]
                    for mount in container_run.mounts
                    if mount.is_file
                ) == [
                    "data/en_us/audio/test.tar.gz",
                    "data/en_us/test.tsv",
                    "data/fr_fr/audio/test.tar.gz",
                    "data/fr_fr/test.tsv",
                ]

            def test_should_bind_its_request_read_only(self, container_run: ContainerRun) -> None:
                assert (
                    BindMount(REQUEST_DIRECTORY, REQUEST_IN_THE_CONTAINER, True, False)
                    in container_run.mounts
                )

        class TestWhenARowAsksForAScratch:
            @pytest.fixture
            def container_run(self) -> ContainerRun:
                return plan_container_run(
                    replace(SCRATCH_ROW, sources=()),
                    IMAGE_ID,
                    PLACEMENT,
                    {},
                    WORK_ROOT,
                    GROUP_ID,
                    {},
                )

            def test_should_add_a_scratch_temporary_filesystem(
                self, container_run: ContainerRun
            ) -> None:
                assert (
                    TemporaryFilesystem(PurePosixPath("/scratch"), 256)
                    in container_run.temporary_filesystems
                )

            def test_should_point_the_temporary_directory_at_it(
                self, container_run: ContainerRun
            ) -> None:
                assert ("TMPDIR", "/scratch") in container_run.environment

        class TestWhenARowWithARequestIsGivenNone:
            @pytest.fixture
            def plan_act(self) -> Callable[[], ContainerRun]:
                return partial(
                    plan_container_run,
                    LOCK_ROW,
                    IMAGE_ID,
                    replace(PLACEMENT, request_directory=None),
                    {},
                    WORK_ROOT,
                    GROUP_ID,
                    {},
                )

            def test_should_refuse_it(self, plan_act: Callable[[], ContainerRun]) -> None:
                with pytest.raises(StepTableError, match="needs a request, and was given none"):
                    plan_act()
