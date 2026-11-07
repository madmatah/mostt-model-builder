import os
import re
import subprocess
from collections.abc import Callable, Iterator
from dataclasses import replace
from functools import partial
from pathlib import Path, PurePosixPath

import pytest

from mostt_model_builder.adapters.docker_engine import (
    MOUNT_PATH_GRAMMAR,
    ContainerEngineError,
    DockerEngine,
    build_image_arguments,
    build_run_arguments,
    give_token,
    parse_image_inspection,
    run_logged,
)
from mostt_model_builder.application.build_images import TarArchive
from mostt_model_builder.application.run_step import ContainerExit, ImageFacts
from mostt_model_builder.domain.containers.container_steps import (
    BENCH_IMAGE,
    BOOTSTRAP_IN_THE_IMAGE,
    CORE_IMAGES,
    FLEURS_CHECK_ROW,
    INSPECT_ROW,
    LOCK_ROW,
    LOCK_STEP_NAME,
    PUBLISH_ROW,
    PUBLISH_STEP_NAME,
    BindMount,
    ContainerRun,
    StepPlacement,
    StepRow,
    build_clean_row,
    plan_container_run,
)
from mostt_model_builder.domain.containers.secret_redaction import REDACTED_SECRET
from mostt_model_builder.domain.contract.recipe_contract import (
    RECIPES_DIRECTORY_NAME,
    JudgedRecipe,
    RecipeFacts,
    judge_recipe,
)
from mostt_model_builder.domain.publication.hub_token import HubToken, parse_hub_token
from mostt_model_builder.host_main import (
    CLONE_ROOT,
    RECIPES,
    find_present_entry_modules,
    is_card_template_present,
    read_pins,
)

IMAGE_ID = "sha256:" + "b" * 64
TREE_HASH = "3f1d1ffe79bc3336eb583331cfbf9b189fd45f01"
GROUP_ID = 1000
WORK_ROOT = PurePosixPath("/data/builds/work")
REVISION = "70bb2e84b976b7e960aa89f1c648e09c59f894dd"
PLACEMENT = StepPlacement(
    output_directory=PurePosixPath("runs/core/c0ffee/step"),
    request_directory=PurePosixPath("runs/core/c0ffee/requests/step"),
    log_path=PurePosixPath("runs/core/c0ffee/logs/step.log"),
    candidate_directory=PurePosixPath("runs/core/c0ffee/candidate"),
)
HOST_CPU_COUNT = 20
PLACEMENT_CANDIDATE = PurePosixPath("runs/core/c0ffee/candidate")


def judge_every_recipe() -> tuple[JudgedRecipe, ...]:
    return tuple(
        judge_recipe(
            contract,
            RecipeFacts(
                core_pins=read_pins(CLONE_ROOT),
                recipe_pins=read_pins(CLONE_ROOT / RECIPES_DIRECTORY_NAME / contract.name),
                present_entry_modules=find_present_entry_modules(contract),
                host_cpu_count=HOST_CPU_COUNT,
                has_card_template=is_card_template_present(contract, CLONE_ROOT),
            ),
        )
        for contract in RECIPES.values()
    )


JUDGED_RECIPES = judge_every_recipe()
RUN_ROWS_WITH_CANDIDATE_FILES = tuple(
    derived_row.row
    for recipe in JUDGED_RECIPES
    for derived_row in recipe.derived_rows
    if derived_row.row.candidate_files
)
BUILD_ROWS = (
    FLEURS_CHECK_ROW,
    INSPECT_ROW,
    LOCK_ROW,
    *(build_clean_row(image) for image in CORE_IMAGES),
    *(row for recipe in JUDGED_RECIPES for row in recipe.rows),
    *(build_clean_row(image) for recipe in JUDGED_RECIPES for image in recipe.images),
)
BUILD_ROW_IDS = [f"{row.name} in {row.image.tag}" for row in BUILD_ROWS]
EVERY_ROW = (*BUILD_ROWS, PUBLISH_ROW)
ROW_IDS = [f"{row.name} in {row.image.tag}" for row in EVERY_ROW]
CONFINEMENT_SEQUENCES = {
    "removed once it exits": ["--rm"],
    "never pulled, so only a built image runs": ["--pull", "never"],
    "a read-only root": ["--read-only"],
    "every capability dropped": ["--cap-drop", "ALL"],
    "no new privileges": ["--security-opt", "no-new-privileges"],
    "no shared memory with the host": ["--ipc", "none"],
    "no log kept by docker, since the host keeps its own": ["--log-driver", "none"],
    "no core dump": ["--ulimit", "core=0"],
    "the container user in the user's group": ["--user", f"65531:{GROUP_ID}"],
}
FAKE_DOCKER_OUTPUT = b"line one\n\x1b[31mred\x07\x00\xff\n"
FAKE_DOCKER_STATUS = 3
CONTEXT_ARCHIVE = TarArchive(b"the archive of the tree\0\xff")
NETWORK_BY_STEP_NAME = {LOCK_STEP_NAME: "bridge", PUBLISH_STEP_NAME: "bridge"}
NO_NETWORK = "none"
READ_ONLY_OPTION = {True: ",readonly", False: ""}

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


def find_sequence(arguments: list[str], sequence: list[str]) -> int | None:
    for start in range(len(arguments) - len(sequence) + 1):
        if arguments[start : start + len(sequence)] == sequence:
            return start
    return None


def read_option_values(arguments: list[str], option: str) -> list[str]:
    return [
        arguments[position + 1]
        for position, argument in enumerate(arguments[:-1])
        if argument == option
    ]


def write_fake_docker(directory: Path, output: bytes, status: int) -> Path:
    output_path = directory / "output.bin"
    output_path.write_bytes(output)
    fake_docker = directory / "docker"
    fake_docker.write_text(f'#!/usr/bin/env bash\ncat "{output_path}"\nexit {status}\n')
    fake_docker.chmod(0o755)
    return fake_docker


PLACEHOLDER_TOKEN = parse_hub_token("hf_placeholder0123456789")
TOKEN_ROW = replace(INSPECT_ROW, takes_token=True)


def write_echoing_docker(directory: Path) -> None:
    echoing_docker = directory / "docker"
    echoing_docker.write_text(
        "#!/usr/bin/env bash\n"
        "IFS= read -r line\n"
        'printf "read %s, then %s%s\\n" "$line" "${line:0:7}" "${line:7}"\n'
    )
    echoing_docker.chmod(0o755)


def run_with_echoing_docker(
    directory: Path, monkeypatch: pytest.MonkeyPatch, token: HubToken | None
) -> tuple[ContainerExit, bytes]:
    program_directory = directory / "programs"
    program_directory.mkdir()
    write_echoing_docker(program_directory)
    monkeypatch.setenv("PATH", f"{program_directory}{os.pathsep}{os.environ['PATH']}")
    log_path = directory / "publish.log"
    container_exit = DockerEngine().run_container(
        plan_row(TOKEN_ROW), PurePosixPath(log_path), token
    )
    return container_exit, log_path.read_bytes()


def write_recording_docker(directory: Path, record_directory: Path) -> None:
    directory.mkdir()
    recording_docker = directory / "docker"
    recording_docker.write_text(
        "#!/usr/bin/env bash\n"
        f'printf "%s\\n" "$@" > "{record_directory / "arguments.txt"}"\n'
        f'cat > "{record_directory / "input.bin"}"\n'
    )
    recording_docker.chmod(0o755)


class TestTheDockerEngine:
    class TestTheBuildRunArgumentsFunction:
        @pytest.mark.parametrize("row", EVERY_ROW, ids=ROW_IDS)
        @pytest.mark.parametrize(
            "sequence",
            list(CONFINEMENT_SEQUENCES.values()),
            ids=list(CONFINEMENT_SEQUENCES.keys()),
        )
        class TestWhenARowRunsWithAConfinementOption:
            @pytest.fixture
            def run_arguments(self, row: StepRow) -> list[str]:
                return build_run_arguments(plan_row(row))

            def test_should_hold_it(self, run_arguments: list[str], sequence: list[str]) -> None:
                assert find_sequence(run_arguments, sequence) is not None

        @pytest.mark.parametrize("row", EVERY_ROW, ids=ROW_IDS)
        class TestWhenARowRuns:
            @pytest.fixture
            def run_arguments(self, row: StepRow) -> list[str]:
                return build_run_arguments(plan_row(row))

            def test_should_end_with_the_images_python_isolated_on_the_bootstrap_and_the_entry(
                self, run_arguments: list[str], row: StepRow
            ) -> None:
                assert run_arguments[-7:] == [
                    "--entrypoint",
                    str(row.image.python),
                    IMAGE_ID,
                    "-I",
                    "-B",
                    str(BOOTSTRAP_IN_THE_IMAGE),
                    row.entry_module,
                ]

            def test_should_bound_the_swap_to_the_memory_so_no_step_swaps_beyond_it(
                self, run_arguments: list[str], row: StepRow
            ) -> None:
                assert (
                    read_option_values(run_arguments, "--memory"),
                    read_option_values(run_arguments, "--memory-swap"),
                ) == ([f"{row.limits.memory_mib}m"], [f"{row.limits.memory_mib}m"])

            def test_should_bound_its_processes_and_cpus(
                self, run_arguments: list[str], row: StepRow
            ) -> None:
                assert (
                    read_option_values(run_arguments, "--pids-limit"),
                    read_option_values(run_arguments, "--cpus"),
                ) == ([str(row.limits.process_count)], [str(row.limits.cpu_count)])

            def test_should_mount_tmp_and_any_scratch_as_bounded_filesystems_that_run_nothing(
                self, run_arguments: list[str], row: StepRow
            ) -> None:
                assert read_option_values(run_arguments, "--tmpfs") == [
                    f"/tmp:rw,noexec,nosuid,nodev,size={row.limits.temporary_directory_mib}m",
                    *(
                        [f"/scratch:rw,noexec,nosuid,nodev,size={row.limits.scratch_mib}m"]
                        if row.limits.scratch_mib is not None
                        else []
                    ),
                ]

            def test_should_give_the_network_to_the_lock_and_the_publication_alone(
                self, run_arguments: list[str], row: StepRow
            ) -> None:
                assert read_option_values(run_arguments, "--network") == [
                    NETWORK_BY_STEP_NAME.get(row.name, NO_NETWORK)
                ]

            def test_should_set_every_proxy_variable_empty_so_that_dockers_own_never_reaches_it(
                self, run_arguments: list[str]
            ) -> None:
                assert sorted(
                    setting
                    for setting in read_option_values(run_arguments, "--env")
                    if setting.split("=", 1)[0] in PROXY_VARIABLE_NAMES
                ) == sorted(f"{name}=" for name in PROXY_VARIABLE_NAMES)

            def test_should_give_the_gpus_to_a_row_that_uses_them_alone(
                self, run_arguments: list[str], row: StepRow
            ) -> None:
                assert read_option_values(run_arguments, "--device") == (
                    ["nvidia.com/gpu=all"] if row.uses_gpu else []
                )

            def test_should_mount_only_the_rows_files_and_directories(
                self, run_arguments: list[str], row: StepRow
            ) -> None:
                assert read_option_values(run_arguments, "--mount") == [
                    f"type=bind,source={WORK_ROOT / mount.host_path},"
                    f"target={mount.container_path}{READ_ONLY_OPTION[mount.is_read_only]}"
                    for mount in plan_row(row).mounts
                ]

        class TestWhenARunUsesTheGpu:
            @pytest.fixture
            def run_arguments(self) -> list[str]:
                return build_run_arguments(replace(plan_row(FLEURS_CHECK_ROW), uses_gpu=True))

            def test_should_add_every_gpu_through_the_container_device_interface(
                self, run_arguments: list[str]
            ) -> None:
                assert read_option_values(run_arguments, "--device") == ["nvidia.com/gpu=all"]

        class TestWhenAMountPathHoldsAComma:
            @pytest.fixture
            def build_act(self) -> Callable[[], list[str]]:
                run = plan_row(LOCK_ROW)
                mount = BindMount(
                    PurePosixPath("runs/a,readonly=false"), PurePosixPath("/x"), True, False
                )
                return partial(build_run_arguments, replace(run, mounts=(mount,)))

            def test_should_refuse_it_since_docker_reads_a_comma_as_syntax(
                self, build_act: Callable[[], list[str]]
            ) -> None:
                with pytest.raises(
                    ContainerEngineError, match="a mount '/data/builds/work/runs/a,"
                ):
                    build_act()

        @pytest.mark.parametrize(
            "host_path",
            ["runs/../../etc", "runs/.."],
            ids=["a dot-dot segment inside", "a dot-dot segment last"],
        )
        class TestWhenAMountPathHoldsADotDotSegment:
            @pytest.fixture
            def build_act(self, host_path: str) -> Callable[[], list[str]]:
                run = plan_row(LOCK_ROW)
                mount = BindMount(PurePosixPath(host_path), PurePosixPath("/x"), True, False)
                return partial(build_run_arguments, replace(run, mounts=(mount,)))

            def test_should_refuse_it_since_docker_would_resolve_it_elsewhere(
                self, build_act: Callable[[], list[str]], host_path: str
            ) -> None:
                with pytest.raises(
                    ContainerEngineError,
                    match=re.escape(f"a mount '/data/builds/work/{host_path}' does not match"),
                ):
                    build_act()

        class TestWhenAMountPathHoldsANameThatStartsWithADot:
            @pytest.fixture
            def run_arguments(self) -> list[str]:
                run = plan_row(LOCK_ROW)
                mount = BindMount(PurePosixPath("runs/..step"), PurePosixPath("/x"), True, False)
                return build_run_arguments(replace(run, mounts=(mount,)))

            def test_should_bind_it(self, run_arguments: list[str]) -> None:
                assert read_option_values(run_arguments, "--mount") == [
                    "type=bind,source=/data/builds/work/runs/..step,target=/x,readonly"
                ]

        class TestWhenTheRowTakesTheToken:
            @pytest.fixture
            def run_arguments(self) -> list[str]:
                return build_run_arguments(plan_row(TOKEN_ROW))

            def test_should_keep_the_standard_input_open_and_name_no_token(
                self, run_arguments: list[str]
            ) -> None:
                assert (
                    "--interactive" in run_arguments,
                    any("hf_" in argument for argument in run_arguments),
                ) == (True, False)

        @pytest.mark.parametrize("row", BUILD_ROWS, ids=BUILD_ROW_IDS)
        class TestWhenARowOfTheBuildRuns:
            @pytest.fixture
            def run_arguments(self, row: StepRow) -> list[str]:
                return build_run_arguments(plan_row(row))

            def test_should_give_it_no_standard_input(self, run_arguments: list[str]) -> None:
                assert "--interactive" not in run_arguments

        @pytest.mark.parametrize(
            "row",
            RUN_ROWS_WITH_CANDIDATE_FILES,
            ids=[row.name for row in RUN_ROWS_WITH_CANDIDATE_FILES],
        )
        class TestWhenARunOfAVariantRuns:
            @pytest.fixture
            def run_arguments(self, row: StepRow) -> list[str]:
                return build_run_arguments(plan_row(row))

            def test_should_mount_each_file_of_its_variant_by_itself_read_only_under_candidate(
                self, run_arguments: list[str], row: StepRow
            ) -> None:
                assert [
                    mount
                    for mount in read_option_values(run_arguments, "--mount")
                    if ",target=/candidate" in mount
                ] == [
                    f"type=bind,source={WORK_ROOT / PLACEMENT_CANDIDATE / name},"
                    f"target=/candidate/{name},readonly"
                    for name in row.candidate_files
                ]

        class TestWhenThePublicationRuns:
            @pytest.fixture
            def run_arguments(self) -> list[str]:
                return build_run_arguments(plan_row(PUBLISH_ROW))

            def test_should_give_it_the_bridge_and_its_standard_input(
                self, run_arguments: list[str]
            ) -> None:
                assert (
                    read_option_values(run_arguments, "--network"),
                    "--interactive" in run_arguments,
                ) == (
                    ["bridge"],
                    True,
                )

    class TestTheMountPathGrammarConstant:
        @pytest.mark.parametrize(
            "path", ["/a/./b", "/a/../b", "/a/.", "/.."], ids=["dot", "dot dot", "dot last", "root"]
        )
        class TestWhenAPathHoldsADotSegment:
            def test_should_refuse_it(self, path: str) -> None:
                assert MOUNT_PATH_GRAMMAR.fullmatch(path) is None

        @pytest.mark.parametrize(
            "path", ["/a/..b", "/a/.hidden", "/a/b.c"], ids=["dot dot name", "dot name", "suffix"]
        )
        class TestWhenANameOnlyHoldsDots:
            def test_should_accept_it(self, path: str) -> None:
                assert MOUNT_PATH_GRAMMAR.fullmatch(path) is not None

    class TestTheBuildImageArgumentsFunction:
        class TestWhenTheBenchImageIsBuilt:
            @pytest.fixture
            def build_arguments(self) -> list[str]:
                return build_image_arguments(
                    PurePosixPath("images/bench/Containerfile"),
                    "mostt-model-builder-core-bench",
                    TREE_HASH,
                )

            def test_should_build_its_containerfile_from_its_input_under_its_tag_and_tree(
                self, build_arguments: list[str]
            ) -> None:
                assert build_arguments == [
                    "docker",
                    "build",
                    "--progress",
                    "plain",
                    "--file",
                    "images/bench/Containerfile",
                    "--tag",
                    "mostt-model-builder-core-bench",
                    "--build-arg",
                    f"TREE_HASH={TREE_HASH}",
                    *[
                        argument
                        for name in PROXY_VARIABLE_NAMES
                        for argument in ("--build-arg", f"{name}=")
                    ],
                    "-",
                ]

            def test_should_set_every_proxy_variable_empty_so_that_dockers_own_never_reaches_it(
                self, build_arguments: list[str]
            ) -> None:
                assert sorted(
                    setting
                    for setting in read_option_values(build_arguments, "--build-arg")
                    if setting.split("=", 1)[0] in PROXY_VARIABLE_NAMES
                ) == sorted(f"{name}=" for name in PROXY_VARIABLE_NAMES)

            def test_should_pass_no_frontend_so_the_built_in_one_reads_the_containerfile(
                self, build_arguments: list[str]
            ) -> None:
                assert [
                    argument for argument in build_arguments if "BUILDKIT_SYNTAX" in argument
                ] == []

    class TestTheParseImageInspectionFunction:
        class TestWhenTheImageCarriesItsTree:
            @pytest.fixture
            def image_facts(self) -> ImageFacts:
                return parse_image_inspection(
                    f'"{IMAGE_ID}" {{"mostt-model-builder.tree":"{TREE_HASH}"}}\n'.encode()
                )

            def test_should_read_its_id_and_its_tree(self, image_facts: ImageFacts) -> None:
                assert image_facts == ImageFacts(IMAGE_ID, TREE_HASH)

        class TestWhenTheImageCarriesNoLabel:
            @pytest.fixture
            def image_facts(self) -> ImageFacts:
                return parse_image_inspection(f'"{IMAGE_ID}" null\n'.encode())

            def test_should_find_no_tree(self, image_facts: ImageFacts) -> None:
                assert image_facts.tree_label is None

        class TestWhenTheLabelsNest3000Deep:
            @pytest.fixture
            def parse_act(self) -> Callable[[], ImageFacts]:
                return partial(
                    parse_image_inspection, f'"{IMAGE_ID}" '.encode() + b"[" * 3000 + b"\n"
                )

            def test_should_refuse_them_before_json_decodes_them(
                self, parse_act: Callable[[], ImageFacts]
            ) -> None:
                with pytest.raises(
                    ContainerEngineError, match="the document nests deeper than 32 levels"
                ):
                    parse_act()

        class TestWhenTheIdIsMalformed:
            @pytest.fixture
            def parse_act(self) -> Callable[[], ImageFacts]:
                return partial(parse_image_inspection, b'"sha256:abc" null\n')

            def test_should_refuse_it(self, parse_act: Callable[[], ImageFacts]) -> None:
                with pytest.raises(
                    ContainerEngineError, match="docker gave the image id 'sha256:abc'"
                ):
                    parse_act()

    class TestTheInspectImageMethod:
        class TestWhenDockerGivesNoAnswerInTime:
            @pytest.fixture
            def inspect_act(
                self, monkeypatch: pytest.MonkeyPatch
            ) -> Callable[[], ImageFacts | None]:
                def run_until_the_timeout(
                    arguments: list[str], *, timeout: float, **_options: object
                ) -> subprocess.CompletedProcess[bytes]:
                    raise subprocess.TimeoutExpired(arguments, timeout)

                monkeypatch.setattr(subprocess, "run", run_until_the_timeout)
                return partial(
                    DockerEngine().inspect_image,
                    "mostt-model-builder-core-bench",
                )

            def test_should_refuse_it_and_name_the_command_and_its_timeout(
                self, inspect_act: Callable[[], ImageFacts | None]
            ) -> None:
                with pytest.raises(
                    ContainerEngineError,
                    match=r"^docker image inspect gave no answer within 60 s$",
                ):
                    inspect_act()

    class TestTheBuildImageMethod:
        class TestWhenAnImageIsBuilt:
            @pytest.fixture
            def recorded_build(
                self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
            ) -> tuple[list[str], bytes]:
                program_directory = tmp_path / "programs"
                write_recording_docker(program_directory, tmp_path)
                monkeypatch.setenv("PATH", f"{program_directory}{os.pathsep}{os.environ['PATH']}")
                DockerEngine().build_image(
                    BENCH_IMAGE, TREE_HASH, CONTEXT_ARCHIVE, PurePosixPath(tmp_path / "build.log")
                )
                return (
                    (tmp_path / "arguments.txt").read_text().splitlines(),
                    (tmp_path / "input.bin").read_bytes(),
                )

            def test_should_give_docker_the_archive_on_its_standard_input(
                self, recorded_build: tuple[list[str], bytes]
            ) -> None:
                assert recorded_build[1] == CONTEXT_ARCHIVE

            def test_should_find_the_containerfile_in_the_archive_it_reads_as_its_context(
                self, recorded_build: tuple[list[str], bytes]
            ) -> None:
                assert (
                    read_option_values(recorded_build[0], "--file"),
                    recorded_build[0][-1],
                ) == ([str(BENCH_IMAGE.containerfile)], "-")

    class TestTheRunLoggedFunction:
        class TestWhenTheProgramPrintsBytesATerminalWouldRead:
            @pytest.fixture
            def logged_run(self, tmp_path: Path) -> tuple[ContainerExit, bytes]:
                fake_docker = write_fake_docker(tmp_path, FAKE_DOCKER_OUTPUT, FAKE_DOCKER_STATUS)
                log_path = tmp_path / "step.log"
                container_exit = run_logged([str(fake_docker)], PurePosixPath(log_path))
                return container_exit, log_path.read_bytes()

            def test_should_log_them_through_the_printable_filter(
                self, logged_run: tuple[ContainerExit, bytes]
            ) -> None:
                assert logged_run[1] == b"line one\n?[31mred???\n"

            def test_should_keep_the_same_filtered_tail(
                self, logged_run: tuple[ContainerExit, bytes]
            ) -> None:
                assert logged_run[0].log_tail == b"line one\n?[31mred???\n"

            def test_should_report_its_status(
                self, logged_run: tuple[ContainerExit, bytes]
            ) -> None:
                assert logged_run[0].status == FAKE_DOCKER_STATUS

        class TestWhenTheLogIsALink:
            @pytest.fixture
            def run_act(self, tmp_path: Path) -> Callable[[], ContainerExit]:
                fake_docker = write_fake_docker(tmp_path, FAKE_DOCKER_OUTPUT, 0)
                (tmp_path / "elsewhere.log").write_bytes(b"")
                (tmp_path / "step.log").symlink_to(tmp_path / "elsewhere.log")
                return partial(run_logged, [str(fake_docker)], PurePosixPath(tmp_path / "step.log"))

            def test_should_refuse_to_follow_it(self, run_act: Callable[[], ContainerExit]) -> None:
                with pytest.raises(OSError, match=re.escape("Too many levels of symbolic links")):
                    run_act()

    class TestTheRunContainerMethod:
        class TestWhenARowTakesTheToken:
            @pytest.fixture
            def logged_run(
                self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
            ) -> tuple[ContainerExit, bytes]:
                return run_with_echoing_docker(tmp_path, monkeypatch, PLACEHOLDER_TOKEN)

            def test_should_give_it_on_the_standard_input_and_log_only_its_placeholder(
                self, logged_run: tuple[ContainerExit, bytes]
            ) -> None:
                assert logged_run[1] == (
                    b"read " + REDACTED_SECRET + b", then " + REDACTED_SECRET + b"\n"
                )

            def test_should_show_only_the_placeholder_in_the_tail_too(
                self, logged_run: tuple[ContainerExit, bytes]
            ) -> None:
                assert PLACEHOLDER_TOKEN.encode_secret() not in logged_run[0].log_tail

        class TestWhenARowThatTakesTheTokenIsGivenNone:
            @pytest.fixture
            def run_act(self, tmp_path: Path) -> Callable[[], ContainerExit]:
                return partial(
                    DockerEngine().run_container,
                    plan_row(TOKEN_ROW),
                    PurePosixPath(tmp_path / "publish.log"),
                )

            def test_should_refuse_to_start_it(self, run_act: Callable[[], ContainerExit]) -> None:
                with pytest.raises(ContainerEngineError, match="exactly when its row takes it"):
                    run_act()

        class TestWhenARowThatTakesNoTokenIsGivenOne:
            @pytest.fixture
            def run_act(self, tmp_path: Path) -> Callable[[], ContainerExit]:
                return partial(
                    DockerEngine().run_container,
                    plan_row(INSPECT_ROW),
                    PurePosixPath(tmp_path / "inspect.log"),
                    PLACEHOLDER_TOKEN,
                )

            def test_should_refuse_to_start_it(self, run_act: Callable[[], ContainerExit]) -> None:
                with pytest.raises(ContainerEngineError, match="exactly when its row takes it"):
                    run_act()

    class TestTheGiveTokenFunction:
        class TestWhenDockerExitedBeforeReadingItsStandardInput:
            @pytest.fixture
            def give_act(self) -> Iterator[Callable[[], object]]:
                with subprocess.Popen(["true"], stdin=subprocess.PIPE) as exited_process:
                    exited_process.wait()
                    yield partial(give_token, exited_process, PLACEHOLDER_TOKEN)

            def test_should_leave_the_exit_to_the_status_and_the_log_of_docker(
                self, give_act: Callable[[], object]
            ) -> None:
                assert give_act() is None
