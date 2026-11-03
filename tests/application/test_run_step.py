import contextlib
import re
from collections.abc import Callable
from dataclasses import dataclass, replace
from functools import partial
from pathlib import PurePosixPath

import pytest
from run_step_fakes import (
    GROUP_ID,
    USER_ID,
    FakeContainerEngine,
    FakeStepFiles,
    build_directory_facts,
    build_step_output,
)

from mostt_model_builder.application.run_step import (
    HostIdentity,
    ImageFacts,
    StepError,
    StepOutputs,
    StepRunRequest,
    run_step,
)
from mostt_model_builder.domain.containers.account_ranges import (
    AccountRangeError,
    HostAccounts,
)
from mostt_model_builder.domain.containers.container_files import (
    ContainerFileError,
    FileFacts,
    FileKind,
)
from mostt_model_builder.domain.containers.container_steps import (
    BOOTSTRAP_IN_THE_IMAGE,
    CONTAINER_USER_ID,
    LOCK_ROW,
    PEAKS_FILE_NAME,
    BindMount,
    ContainerRun,
    DeclaredOutput,
    EarlierOutput,
    SourceReference,
    StepPlacement,
)
from mostt_model_builder.domain.containers.provenance import ProvenanceError
from mostt_model_builder.domain.containers.step_peaks import StepPeaks, render_step_peaks

WORK_ROOT = PurePosixPath("/data/builds/work")
HEAD_TREE = "3f1d1ffe79bc3336eb583331cfbf9b189fd45f01"
OTHER_TREE = "e949fa3a2eb8bf218e96c2c03625bbd36e4342b1"
IMAGE_ID = "sha256:" + "c" * 64
TAG = LOCK_ROW.image.tag
PLACEMENT = StepPlacement(
    output_directory=PurePosixPath("lock/1/output"),
    request_directory=PurePosixPath("lock/1/request"),
    log_path=PurePosixPath("lock/1/logs/lock.log"),
)
OUTPUT_DIRECTORY = WORK_ROOT / PLACEMENT.output_directory
REQUEST_DIRECTORY = WORK_ROOT / "lock/1/request"
PEAKS = StepPeaks(memory_peak_bytes=64 * 2**20, process_count_peak=6)
LOCK_CONTENT = b'version = 1\nrevision = 3\nrequires-python = "==3.12.*"\n'
MANIFEST_CONTENT = b'[project]\nname = "mostt-model-builder-images"\n'
HOST = HostIdentity(USER_ID, GROUP_ID, 20, HostAccounts(False, (), ()))
REQUEST = StepRunRequest(
    row=LOCK_ROW,
    placement=PLACEMENT,
    request_files={"pyproject.toml": MANIFEST_CONTENT},
    source_paths={},
    work_root=WORK_ROOT,
    head_tree=HEAD_TREE,
    host=HOST,
)
GOOD_OUTPUTS = {
    "uv.lock": build_step_output(LOCK_CONTENT),
    PEAKS_FILE_NAME: build_step_output(render_step_peaks(PEAKS)),
}
LOCK_ROW_SOURCE = SourceReference("google/fleurs", "data/fr_fr/test.tsv")
FAILED_STEP_TAIL = b"Traceback\n\x1b[31mError\n"
NAMES_OUTPUT = DeclaredOutput("names", 1024)
NAMES_CONTENT = b"encoder.weight\tfloat32\t[2,2]\n"
EARLIER_DIRECTORY = PurePosixPath("runs/family/c0ffee/describe")
EARLIER_OUTPUT = EarlierOutput("describe", NAMES_OUTPUT, (NAMES_OUTPUT.name, PEAKS_FILE_NAME))
EARLIER_OUTPUT_REQUEST = replace(
    REQUEST,
    row=replace(LOCK_ROW, earlier_outputs=(EARLIER_OUTPUT,)),
    earlier_output_directories={"describe": EARLIER_DIRECTORY},
)
EARLIER_FILES = {
    NAMES_OUTPUT.name: build_step_output(NAMES_CONTENT),
    PEAKS_FILE_NAME: build_step_output(render_step_peaks(PEAKS)),
}


@dataclass(frozen=True)
class StepRecord:
    outputs: StepOutputs | None
    runs: list[ContainerRun]
    files: FakeStepFiles


@dataclass(frozen=True)
class CompletedStep:
    outputs: StepOutputs
    runs: list[ContainerRun]
    files: FakeStepFiles


def build_files() -> FakeStepFiles:
    files = FakeStepFiles()
    files.add_directory(WORK_ROOT)
    return files


def build_engine(
    files: FakeStepFiles,
    outputs: dict[str, tuple[FileFacts, bytes]] | None = None,
    tree_label: str | None = HEAD_TREE,
    status: int = 0,
) -> FakeContainerEngine:
    return FakeContainerEngine(
        files=files,
        images={TAG: ImageFacts(IMAGE_ID, tree_label)},
        step_outputs=dict(GOOD_OUTPUTS if outputs is None else outputs),
        status=status,
        log_tail=FAILED_STEP_TAIL,
    )


def build_files_with_earlier_outputs(
    earlier_files: dict[str, tuple[FileFacts, bytes]],
) -> FakeStepFiles:
    files = build_files()
    files.add_directory(WORK_ROOT / EARLIER_DIRECTORY, build_directory_facts(0o2770))
    for name, (facts, content) in earlier_files.items():
        files.add_file(WORK_ROOT / EARLIER_DIRECTORY / name, facts, content)
    return files


def run_quietly(
    request: StepRunRequest, engine: FakeContainerEngine, files: FakeStepFiles
) -> StepRecord:
    outputs = None
    with contextlib.suppress(StepError, ContainerFileError, ProvenanceError, AccountRangeError):
        outputs = run_step(request, engine, files)
    return StepRecord(outputs, engine.runs, files)


class TestTheRunStep:
    class TestTheRunStepFunction:
        class TestWhenTheStepWritesItsDeclaredOutputs:
            @pytest.fixture
            def step_record(self) -> CompletedStep:
                files = build_files()
                engine = build_engine(files)
                return CompletedStep(run_step(REQUEST, engine, files), engine.runs, files)

            def test_should_give_back_its_peaks(self, step_record: CompletedStep) -> None:
                assert step_record.outputs.peaks == PEAKS

            def test_should_read_its_peaks_alone_since_no_parser_of_the_host_reads_the_rest(
                self, step_record: CompletedStep
            ) -> None:
                assert (step_record.outputs.contents, step_record.files.read_paths) == (
                    {},
                    [OUTPUT_DIRECTORY / PEAKS_FILE_NAME],
                )

            def test_should_give_back_the_size_of_its_other_outputs(
                self, step_record: CompletedStep
            ) -> None:
                assert step_record.outputs.sizes_bytes == {"uv.lock": len(LOCK_CONTENT)}

            def test_should_start_the_images_python_isolated_on_the_bootstrap_of_the_image(
                self, step_record: CompletedStep
            ) -> None:
                assert [(run.image_id, run.python_arguments) for run in step_record.runs] == [
                    (IMAGE_ID, ("-I", "-B", str(BOOTSTRAP_IN_THE_IMAGE), LOCK_ROW.entry_module))
                ]

            def test_should_make_the_output_directory_writable_by_the_users_group(
                self, step_record: CompletedStep
            ) -> None:
                assert step_record.files.entries[OUTPUT_DIRECTORY] == build_directory_facts(0o2770)

            def test_should_write_the_request_readable_by_the_users_group_alone(
                self, step_record: CompletedStep
            ) -> None:
                assert (
                    step_record.files.entries[REQUEST_DIRECTORY].permission_bits,
                    step_record.files.file_modes[REQUEST_DIRECTORY / "pyproject.toml"],
                ) == (0o750, 0o640)

            def test_should_keep_the_log_outside_every_output_directory(
                self, step_record: CompletedStep
            ) -> None:
                assert step_record.files.entries[WORK_ROOT / PLACEMENT.log_path].kind is (
                    FileKind.REGULAR
                )

        class TestWhenTheHostParsesOneOfItsOutputs:
            @pytest.fixture
            def step_outputs(self) -> StepOutputs:
                files = build_files()
                request = replace(REQUEST, parsed_output_names=frozenset({"uv.lock"}))
                return run_step(request, build_engine(files), files)

            def test_should_give_back_its_content_read_through_its_descriptor(
                self, step_outputs: StepOutputs
            ) -> None:
                assert step_outputs.contents == {"uv.lock": LOCK_CONTENT}

        class TestWhenTheHostParsesAnOutputTheStepDoesNotDeclare:
            @pytest.fixture
            def run_act(self) -> Callable[[], StepOutputs]:
                files = build_files()
                request = replace(REQUEST, parsed_output_names=frozenset({"report.json"}))
                return partial(run_step, request, build_engine(files), files)

            def test_should_refuse_it_and_name_the_output(
                self, run_act: Callable[[], StepOutputs]
            ) -> None:
                with pytest.raises(
                    StepError,
                    match=re.escape(
                        "the host parses report.json, which the step lock does not declare"
                    ),
                ):
                    run_act()

        class TestWhenAnUndeclaredOutputIsToBeParsed:
            @pytest.fixture
            def step_record(self) -> StepRecord:
                files = build_files()
                request = replace(REQUEST, parsed_output_names=frozenset({"report.json"}))
                return run_quietly(request, build_engine(files), files)

            def test_should_refuse_it_before_any_container_runs(
                self, step_record: StepRecord
            ) -> None:
                assert (step_record.outputs, step_record.runs) == (None, [])

        class TestWhenItsOutputsAlreadyExist:
            @pytest.fixture
            def run_act(self) -> Callable[[], StepOutputs]:
                files = build_files()
                files.add_directory(OUTPUT_DIRECTORY)
                return partial(run_step, REQUEST, build_engine(files), files)

            def test_should_refuse_to_run_and_name_clean(
                self, run_act: Callable[[], StepOutputs]
            ) -> None:
                with pytest.raises(
                    StepError,
                    match=re.escape(
                        "the outputs of the step lock already exist in lock/1/output: "
                        "remove them with ./model-build.sh --work <dir> --recipe <name> clean lock"
                    ),
                ):
                    run_act()

        class TestWhenADirectoryStandsWhereAFileIsDeclared:
            @pytest.fixture
            def step_record(self) -> StepRecord:
                files = build_files()
                source = PurePosixPath("sources/google/fleurs/0123/data/fr_fr/test.tsv")
                files.add_directory(WORK_ROOT / source)
                row = replace(LOCK_ROW, sources=(LOCK_ROW_SOURCE,))
                request = replace(REQUEST, row=row, source_paths={LOCK_ROW_SOURCE: source})
                return run_quietly(request, build_engine(files), files)

            def test_should_refuse_the_mount_before_any_container_runs(
                self, step_record: StepRecord
            ) -> None:
                assert (step_record.outputs, step_record.runs) == (None, [])

        class TestWhenTheImageWasBuiltFromAnotherTree:
            @pytest.fixture
            def step_record(self) -> StepRecord:
                files = build_files()
                return run_quietly(REQUEST, build_engine(files, tree_label=OTHER_TREE), files)

            def test_should_run_nothing_and_make_no_directory(
                self, step_record: StepRecord
            ) -> None:
                assert (step_record.runs, OUTPUT_DIRECTORY in step_record.files.entries) == (
                    [],
                    False,
                )

        class TestWhenTheImageIsMissing:
            @pytest.fixture
            def run_act(self) -> Callable[[], StepOutputs]:
                files = build_files()
                engine = build_engine(files)
                engine.images.clear()
                return partial(run_step, REQUEST, engine, files)

            def test_should_ask_for_the_images_to_be_built(
                self, run_act: Callable[[], StepOutputs]
            ) -> None:
                with pytest.raises(StepError, match=re.escape(f"the image {TAG} is missing")):
                    run_act()

        class TestWhenAnAccountHoldsTheContainerUser:
            @pytest.fixture
            def run_act(self) -> Callable[[], StepOutputs]:
                files = build_files()
                host = replace(HOST, accounts=HostAccounts(True, (), ()))
                return partial(run_step, replace(REQUEST, host=host), build_engine(files), files)

            def test_should_refuse_to_run(self, run_act: Callable[[], StepOutputs]) -> None:
                with pytest.raises(AccountRangeError, match="holds the user id 65531"):
                    run_act()

        class TestWhenTheStepFails:
            @pytest.fixture
            def run_act(self) -> Callable[[], StepOutputs]:
                files = build_files()
                return partial(run_step, REQUEST, build_engine(files, status=1), files)

            def test_should_show_the_end_of_its_log_through_the_printable_filter(
                self, run_act: Callable[[], StepOutputs]
            ) -> None:
                with pytest.raises(
                    StepError,
                    match=re.escape(
                        "the step lock exited with the status 1; the end of its log, "
                        "lock/1/logs/lock.log:\nTraceback\n?[31mError\n"
                    ),
                ):
                    run_act()

        @pytest.mark.parametrize(
            ("outputs", "reason"),
            [
                pytest.param(
                    {**GOOD_OUTPUTS, "notes.txt": build_step_output(b"x")},
                    "the step wrote 'notes.txt', which its row does not declare",
                    id="an undeclared entry",
                ),
                pytest.param(
                    {
                        **GOOD_OUTPUTS,
                        "uv.lock": (
                            replace(GOOD_OUTPUTS["uv.lock"][0], permission_bits=0o600),
                            LOCK_CONTENT,
                        ),
                    },
                    "the output uv.lock has the mode 0600, not 0640",
                    id="another mode",
                ),
                pytest.param(
                    {
                        **GOOD_OUTPUTS,
                        "uv.lock": (
                            replace(GOOD_OUTPUTS["uv.lock"][0], size_bytes=16 * 2**20 + 1),
                            LOCK_CONTENT,
                        ),
                    },
                    "the output uv.lock holds 16777217 bytes, beyond its bound of 16777216",
                    id="a file beyond its size bound",
                ),
                pytest.param(
                    {"uv.lock": GOOD_OUTPUTS["uv.lock"]},
                    "the step did not write its output peaks.json",
                    id="a missing output",
                ),
            ],
        )
        class TestWhenTheOutputsBreakTheirRow:
            @pytest.fixture
            def run_act(
                self, outputs: dict[str, tuple[FileFacts, bytes]]
            ) -> Callable[[], StepOutputs]:
                files = build_files()
                return partial(run_step, REQUEST, build_engine(files, outputs), files)

            def test_should_refuse_them_and_say_why(
                self, run_act: Callable[[], StepOutputs], reason: str
            ) -> None:
                with pytest.raises(ContainerFileError, match=re.escape(reason)):
                    run_act()

        class TestWhenTheRowBindsADeclaredOutputOfAnEarlierStep:
            @pytest.fixture
            def step_record(self) -> CompletedStep:
                files = build_files_with_earlier_outputs(EARLIER_FILES)
                engine = build_engine(files)
                return CompletedStep(
                    run_step(EARLIER_OUTPUT_REQUEST, engine, files), engine.runs, files
                )

            def test_should_bind_that_file_alone_read_only_under_the_inputs_of_its_step(
                self, step_record: CompletedStep
            ) -> None:
                assert [
                    mount
                    for run in step_record.runs
                    for mount in run.mounts
                    if mount.host_path.is_relative_to(EARLIER_DIRECTORY)
                ] == [
                    BindMount(
                        host_path=EARLIER_DIRECTORY / NAMES_OUTPUT.name,
                        container_path=PurePosixPath("/inputs/describe/names"),
                        is_read_only=True,
                        is_file=True,
                        is_written_by_a_step=True,
                    )
                ]

        class TestWhenAnEarlierOutputIsALink:
            @pytest.fixture
            def step_record(self) -> StepRecord:
                link = FileFacts(FileKind.LINK, CONTAINER_USER_ID, 0o777, 12, 1)
                files = build_files_with_earlier_outputs(
                    {**EARLIER_FILES, NAMES_OUTPUT.name: (link, b"")}
                )
                return run_quietly(EARLIER_OUTPUT_REQUEST, build_engine(files), files)

            def test_should_refuse_it_before_any_container_runs(
                self, step_record: StepRecord
            ) -> None:
                assert (step_record.outputs, step_record.runs) == (None, [])

        @pytest.mark.parametrize(
            ("earlier_files", "reason"),
            [
                pytest.param(
                    {**EARLIER_FILES, "notes.txt": build_step_output(b"x")},
                    "the step wrote 'notes.txt', which its row does not declare",
                    id="an undeclared entry beside it",
                ),
                pytest.param(
                    {PEAKS_FILE_NAME: EARLIER_FILES[PEAKS_FILE_NAME]},
                    "the step did not write its output names",
                    id="a missing file",
                ),
                pytest.param(
                    {
                        **EARLIER_FILES,
                        NAMES_OUTPUT.name: (
                            FileFacts(FileKind.LINK, CONTAINER_USER_ID, 0o777, 12, 1),
                            b"",
                        ),
                    },
                    "the output names is a link, not a regular file",
                    id="a link",
                ),
                pytest.param(
                    {
                        **EARLIER_FILES,
                        NAMES_OUTPUT.name: (
                            FileFacts(FileKind.DIRECTORY, CONTAINER_USER_ID, 0o750, 4096, 2),
                            b"",
                        ),
                    },
                    "the output names is a directory, not a regular file",
                    id="a directory in place of the file",
                ),
                pytest.param(
                    {
                        **EARLIER_FILES,
                        NAMES_OUTPUT.name: (
                            replace(build_step_output(NAMES_CONTENT)[0], permission_bits=0o644),
                            NAMES_CONTENT,
                        ),
                    },
                    "the output names has the mode 0644, not 0640",
                    id="another mode",
                ),
                pytest.param(
                    {
                        **EARLIER_FILES,
                        NAMES_OUTPUT.name: (
                            replace(build_step_output(NAMES_CONTENT)[0], owner_id=USER_ID),
                            NAMES_CONTENT,
                        ),
                    },
                    f"the output names is owned by the user id {USER_ID}, not by 65531",
                    id="another owner",
                ),
                pytest.param(
                    {
                        **EARLIER_FILES,
                        NAMES_OUTPUT.name: (
                            replace(build_step_output(NAMES_CONTENT)[0], size_bytes=1025),
                            NAMES_CONTENT,
                        ),
                    },
                    "the output names holds 1025 bytes, beyond its bound of 1024",
                    id="a file beyond its size bound",
                ),
                pytest.param(
                    {
                        **EARLIER_FILES,
                        NAMES_OUTPUT.name: (
                            replace(build_step_output(NAMES_CONTENT)[0], link_count=2),
                            NAMES_CONTENT,
                        ),
                    },
                    "the output names has 2 links, not one",
                    id="several links",
                ),
            ],
        )
        class TestWhenAnEarlierOutputBreaksTheRowOfItsStep:
            @pytest.fixture
            def run_act(
                self, earlier_files: dict[str, tuple[FileFacts, bytes]]
            ) -> Callable[[], StepOutputs]:
                files = build_files_with_earlier_outputs(earlier_files)
                return partial(run_step, EARLIER_OUTPUT_REQUEST, build_engine(files), files)

            def test_should_refuse_it_and_say_why(
                self, run_act: Callable[[], StepOutputs], reason: str
            ) -> None:
                with pytest.raises(ContainerFileError, match=re.escape(reason)):
                    run_act()

        @pytest.mark.parametrize(
            ("directory_facts_of_the_earlier_step", "reason"),
            [
                pytest.param(
                    None,
                    f"{WORK_ROOT / EARLIER_DIRECTORY} is missing, where a directory is declared",
                    id="an earlier step that has not run",
                ),
                pytest.param(
                    FileFacts(FileKind.LINK, USER_ID, 0o777, 12, 1),
                    "runs/family/c0ffee/describe is a link, where a directory is declared",
                    id="a link in place of its directory",
                ),
                pytest.param(
                    build_directory_facts(0o2770, owner_id=CONTAINER_USER_ID),
                    "runs/family/c0ffee/describe is owned by the user id 65531, not by 1000",
                    id="a directory that a container made",
                ),
            ],
        )
        class TestWhenTheDirectoryOfAnEarlierStepIsNotTheOneTheHostMade:
            @pytest.fixture
            def run_act(
                self, directory_facts_of_the_earlier_step: FileFacts | None
            ) -> Callable[[], StepOutputs]:
                files = build_files()
                files.add_directory(WORK_ROOT / EARLIER_DIRECTORY.parent)
                if directory_facts_of_the_earlier_step is not None:
                    files.add_directory(
                        WORK_ROOT / EARLIER_DIRECTORY, directory_facts_of_the_earlier_step
                    )
                return partial(run_step, EARLIER_OUTPUT_REQUEST, build_engine(files), files)

            def test_should_refuse_it_and_say_why(
                self, run_act: Callable[[], StepOutputs], reason: str
            ) -> None:
                with pytest.raises(ContainerFileError, match=re.escape(reason)):
                    run_act()
