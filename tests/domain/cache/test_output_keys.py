import re
from collections.abc import Callable
from dataclasses import replace
from functools import partial
from pathlib import PurePosixPath

import pytest

from mostt_model_builder.domain.cache.hardware_facts import HardwareFacts
from mostt_model_builder.domain.cache.output_keys import (
    OutputKeyError,
    PlainData,
    RowKeyInputs,
    compute_key,
    document_candidate_key,
    document_row_key,
    name_image_extra,
    select_image_files,
)
from mostt_model_builder.domain.containers.base_images import ENVIRONMENT_PYTHON
from mostt_model_builder.domain.containers.container_steps import (
    BENCH_IMAGE,
    PEAKS_OUTPUT,
    ContainerImage,
    ContainerRun,
    DeclaredOutput,
    EarlierOutput,
    SourceReference,
    StepLimits,
    StepPlacement,
    StepRow,
    plan_container_run,
)
from mostt_model_builder.domain.containers.devices import Device
from mostt_model_builder.domain.contract.recipe_contract import (
    HostContentOrigin,
    JudgedCandidateFile,
    StepOutputOrigin,
)
from mostt_model_builder.domain.formats.file_formats import FileFormat
from mostt_model_builder.domain.lock.extra_closure import LockedDistribution, LockedEntry
from mostt_model_builder.domain.sources.source_pins import RepositoryType, SourcePin

REVISION = "0123456789abcdef0123456789abcdef01234567"
ULTRA_PIN = SourcePin(
    RepositoryType.MODEL, "maker/weights", REVISION, "model.safetensors", 10, "a" * 64
)
UNBOUND_PIN = SourcePin(RepositoryType.MODEL, "maker/weights", REVISION, "other.json", 10, "b" * 64)
IMAGE = ContainerImage("family", "transcribe-gpu", Device.GPU, ENVIRONMENT_PYTHON)
EXPORT_OUTPUT = DeclaredOutput("encoder.onnx", 2**20)
ROW = StepRow(
    name="transcribe-float32-gpu-fr_fr-0",
    image=IMAGE,
    entry_module="mostt_recipe_family.steps.transcription",
    sources=(SourceReference(ULTRA_PIN.repository, ULTRA_PIN.path),),
    has_request=True,
    outputs=(DeclaredOutput("lines.jsonl", 2**20), PEAKS_OUTPUT),
    limits=StepLimits(memory_mib=4096, process_count=64, cpu_count=8, temporary_directory_mib=256),
    uses_gpu=True,
    earlier_outputs=(EarlierOutput("export", EXPORT_OUTPUT, ("encoder.onnx", "peaks.json")),),
    candidate_files=("encoder.onnx", "vocab.txt"),
)
PLACEMENT = StepPlacement(
    output_directory=PurePosixPath("cache/family/row/output"),
    request_directory=PurePosixPath("cache/family/row/request"),
    log_path=PurePosixPath("cache/family/row/step.log"),
    candidate_directory=PurePosixPath("cache/family/candidate/files"),
)
WORK_ROOT = PurePosixPath("/data/builds/work")
IMAGE_ID = "sha256:" + "c" * 64
GROUP_ID = 1000


def plan_run(
    row: StepRow = ROW,
    placement: StepPlacement = PLACEMENT,
    work_root: PurePosixPath = WORK_ROOT,
    image_id: str = IMAGE_ID,
    group_id: int = GROUP_ID,
) -> ContainerRun:
    return plan_container_run(
        row,
        image_id,
        placement,
        {source: PurePosixPath("sources", source.path) for source in row.sources},
        work_root,
        group_id,
        {"export": PurePosixPath("cache/family/export/output")},
    )


GPU_FACTS = HardwareFacts(("AMD Ryzen 9 7950X 16-Core Processor",), ("RTX 4090",), "550.54.14")
INPUTS = RowKeyInputs(
    row=ROW,
    planned_run=plan_run(),
    request_files={"shard.json": b'{"run": "float32-gpu"}'},
    image_files={PurePosixPath("recipes/family/images/transcribe-gpu/Containerfile"): b"FROM x\n"},
    image_packages=(
        LockedEntry("onnx-asr", "0.12.0", (LockedDistribution("https://f/onnx_asr.whl", "h"),)),
    ),
    bootstrap=b"import sys\n",
    code_files={PurePosixPath("recipes/family/src/mostt_recipe_family/steps/t.py"): b"x = 1\n"},
    pins=(ULTRA_PIN, UNBOUND_PIN),
    earlier_output_digests={"export/encoder.onnx": "d" * 64, "export/peaks.json": "e" * 64},
    candidate_digests={"encoder.onnx": "1" * 64, "vocab.txt": "2" * 64, "other.onnx": "3" * 64},
    hardware=GPU_FACTS,
)


def replan(**fields: object) -> RowKeyInputs:
    planned_run = replace(INPUTS.planned_run, **fields)  # type: ignore[arg-type]  # the test feeds each field a value of its own type
    return replace(INPUTS, planned_run=planned_run)


CHANGED_COMPONENTS = {
    "a file of the step's import closure": replace(
        INPUTS, code_files={path: b"x = 2\n" for path in INPUTS.code_files}
    ),
    "the bootstrap": replace(INPUTS, bootstrap=b"import os\n"),
    "a file of the image's directory": replace(
        INPUTS, image_files={path: b"FROM y\n" for path in INPUTS.image_files}
    ),
    "a package of the image's extra": replace(
        INPUTS, image_packages=(LockedEntry("onnx-asr", "0.13.0", ()),)
    ),
    "the pin of a bound source": replace(
        INPUTS, pins=(replace(ULTRA_PIN, sha256="f" * 64), UNBOUND_PIN)
    ),
    "the content of a bound earlier output": replace(
        INPUTS, earlier_output_digests={"export/encoder.onnx": "0" * 64}
    ),
    "the content of a bound file of the candidate": replace(
        INPUTS,
        candidate_digests={**INPUTS.candidate_digests, "vocab.txt": "9" * 64},
    ),
    "a request file": replace(INPUTS, request_files={"shard.json": b'{"run": "other"}'}),
    "a limit of the row": replace(
        INPUTS, row=replace(ROW, limits=replace(ROW.limits, memory_mib=8192))
    ),
    "the planned run's environment": replan(environment=(("HOME", "/tmp/other"),)),
    "the planned run's Python arguments": replan(entry_module="mostt_recipe_family.steps.other"),
    "the planned run's temporary filesystems": replan(temporary_filesystems=()),
    "the driver's version for a row on the GPU": replace(
        INPUTS, hardware=replace(GPU_FACTS, driver_version="570.133.07")
    ),
    "the model of the CPU": replace(INPUTS, hardware=replace(GPU_FACTS, cpu_models=("Xeon",))),
}
UNCHANGED_COMPONENTS = {
    "the image's ID": replace(INPUTS, planned_run=plan_run(image_id="sha256:" + "d" * 64)),
    "the work directory": replace(
        INPUTS, planned_run=plan_run(work_root=PurePosixPath("/srv/other"))
    ),
    "the host's paths": replace(
        INPUTS,
        planned_run=plan_run(
            placement=replace(PLACEMENT, output_directory=PurePosixPath("elsewhere/output"))
        ),
    ),
    "the user's group": replace(INPUTS, planned_run=plan_run(group_id=4242)),
    "an earlier output the row does not bind": replace(
        INPUTS,
        earlier_output_digests={**INPUTS.earlier_output_digests, "export/peaks.json": "7" * 64},
    ),
    "a pin of a source the row does not bind": replace(
        INPUTS, pins=(ULTRA_PIN, replace(UNBOUND_PIN, sha256="8" * 64))
    ),
    "a file of the candidate the row does not bind": replace(
        INPUTS, candidate_digests={**INPUTS.candidate_digests, "other.onnx": "6" * 64}
    ),
}
REFUSED_INPUTS = {
    "a row on the GPU without a GPU": (
        replace(INPUTS, hardware=HardwareFacts(GPU_FACTS.cpu_models)),
        "transcribe-float32-gpu-fr_fr-0: /proc/driver/nvidia holds no GPU, which the key of a "
        "run on the GPU needs",
    ),
    "a row on the GPU without hardware facts": (
        replace(INPUTS, hardware=None),
        "/proc/driver/nvidia holds no GPU",
    ),
    "a bound earlier output without its digest": (
        replace(INPUTS, earlier_output_digests={}),
        "needs the SHA-256 of the earlier output export/encoder.onnx, which it was not given",
    ),
    "a bound source without its pin": (
        replace(INPUTS, pins=(UNBOUND_PIN,)),
        "binds model.safetensors of maker/weights, which no pin gives",
    ),
}
VOCABULARY = JudgedCandidateFile(
    "vocab.txt", FileFormat.TEXT, ("float32", "int8"), StepOutputOrigin("export", "vocab.txt")
)
CONFIGURATION = JudgedCandidateFile(
    "config.json", FileFormat.TEXT, ("float32", "int8"), HostContentOrigin(b"{}")
)
CANDIDATE = (VOCABULARY, CONFIGURATION)
ORIGIN_DIGESTS = {"export/vocab.txt": "a" * 64, "convert-int8/encoder.int8.onnx": "b" * 64}


class TestTheOutputKeys:
    class TestTheComputeKeyFunction:
        class TestWhenTwoDocumentsDifferOnlyInTheOrderOfTheirMappings:
            @pytest.fixture
            def keys(self) -> tuple[str, str]:
                document: dict[str, PlainData] = {"b": 1, "a": {"y": [1, 2], "x": "z"}}
                reordered: dict[str, PlainData] = {"a": {"x": "z", "y": [1, 2]}, "b": 1}
                return compute_key(document), compute_key(reordered)

            def test_should_give_both_the_same_key(self, keys: tuple[str, str]) -> None:
                assert keys[0] == keys[1]

    class TestTheDocumentRowKeyFunction:
        @pytest.mark.parametrize(
            "changed_inputs",
            list(CHANGED_COMPONENTS.values()),
            ids=list(CHANGED_COMPONENTS.keys()),
        )
        class TestWhenAComponentThatDeterminesTheOutputsChanges:
            @pytest.fixture
            def keys(self, changed_inputs: RowKeyInputs) -> tuple[str, str]:
                return (
                    compute_key(document_row_key(INPUTS)),
                    compute_key(document_row_key(changed_inputs)),
                )

            def test_should_change_the_key(self, keys: tuple[str, str]) -> None:
                assert keys[0] != keys[1]

        @pytest.mark.parametrize(
            "changed_inputs",
            list(UNCHANGED_COMPONENTS.values()),
            ids=list(UNCHANGED_COMPONENTS.keys()),
        )
        class TestWhenOnlyTheHostOrWhatTheRowDoesNotBindChanges:
            @pytest.fixture
            def keys(self, changed_inputs: RowKeyInputs) -> tuple[str, str]:
                return (
                    compute_key(document_row_key(INPUTS)),
                    compute_key(document_row_key(changed_inputs)),
                )

            def test_should_keep_the_key_so_that_a_copied_work_directory_reuses_it(
                self, keys: tuple[str, str]
            ) -> None:
                assert keys[0] == keys[1]

        class TestWhenTheRowIsNoRun:
            @pytest.fixture
            def document(self) -> dict[str, PlainData]:
                row = replace(ROW, uses_gpu=False, image=BENCH_IMAGE, candidate_files=())
                return document_row_key(
                    replace(INPUTS, row=row, planned_run=plan_run(row), hardware=None)
                )

            def test_should_hold_no_hardware(self, document: dict[str, PlainData]) -> None:
                assert "hardware" not in document

        class TestWhenTheRowRunsOnTheCpu:
            @pytest.fixture
            def document(self) -> dict[str, PlainData]:
                row = replace(ROW, uses_gpu=False)
                return document_row_key(replace(INPUTS, row=row, planned_run=plan_run(row)))

            def test_should_hold_the_model_of_the_cpu_alone(
                self, document: dict[str, PlainData]
            ) -> None:
                assert document["hardware"] == {"cpu_models": list(GPU_FACTS.cpu_models)}

        @pytest.mark.parametrize(
            ("refused_inputs", "reason"),
            list(REFUSED_INPUTS.values()),
            ids=list(REFUSED_INPUTS.keys()),
        )
        class TestWhenAComponentIsMissing:
            @pytest.fixture
            def document_act(self, refused_inputs: RowKeyInputs) -> Callable[[], object]:
                return partial(document_row_key, refused_inputs)

            def test_should_refuse_a_key_that_would_miss_it(
                self, document_act: Callable[[], object], reason: str
            ) -> None:
                with pytest.raises(OutputKeyError, match=re.escape(reason)):
                    document_act()

    class TestTheDocumentCandidateKeyFunction:
        @pytest.mark.parametrize(
            ("changed_candidate", "changed_digests", "is_changed"),
            [
                pytest.param(CANDIDATE, ORIGIN_DIGESTS, False, id="nothing"),
                pytest.param(
                    CANDIDATE,
                    {**ORIGIN_DIGESTS, "convert-int8/encoder.int8.onnx": "c" * 64},
                    False,
                    id="an output that no file of the candidate copies",
                ),
                pytest.param(
                    CANDIDATE,
                    {**ORIGIN_DIGESTS, "export/vocab.txt": "c" * 64},
                    True,
                    id="the content of the output a file copies",
                ),
                pytest.param(
                    (replace(VOCABULARY, variants=("float32",)), CONFIGURATION),
                    ORIGIN_DIGESTS,
                    True,
                    id="the variants a file serves",
                ),
                pytest.param(
                    (VOCABULARY, replace(CONFIGURATION, origin=HostContentOrigin(b"[]"))),
                    ORIGIN_DIGESTS,
                    True,
                    id="a content the host writes",
                ),
                pytest.param((VOCABULARY,), ORIGIN_DIGESTS, True, id="a file left out"),
            ],
        )
        class TestWhenTheCandidateChanges:
            @pytest.fixture
            def keys(
                self,
                changed_candidate: tuple[JudgedCandidateFile, ...],
                changed_digests: dict[str, str],
            ) -> tuple[str, str]:
                return (
                    compute_key(document_candidate_key(CANDIDATE, ORIGIN_DIGESTS)),
                    compute_key(document_candidate_key(changed_candidate, changed_digests)),
                )

            def test_should_change_the_key_exactly_when_a_file_it_holds_changes(
                self, keys: tuple[str, str], is_changed: bool
            ) -> None:
                assert (keys[0] != keys[1]) is is_changed

    class TestTheNameImageExtraFunction:
        @pytest.mark.parametrize(
            ("image", "extra"),
            [
                pytest.param(IMAGE, "family-transcribe-gpu", id="an image with an environment"),
                pytest.param(BENCH_IMAGE, None, id="an image on the base's Python alone"),
            ],
        )
        class TestWhenGivenAnImage:
            @pytest.fixture
            def named_extra(self, image: ContainerImage) -> str | None:
                return name_image_extra(image)

            def test_should_name_the_extra_its_environment_installs(
                self, named_extra: str | None, extra: str | None
            ) -> None:
                assert named_extra == extra

    class TestTheSelectImageFilesFunction:
        class TestWhenTheTreeHoldsOtherImages:
            @pytest.fixture
            def image_files(self) -> dict[PurePosixPath, bytes]:
                return select_image_files(
                    IMAGE,
                    {
                        PurePosixPath("recipes/family/images/transcribe-gpu/Containerfile"): b"a",
                        PurePosixPath("recipes/family/images/transcribe-gpu/hashes"): b"b",
                        PurePosixPath("recipes/family/images/transcribe-cpu/Containerfile"): b"c",
                        PurePosixPath("images/bench/Containerfile"): b"d",
                    },
                )

            def test_should_keep_the_files_of_the_images_own_directory(
                self, image_files: dict[PurePosixPath, bytes]
            ) -> None:
                assert sorted(map(str, image_files)) == [
                    "recipes/family/images/transcribe-gpu/Containerfile",
                    "recipes/family/images/transcribe-gpu/hashes",
                ]
