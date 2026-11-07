from dataclasses import replace
from pathlib import PurePosixPath

import pytest

from mostt_model_builder.domain.containers.container_steps import (
    ContainerImage,
    DeclaredOutput,
    EarlierOutput,
    SourceReference,
    StepLimits,
)
from mostt_model_builder.domain.containers.devices import Device
from mostt_model_builder.domain.contract.derived_rows import (
    DerivedRow,
    DeviceRow,
    ShardedRun,
    StepTemplate,
    derive_rows,
)
from mostt_model_builder.domain.contract.shard_request import (
    ShardRequest,
)

RECIPE_PYTHON = PurePosixPath("/opt/venv/bin/python")
CPU_IMAGE = ContainerImage("parakeet-tdt", "bench-cpu", Device.CPU, RECIPE_PYTHON)
GPU_IMAGE = ContainerImage("parakeet-tdt", "bench-gpu", Device.GPU, RECIPE_PYTHON)
CPU_LIMITS = StepLimits(memory_mib=8192, process_count=64, cpu_count=8, temporary_directory_mib=256)
GPU_LIMITS = StepLimits(
    memory_mib=16_384, process_count=128, cpu_count=4, temporary_directory_mib=512, scratch_mib=2048
)
CPU_OUTPUT = DeclaredOutput("threads.json", 1024)
GPU_OUTPUT = DeclaredOutput("placement.json", 4 * 2**20)
STEP_OUTPUT = DeclaredOutput("lines.jsonl", 16 * 2**20)
FLEURS_SOURCE = SourceReference("google/fleurs", "data/fr_fr/test.tsv")
ENCODER_OUTPUT = DeclaredOutput("encoder.onnx", 2**30)
CONVERTED_ENCODER = EarlierOutput("convert", ENCODER_OUTPUT, ("encoder.onnx",))
ENTRY_MODULE = "mostt_recipe_parakeet_tdt.steps.bench"
TEMPLATE = StepTemplate(
    name="bench",
    entry_module=ENTRY_MODULE,
    sources=(FLEURS_SOURCE,),
    earlier_outputs=(CONVERTED_ENCODER,),
    binds_candidate=True,
    outputs=(STEP_OUTPUT,),
    devices={
        Device.CPU: DeviceRow(CPU_IMAGE, CPU_LIMITS, (CPU_OUTPUT,)),
        Device.GPU: DeviceRow(GPU_IMAGE, GPU_LIMITS, (GPU_OUTPUT,)),
    },
)
BOTH_LANGUAGES = ("fr_fr", "en_us")
INT8_FILES = ("encoder.int8.onnx", "vocab.txt")
FLOAT16_FILES = ("encoder.fp16.onnx", "vocab.txt")
CPU_RUN = ShardedRun("int8-cpu", "int8", Device.CPU, BOTH_LANGUAGES, 2, INT8_FILES)
GPU_RUN = ShardedRun("fp16-gpu", "fp16", Device.GPU, ("fr_fr",), 1, FLOAT16_FILES)
SINGLE_SHARD_RUN = ShardedRun("nemo", "reference", Device.CPU, ("fr_fr",), 1, ())
FRENCH_BENCH = (
    SourceReference("google/fleurs", "data/fr_fr/test.tsv"),
    SourceReference("google/fleurs", "data/fr_fr/audio/test.tar.gz"),
)
ENGLISH_BENCH = (
    SourceReference("google/fleurs", "data/en_us/test.tsv"),
    SourceReference("google/fleurs", "data/en_us/audio/test.tar.gz"),
)


class TestTheDerivedRows:
    class TestTheDeriveRowsFunction:
        class TestWhenTheRunsReadSeveralLanguagesAndShards:
            @pytest.fixture
            def derived_rows(self) -> tuple[DerivedRow, ...]:
                return derive_rows(TEMPLATE, (CPU_RUN, GPU_RUN))

            def test_should_name_each_row_after_its_step_its_run_its_language_and_its_shard(
                self, derived_rows: tuple[DerivedRow, ...]
            ) -> None:
                assert [derived.row.name for derived in derived_rows] == [
                    "bench-int8-cpu-fr_fr-0",
                    "bench-int8-cpu-fr_fr-1",
                    "bench-int8-cpu-en_us-0",
                    "bench-int8-cpu-en_us-1",
                    "bench-fp16-gpu-fr_fr-0",
                ]

            def test_should_keep_the_step_the_run_and_the_language_of_each_row(
                self, derived_rows: tuple[DerivedRow, ...]
            ) -> None:
                assert [
                    (derived.step, derived.run, derived.language) for derived in derived_rows
                ] == [
                    ("bench", "int8-cpu", "fr_fr"),
                    ("bench", "int8-cpu", "fr_fr"),
                    ("bench", "int8-cpu", "en_us"),
                    ("bench", "int8-cpu", "en_us"),
                    ("bench", "fp16-gpu", "fr_fr"),
                ]

            def test_should_give_each_row_the_image_of_its_runs_device(
                self, derived_rows: tuple[DerivedRow, ...]
            ) -> None:
                assert [derived.row.image for derived in derived_rows] == [
                    CPU_IMAGE,
                    CPU_IMAGE,
                    CPU_IMAGE,
                    CPU_IMAGE,
                    GPU_IMAGE,
                ]

            def test_should_give_each_row_the_limits_of_its_runs_device(
                self, derived_rows: tuple[DerivedRow, ...]
            ) -> None:
                assert [derived.row.limits for derived in derived_rows] == [
                    CPU_LIMITS,
                    CPU_LIMITS,
                    CPU_LIMITS,
                    CPU_LIMITS,
                    GPU_LIMITS,
                ]

            def test_should_declare_the_outputs_of_the_device_before_those_of_the_step(
                self, derived_rows: tuple[DerivedRow, ...]
            ) -> None:
                assert [derived.row.outputs for derived in derived_rows] == [
                    (CPU_OUTPUT, STEP_OUTPUT),
                    (CPU_OUTPUT, STEP_OUTPUT),
                    (CPU_OUTPUT, STEP_OUTPUT),
                    (CPU_OUTPUT, STEP_OUTPUT),
                    (GPU_OUTPUT, STEP_OUTPUT),
                ]

            def test_should_give_the_gpu_to_the_rows_of_a_gpu_run_only(
                self, derived_rows: tuple[DerivedRow, ...]
            ) -> None:
                assert [derived.row.uses_gpu for derived in derived_rows] == [
                    False,
                    False,
                    False,
                    False,
                    True,
                ]

            def test_should_give_every_row_a_request_and_no_network(
                self, derived_rows: tuple[DerivedRow, ...]
            ) -> None:
                assert {
                    (derived.row.has_request, derived.row.has_network) for derived in derived_rows
                } == {(True, False)}

            def test_should_give_every_row_the_entry_sources_and_inputs_of_its_step(
                self, derived_rows: tuple[DerivedRow, ...]
            ) -> None:
                assert {
                    (derived.row.entry_module, derived.row.sources, derived.row.earlier_outputs)
                    for derived in derived_rows
                } == {(ENTRY_MODULE, (FLEURS_SOURCE,), (CONVERTED_ENCODER,))}

            def test_should_bind_each_row_to_the_candidate_files_of_its_variant_alone(
                self, derived_rows: tuple[DerivedRow, ...]
            ) -> None:
                assert [
                    (derived.row.binds_candidate, derived.row.candidate_files)
                    for derived in derived_rows
                ] == [
                    (False, INT8_FILES),
                    (False, INT8_FILES),
                    (False, INT8_FILES),
                    (False, INT8_FILES),
                    (False, FLOAT16_FILES),
                ]

            def test_should_request_of_each_shard_its_run_variant_device_language_and_place(
                self, derived_rows: tuple[DerivedRow, ...]
            ) -> None:
                assert [derived.request for derived in derived_rows] == [
                    ShardRequest("int8-cpu", "int8", Device.CPU, ("fr_fr",), 0, 2, INT8_FILES),
                    ShardRequest("int8-cpu", "int8", Device.CPU, ("fr_fr",), 1, 2, INT8_FILES),
                    ShardRequest("int8-cpu", "int8", Device.CPU, ("en_us",), 0, 2, INT8_FILES),
                    ShardRequest("int8-cpu", "int8", Device.CPU, ("en_us",), 1, 2, INT8_FILES),
                    ShardRequest("fp16-gpu", "fp16", Device.GPU, ("fr_fr",), 0, 1, FLOAT16_FILES),
                ]

            def test_should_bind_no_bench_beyond_the_sources_of_a_step_that_binds_none(
                self, derived_rows: tuple[DerivedRow, ...]
            ) -> None:
                assert {derived.row.sources for derived in derived_rows} == {(FLEURS_SOURCE,)}

        class TestWhenTheRunsMakeASingleRow:
            @pytest.fixture
            def derived_rows(self) -> tuple[DerivedRow, ...]:
                return derive_rows(TEMPLATE, (SINGLE_SHARD_RUN,))

            def test_should_name_it_as_any_other_since_no_later_step_binds_a_runs_outputs(
                self, derived_rows: tuple[DerivedRow, ...]
            ) -> None:
                assert [derived.row.name for derived in derived_rows] == ["bench-nemo-fr_fr-0"]

        class TestWhenTheStepBindsTheBenchOfItsRowsLanguage:
            @pytest.fixture
            def derived_rows(self) -> tuple[DerivedRow, ...]:
                return derive_rows(replace(TEMPLATE, binds_bench=True), (CPU_RUN,))

            def test_should_bind_each_row_to_its_own_languages_bench_after_the_steps_sources(
                self, derived_rows: tuple[DerivedRow, ...]
            ) -> None:
                assert [(derived.language, derived.row.sources) for derived in derived_rows] == [
                    ("fr_fr", (FLEURS_SOURCE, *FRENCH_BENCH)),
                    ("fr_fr", (FLEURS_SOURCE, *FRENCH_BENCH)),
                    ("en_us", (FLEURS_SOURCE, *ENGLISH_BENCH)),
                    ("en_us", (FLEURS_SOURCE, *ENGLISH_BENCH)),
                ]

        class TestWhenTheStepBindsNoCandidate:
            @pytest.fixture
            def derived_rows(self) -> tuple[DerivedRow, ...]:
                return derive_rows(replace(TEMPLATE, binds_candidate=False), (CPU_RUN,))

            def test_should_bind_no_file_of_the_candidate_and_request_none(
                self, derived_rows: tuple[DerivedRow, ...]
            ) -> None:
                assert {
                    (derived.row.candidate_files, derived.request.files) for derived in derived_rows
                } == {((), ())}

        class TestWhenNoRunIsGiven:
            @pytest.fixture
            def derived_rows(self) -> tuple[DerivedRow, ...]:
                return derive_rows(TEMPLATE, ())

            def test_should_derive_no_row(self, derived_rows: tuple[DerivedRow, ...]) -> None:
                assert derived_rows == ()
