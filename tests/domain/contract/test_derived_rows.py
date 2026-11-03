import json
import random
import re
from collections.abc import Callable
from functools import partial
from pathlib import PurePosixPath

import pytest

from mostt_model_builder.domain.containers.container_steps import (
    ContainerImage,
    DeclaredOutput,
    Device,
    EarlierOutput,
    SourceReference,
    StepLimits,
)
from mostt_model_builder.domain.contract.derived_rows import (
    MAX_SHARD_COUNT,
    SHARD_REQUEST_MAX_SIZE_BYTES,
    DerivedRow,
    DeviceRow,
    ShardedRun,
    ShardRequest,
    ShardRequestError,
    StepTemplate,
    derive_rows,
    parse_shard_request,
    render_shard_request,
    select_shard,
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
CPU_RUN = ShardedRun("int8-cpu", "int8", Device.CPU, BOTH_LANGUAGES, 3)
GPU_RUN = ShardedRun("fp16-gpu", "fp16", Device.GPU, ("fr_fr",), 1)
SINGLE_SHARD_RUN = ShardedRun("int8-cpu", "int8", Device.CPU, BOTH_LANGUAGES, 1)
REQUEST = ShardRequest("int8-cpu", "int8", Device.CPU, BOTH_LANGUAGES, 1, 3)
ACCEPTED_REQUESTS = {
    "the only shard of a run": ShardRequest("int8-cpu", "int8", Device.CPU, ("en_us",), 0, 1),
    "the last shard of the most shards": ShardRequest(
        "fp16-gpu", "fp16", Device.GPU, ("fr_fr",), MAX_SHARD_COUNT - 1, MAX_SHARD_COUNT
    ),
    "a run and a variant of 32 characters": ShardRequest(
        "r" * 32, "v" * 32, Device.CPU, BOTH_LANGUAGES, 0, 1
    ),
}
SHARD_REQUEST_FIELDS: dict[str, object] = {
    "run": "int8-cpu",
    "variant": "int8",
    "device": "cpu",
    "languages": ["fr_fr", "en_us"],
    "shard": 1,
    "shard_count": 3,
}


def build_request(**changes: object) -> bytes:
    return json.dumps(SHARD_REQUEST_FIELDS | changes).encode("ascii")


def build_request_without(key: str) -> bytes:
    remaining_fields = {name: field for name, field in SHARD_REQUEST_FIELDS.items() if name != key}
    return json.dumps(remaining_fields).encode("ascii")


REFUSED_REQUESTS = {
    "a document nested 3 000 deep": (b"[" * 3000, "the document nests deeper than 32 levels"),
    "more than 4 KiB": (
        b" " * (SHARD_REQUEST_MAX_SIZE_BYTES + 1),
        f"the shard's request holds {SHARD_REQUEST_MAX_SIZE_BYTES + 1} bytes, too many",
    ),
    "not json": (b"{", "the shard's request is not ASCII JSON"),
    "a key twice": (
        b'{"run": "a", "run": "b"}',
        "the shard's request is not ASCII JSON: a key appears twice in an object",
    ),
    "bytes beyond ascii": (
        build_request().replace(b"int8-cpu", b"int8-cpu\xc3\xa9"),
        "the shard's request is not ASCII JSON",
    ),
    "a list": (b"[]", "the shard's request is not an object of"),
    "a missing key": (build_request_without("languages"), "is not an object of"),
    "an unknown key": (build_request(command="rm"), "is not an object of"),
    "a run outside the grammar": (
        build_request(run="Int8-CPU"),
        "its run, 'Int8-CPU', does not match [a-z0-9][a-z0-9-]{0,31}",
    ),
    "a run that is a path": (build_request(run="../int8"), "its run, '../int8', does not match"),
    "a run beyond 32 characters": (build_request(run="r" * 33), "its run, 'rrrr"),
    "a run that is not a string": (build_request(run=8), "its run, 8, does not match"),
    "a variant that starts with a dash": (
        build_request(variant="-int8"),
        "its variant, '-int8', does not match [a-z0-9][a-z0-9-]{0,31}",
    ),
    "a device neither cpu nor gpu": (
        build_request(device="tpu"),
        "its device, 'tpu', is neither cpu nor gpu",
    ),
    "a device in capitals": (build_request(device="GPU"), "its device, 'GPU', is neither"),
    "no language": (
        build_request(languages=[]),
        "its languages, [], are not distinct languages of the bench",
    ),
    "a language outside the bench": (
        build_request(languages=["fr_fr", "de_de"]),
        "its languages, ['fr_fr', 'de_de'], are not distinct languages of the bench",
    ),
    "a language twice": (
        build_request(languages=["fr_fr", "fr_fr"]),
        "its languages, ['fr_fr', 'fr_fr'], are not distinct",
    ),
    "languages written as one string": (
        build_request(languages="fr_fr"),
        "its languages, 'fr_fr', are not distinct",
    ),
    "a language that is a list": (
        build_request(languages=[["fr_fr"]]),
        "its languages, [['fr_fr']], are not distinct",
    ),
    "no shard": (build_request(shard_count=0), "its count of shards, 0, lies outside 1 to 16"),
    "more shards than the bound": (
        build_request(shard_count=MAX_SHARD_COUNT + 1),
        "its count of shards, 17, lies outside 1 to 16",
    ),
    "a boolean count of shards": (
        build_request(shard_count=True),
        "its count of shards, True, lies outside 1 to 16",
    ),
    "a fractional count of shards": (
        build_request(shard_count=2.0),
        "its count of shards, 2.0, lies outside 1 to 16",
    ),
    "a negative shard": (build_request(shard=-1), "its shard, -1, lies outside 0 to 2"),
    "a shard beyond its count": (build_request(shard=3), "its shard, 3, lies outside 0 to 2"),
    "a boolean shard": (build_request(shard=False), "its shard, False, lies outside 0 to 2"),
}
PROPERTY_SEED = 20_261_103
PROPERTY_CASE_COUNT = 500
MAX_ENTRY_COUNT = 100


def build_sharding_cases(generator: random.Random) -> list[tuple[int, int]]:
    return [
        (generator.randint(0, MAX_ENTRY_COUNT), generator.randint(1, MAX_SHARD_COUNT))
        for _ in range(PROPERTY_CASE_COUNT)
    ]


class TestTheDerivedRows:
    class TestTheDeriveRowsFunction:
        class TestWhenTheRunsTotalSeveralShards:
            @pytest.fixture
            def derived_rows(self) -> tuple[DerivedRow, ...]:
                return derive_rows(TEMPLATE, (CPU_RUN, GPU_RUN))

            def test_should_name_each_row_after_its_step_its_run_and_its_shard(
                self, derived_rows: tuple[DerivedRow, ...]
            ) -> None:
                assert [derived.row.name for derived in derived_rows] == [
                    "bench-int8-cpu-0",
                    "bench-int8-cpu-1",
                    "bench-int8-cpu-2",
                    "bench-fp16-gpu-0",
                ]

            def test_should_keep_the_step_and_the_run_of_each_row(
                self, derived_rows: tuple[DerivedRow, ...]
            ) -> None:
                assert [(derived.step, derived.run) for derived in derived_rows] == [
                    ("bench", "int8-cpu"),
                    ("bench", "int8-cpu"),
                    ("bench", "int8-cpu"),
                    ("bench", "fp16-gpu"),
                ]

            def test_should_give_each_row_the_image_of_its_runs_device(
                self, derived_rows: tuple[DerivedRow, ...]
            ) -> None:
                assert [derived.row.image for derived in derived_rows] == [
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
                    GPU_LIMITS,
                ]

            def test_should_declare_the_outputs_of_the_device_before_those_of_the_step(
                self, derived_rows: tuple[DerivedRow, ...]
            ) -> None:
                assert [derived.row.outputs for derived in derived_rows] == [
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
                    True,
                ]

            def test_should_give_every_row_a_request_and_no_network(
                self, derived_rows: tuple[DerivedRow, ...]
            ) -> None:
                assert {
                    (derived.row.has_request, derived.row.has_network) for derived in derived_rows
                } == {(True, False)}

            def test_should_give_every_row_the_entry_sources_inputs_and_candidate_of_its_step(
                self, derived_rows: tuple[DerivedRow, ...]
            ) -> None:
                assert {
                    (
                        derived.row.entry_module,
                        derived.row.sources,
                        derived.row.earlier_outputs,
                        derived.row.binds_candidate,
                    )
                    for derived in derived_rows
                } == {(ENTRY_MODULE, (FLEURS_SOURCE,), (CONVERTED_ENCODER,), True)}

            def test_should_request_of_each_shard_its_run_variant_device_languages_and_place(
                self, derived_rows: tuple[DerivedRow, ...]
            ) -> None:
                assert [derived.request for derived in derived_rows] == [
                    ShardRequest("int8-cpu", "int8", Device.CPU, BOTH_LANGUAGES, 0, 3),
                    ShardRequest("int8-cpu", "int8", Device.CPU, BOTH_LANGUAGES, 1, 3),
                    ShardRequest("int8-cpu", "int8", Device.CPU, BOTH_LANGUAGES, 2, 3),
                    ShardRequest("fp16-gpu", "fp16", Device.GPU, ("fr_fr",), 0, 1),
                ]

        class TestWhenTheRunsTotalOneShard:
            @pytest.fixture
            def derived_rows(self) -> tuple[DerivedRow, ...]:
                return derive_rows(TEMPLATE, (SINGLE_SHARD_RUN,))

            def test_should_name_its_row_after_the_step_alone_so_later_steps_bind_its_outputs(
                self, derived_rows: tuple[DerivedRow, ...]
            ) -> None:
                assert [derived.row.name for derived in derived_rows] == ["bench"]

            def test_should_request_the_only_shard_of_the_run(
                self, derived_rows: tuple[DerivedRow, ...]
            ) -> None:
                assert [derived.request for derived in derived_rows] == [
                    ShardRequest("int8-cpu", "int8", Device.CPU, BOTH_LANGUAGES, 0, 1)
                ]

        class TestWhenNoRunIsGiven:
            @pytest.fixture
            def derived_rows(self) -> tuple[DerivedRow, ...]:
                return derive_rows(TEMPLATE, ())

            def test_should_derive_no_row(self, derived_rows: tuple[DerivedRow, ...]) -> None:
                assert derived_rows == ()

    class TestTheParseShardRequestFunction:
        class TestWhenTheCoreWroteTheRequestOfAShard:
            @pytest.fixture
            def shard_request(self) -> ShardRequest:
                return parse_shard_request(render_shard_request(REQUEST))

            def test_should_read_the_request_it_rendered(self, shard_request: ShardRequest) -> None:
                assert shard_request == REQUEST

        @pytest.mark.parametrize(
            "request_at_a_bound",
            list(ACCEPTED_REQUESTS.values()),
            ids=list(ACCEPTED_REQUESTS.keys()),
        )
        class TestWhenTheRequestLiesAtABound:
            @pytest.fixture
            def shard_request(self, request_at_a_bound: ShardRequest) -> ShardRequest:
                return parse_shard_request(render_shard_request(request_at_a_bound))

            def test_should_read_it(
                self, shard_request: ShardRequest, request_at_a_bound: ShardRequest
            ) -> None:
                assert shard_request == request_at_a_bound

        @pytest.mark.parametrize(
            ("content", "reason"),
            list(REFUSED_REQUESTS.values()),
            ids=list(REFUSED_REQUESTS.keys()),
        )
        class TestWhenTheRequestIsMalformed:
            @pytest.fixture
            def parse_act(self, content: bytes) -> Callable[[], ShardRequest]:
                return partial(parse_shard_request, content)

            def test_should_refuse_it_and_say_why(
                self, parse_act: Callable[[], ShardRequest], reason: str
            ) -> None:
                with pytest.raises(ShardRequestError, match=re.escape(reason)):
                    parse_act()

    class TestTheSelectShardFunction:
        class TestWhenSevenEntriesAreDealtInThreeShards:
            @pytest.fixture
            def second_shard(self) -> list[str]:
                return select_shard(["a", "b", "c", "d", "e", "f", "g"], 1, 3)

            def test_should_deal_the_entries_in_turn(self, second_shard: list[str]) -> None:
                assert second_shard == ["b", "e"]

        class TestWhenGivenAnyCountOfEntriesAndOfShards:
            def test_should_place_every_entry_in_exactly_one_shard(self) -> None:
                generator = random.Random(PROPERTY_SEED)
                cases = build_sharding_cases(generator)
                mismatches = [
                    (entry_count, shard_count)
                    for entry_count, shard_count in cases
                    if sorted(
                        entry
                        for shard in range(shard_count)
                        for entry in select_shard(range(entry_count), shard, shard_count)
                    )
                    != list(range(entry_count))
                ]
                assert mismatches == [], f"seed {PROPERTY_SEED}"
                assert len(cases) == PROPERTY_CASE_COUNT, f"seed {PROPERTY_SEED}"

            def test_should_deal_shards_whose_sizes_differ_by_at_most_one(self) -> None:
                generator = random.Random(PROPERTY_SEED)
                cases = build_sharding_cases(generator)
                shard_sizes = [
                    [
                        len(select_shard(range(entry_count), shard, shard_count))
                        for shard in range(shard_count)
                    ]
                    for entry_count, shard_count in cases
                ]
                mismatches = [
                    (case, sizes)
                    for case, sizes in zip(cases, shard_sizes, strict=True)
                    if max(sizes) - min(sizes) > 1
                ]
                assert mismatches == [], f"seed {PROPERTY_SEED}"
                assert len(cases) == PROPERTY_CASE_COUNT, f"seed {PROPERTY_SEED}"
