import hashlib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import PurePosixPath

import pytest

from mostt_model_builder.domain.bench.fleurs_check_request import (
    FLEURS_CHECK_REQUEST_FILE_NAME,
    BenchFiles,
    FleursCheckRequest,
    render_fleurs_check_request,
)
from mostt_model_builder.domain.bench.fleurs_index import FLEURS_REPOSITORY, locate_bench_files
from mostt_model_builder.domain.cache.hardware_facts import HardwareFacts
from mostt_model_builder.domain.cache.output_keys import (
    RowKeyInputs,
    compute_key,
    document_candidate_key,
    document_row_key,
    name_earlier_output,
)
from mostt_model_builder.domain.containers.container_paths import CONTAINER_ROOT
from mostt_model_builder.domain.containers.container_steps import (
    FLEURS_CHECK_ROW,
    INSPECT_ROW,
    SourceReference,
    StepPlacement,
    StepRow,
    plan_container_run,
)
from mostt_model_builder.domain.containers.work_directory import locate_referenced_sources
from mostt_model_builder.domain.contract.recipe_contract import (
    HostContentOrigin,
    JudgedCandidateFile,
    JudgedRecipe,
    RecipeContract,
    RecipeFacts,
    judge_recipe,
)
from mostt_model_builder.domain.contract.shard_request import (
    SHARD_REQUEST_FILE_NAME,
    render_shard_request,
)
from mostt_model_builder.domain.formats.file_formats import FileFormat
from mostt_model_builder.domain.onnx.conversion_request import (
    CONVERSION_REQUEST_FILE_NAME,
    render_conversion_request,
)
from mostt_model_builder.domain.onnx.inspection_report import (
    INSPECTION_REQUEST_FILE_NAME,
    render_inspection_request,
)
from mostt_model_builder.domain.sources.source_pins import SourcePin
from mostt_model_builder.host_main import (
    CLONE_ROOT,
    find_present_entry_modules,
    is_card_template_present,
    read_pins,
)
from mostt_recipe_parakeet_tdt.recipe import RECIPE

RECIPE_SOURCES_DIRECTORY = CLONE_ROOT / "recipes" / RECIPE.name
HOST_CPU_COUNT = 20
GROUP_ID = 1000
IMAGE_ID = "sha256:" + "a" * 64
WORK_ROOT = PurePosixPath("/data/builds/work")
PLACEMENT = StepPlacement(
    output_directory=PurePosixPath("cache/row/output"),
    request_directory=PurePosixPath("cache/row/request"),
    log_path=PurePosixPath("cache/row/step.log"),
    candidate_directory=PurePosixPath("cache/candidate/files"),
)
FIXED_CODE = {PurePosixPath("src/mostt_model_builder/steps/step.py"): b"x = 1\n"}
FIXED_IMAGE = {PurePosixPath("images/step/Containerfile"): b"FROM python\n"}
HARDWARE = HardwareFacts(("AMD Ryzen 9 7950X 16-Core Processor",), ("RTX 4090",), "550.54.14")
CANDIDATE_KEY_NAME = "the candidate"
BENCH_LANGUAGES = ("fr_fr", "en_us")
REMOVED_LANGUAGE = "en_us"
REMOVED_VARIANT = "bfloat16"
CHANGED_BY_A_LANGUAGE = {FLEURS_CHECK_ROW.name}
CHANGED_BY_A_VARIANT = {CANDIDATE_KEY_NAME, INSPECT_ROW.name}


def digest_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def build_fleurs_check_row(languages: Sequence[str]) -> StepRow:
    return replace(
        FLEURS_CHECK_ROW,
        sources=tuple(
            SourceReference(FLEURS_REPOSITORY, path)
            for language in languages
            for path in locate_bench_files(language)
        ),
    )


def render_fleurs_check_files(
    languages: Sequence[str], pins: Sequence[SourcePin]
) -> dict[str, bytes]:
    source_paths = locate_referenced_sources(build_fleurs_check_row(languages).sources, pins)
    bench_files = []
    for language in languages:
        index_path, archive_path = locate_bench_files(language)
        bench_files.append(
            BenchFiles(
                language=language,
                index=CONTAINER_ROOT / source_paths[SourceReference(FLEURS_REPOSITORY, index_path)],
                archive=CONTAINER_ROOT
                / source_paths[SourceReference(FLEURS_REPOSITORY, archive_path)],
            )
        )
    return {
        FLEURS_CHECK_REQUEST_FILE_NAME: render_fleurs_check_request(
            FleursCheckRequest(tuple(bench_files))
        )
    }


def list_request_files(recipe: JudgedRecipe, row: StepRow) -> dict[str, bytes]:
    for derived_row in recipe.derived_rows:
        if derived_row.row.name == row.name:
            return {SHARD_REQUEST_FILE_NAME: render_shard_request(derived_row.request)}
    for conversion_row in recipe.conversion_rows:
        if conversion_row.row.name == row.name:
            return {CONVERSION_REQUEST_FILE_NAME: render_conversion_request(conversion_row.request)}
    return {}


def digest_origins(recipe: JudgedRecipe) -> dict[str, str]:
    return {
        name_earlier_output(row.name, output.name): digest_text(f"{row.name}/{output.name}")
        for row in recipe.rows
        for output in row.outputs
    }


def digest_candidate(
    candidate: Sequence[JudgedCandidateFile], origin_digests: Mapping[str, str]
) -> dict[str, str]:
    return {
        candidate_file.name: (
            hashlib.sha256(candidate_file.origin.content).hexdigest()
            if isinstance(candidate_file.origin, HostContentOrigin)
            else origin_digests[
                name_earlier_output(candidate_file.origin.step, candidate_file.origin.output)
            ]
        )
        for candidate_file in candidate
    }


def key_row(
    row: StepRow,
    request_files: Mapping[str, bytes],
    recipe: JudgedRecipe,
    candidate_digests: Mapping[str, str],
) -> str:
    derived_row_names = {derived_row.row.name for derived_row in recipe.derived_rows}
    return compute_key(
        document_row_key(
            RowKeyInputs(
                row=row,
                planned_run=plan_container_run(
                    row,
                    IMAGE_ID,
                    PLACEMENT,
                    locate_referenced_sources(row.sources, recipe.pins),
                    WORK_ROOT,
                    GROUP_ID,
                    {
                        earlier_output.step: PurePosixPath("cache", earlier_output.step)
                        for earlier_output in row.earlier_outputs
                    },
                ),
                request_files=request_files,
                image_files=FIXED_IMAGE,
                image_packages=(),
                bootstrap=b"import sys\n",
                code_files=FIXED_CODE,
                pins=recipe.pins,
                earlier_output_digests=digest_origins(recipe),
                candidate_digests=candidate_digests,
                hardware=HARDWARE if row.name in derived_row_names else None,
            )
        )
    )


def key_every_row(contract: RecipeContract, bench_languages: Sequence[str]) -> dict[str, str]:
    recipe = judge_recipe(
        contract,
        RecipeFacts(
            core_pins=read_pins(CLONE_ROOT),
            recipe_pins=read_pins(RECIPE_SOURCES_DIRECTORY),
            present_entry_modules=find_present_entry_modules(contract),
            host_cpu_count=HOST_CPU_COUNT,
            has_card_template=is_card_template_present(contract, CLONE_ROOT),
        ),
    )
    origin_digests = digest_origins(recipe)
    candidate_digests = digest_candidate(recipe.candidate, origin_digests)
    onnx_names = [
        candidate_file.name
        for candidate_file in recipe.candidate
        if candidate_file.file_format is FileFormat.ONNX
    ]
    return {
        FLEURS_CHECK_ROW.name: key_row(
            build_fleurs_check_row(bench_languages),
            render_fleurs_check_files(bench_languages, recipe.pins),
            recipe,
            candidate_digests,
        ),
        **{
            row.name: key_row(row, list_request_files(recipe, row), recipe, candidate_digests)
            for row in recipe.rows
        },
        CANDIDATE_KEY_NAME: compute_key(document_candidate_key(recipe.candidate, origin_digests)),
        INSPECT_ROW.name: key_row(
            INSPECT_ROW,
            {INSPECTION_REQUEST_FILE_NAME: render_inspection_request(onnx_names)},
            recipe,
            candidate_digests,
        ),
    }


def remove_language(contract: RecipeContract, language: str) -> RecipeContract:
    return replace(
        contract,
        runs=tuple(
            replace(
                run,
                languages=tuple(
                    run_language for run_language in run.languages if run_language != language
                ),
            )
            for run in contract.runs
        ),
    )


def remove_variant(contract: RecipeContract, variant: str) -> RecipeContract:
    removed_runs = {run.name for run in contract.runs if run.variant == variant}
    return replace(
        contract,
        runs=tuple(run for run in contract.runs if run.name not in removed_runs),
        candidate=tuple(
            replace(
                candidate_file,
                variants=tuple(served for served in candidate_file.variants if served != variant),
            )
            for candidate_file in contract.candidate
            if candidate_file.variants != (variant,)
        ),
        conversions=tuple(
            conversion for conversion in contract.conversions if conversion.variant != variant
        ),
        comparisons=tuple(
            comparison
            for comparison in contract.comparisons
            if not {comparison.run, comparison.reference_run} & removed_runs
        ),
    )


@dataclass(frozen=True)
class KeyChanges:
    removed_rows: frozenset[str]
    changed_rows: frozenset[str]
    rows_of_the_removed_part: frozenset[str]


def compare_keys(
    full_keys: Mapping[str, str],
    reduced_keys: Mapping[str, str],
    removed_part: Callable[[str], bool],
) -> KeyChanges:
    return KeyChanges(
        removed_rows=frozenset(set(full_keys) - set(reduced_keys)),
        changed_rows=frozenset(
            name for name in reduced_keys if reduced_keys[name] != full_keys.get(name)
        ),
        rows_of_the_removed_part=frozenset(name for name in full_keys if removed_part(name)),
    )


def is_row_of_the_removed_language(name: str) -> bool:
    return f"-{REMOVED_LANGUAGE}-" in name


def is_row_of_the_removed_variant(name: str) -> bool:
    return name == f"convert-{REMOVED_VARIANT}" or name.startswith(f"transcribe-{REMOVED_VARIANT}-")


class TestTheKeysOfTheParakeetTdtRecipe:
    class TestWhenALanguageLeavesEveryRun:
        @pytest.fixture(scope="class")
        @classmethod
        def key_changes(cls) -> KeyChanges:
            reduced_languages = tuple(
                language for language in BENCH_LANGUAGES if language != REMOVED_LANGUAGE
            )
            return compare_keys(
                key_every_row(RECIPE, BENCH_LANGUAGES),
                key_every_row(remove_language(RECIPE, REMOVED_LANGUAGE), reduced_languages),
                is_row_of_the_removed_language,
            )

        def test_should_drop_every_row_of_that_language_and_no_other(
            self, key_changes: KeyChanges
        ) -> None:
            assert key_changes.removed_rows == key_changes.rows_of_the_removed_part

        def test_should_find_rows_of_that_language_to_drop(self, key_changes: KeyChanges) -> None:
            assert key_changes.rows_of_the_removed_part != frozenset()

        def test_should_change_the_key_of_the_fleurs_check_alone(
            self, key_changes: KeyChanges
        ) -> None:
            assert key_changes.changed_rows == CHANGED_BY_A_LANGUAGE

    class TestWhenAConvertedVariantLeavesTheRecipe:
        @pytest.fixture(scope="class")
        @classmethod
        def key_changes(cls) -> KeyChanges:
            return compare_keys(
                key_every_row(RECIPE, BENCH_LANGUAGES),
                key_every_row(remove_variant(RECIPE, REMOVED_VARIANT), BENCH_LANGUAGES),
                is_row_of_the_removed_variant,
            )

        def test_should_drop_its_conversion_and_its_runs_and_no_other_row(
            self, key_changes: KeyChanges
        ) -> None:
            assert key_changes.removed_rows == key_changes.rows_of_the_removed_part

        def test_should_find_rows_of_that_variant_to_drop(self, key_changes: KeyChanges) -> None:
            assert key_changes.rows_of_the_removed_part != frozenset()

        def test_should_change_the_keys_of_the_candidate_and_its_inspection_alone(
            self, key_changes: KeyChanges
        ) -> None:
            assert key_changes.changed_rows == CHANGED_BY_A_VARIANT
