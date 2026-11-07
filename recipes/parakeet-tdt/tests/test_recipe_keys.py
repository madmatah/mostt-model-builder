import hashlib
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from pathlib import PurePosixPath

import pytest

from mostt_model_builder.application.cache_entries import EntryKeys
from mostt_model_builder.domain.cache.build_units import BuildUnit, list_every_step, plan_units
from mostt_model_builder.domain.cache.hardware_facts import HardwareFacts
from mostt_model_builder.domain.cache.import_closure import KEYED_TREE_DIRECTORIES
from mostt_model_builder.domain.containers.container_steps import INSPECT_ROW
from mostt_model_builder.domain.containers.work_directory import CANDIDATE_ENTRY_NAME
from mostt_model_builder.domain.contract.recipe_contract import (
    HostContentOrigin,
    JudgedRecipe,
    RecipeContract,
    RecipeFacts,
    judge_recipe,
)
from mostt_model_builder.host_main import (
    CLONE_ROOT,
    find_present_entry_modules,
    is_card_template_present,
    read_pins,
)
from mostt_recipe_parakeet_tdt.recipe import RECIPE

RECIPE_SOURCES_DIRECTORY = CLONE_ROOT / "recipes" / RECIPE.name
HOST_CPU_COUNT = 20
HARDWARE = HardwareFacts(("AMD Ryzen 9 7950X 16-Core Processor",), ("RTX 4090",), "550.54.14")
BYTECODE_DIRECTORY_NAME = "__pycache__"
REMOVED_LANGUAGE = "en_us"
REMOVED_VARIANT = "bfloat16"
CHANGED_BY_A_VARIANT = {CANDIDATE_ENTRY_NAME, INSPECT_ROW.name}


def read_clone_tree() -> dict[PurePosixPath, bytes]:
    return {
        PurePosixPath(path.relative_to(CLONE_ROOT)): path.read_bytes()
        for directory in KEYED_TREE_DIRECTORIES
        for path in sorted((CLONE_ROOT / directory).rglob("*"))
        if path.is_file() and BYTECODE_DIRECTORY_NAME not in path.parts
    }


def digest_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def digest_unit_outputs(
    unit: BuildUnit, recipe: JudgedRecipe, digests: Mapping[str, Mapping[str, str]]
) -> dict[str, str]:
    if unit.row is not None:
        return {
            output.name: digest_text(f"{unit.name}/{output.name}") for output in unit.row.outputs
        }
    return {
        candidate_file.name: (
            hashlib.sha256(candidate_file.origin.content).hexdigest()
            if isinstance(candidate_file.origin, HostContentOrigin)
            else digests[candidate_file.origin.step][candidate_file.origin.output]
        )
        for candidate_file in recipe.candidate
    }


def key_every_row(
    contract: RecipeContract, tree_files: Mapping[PurePosixPath, bytes]
) -> dict[str, str]:
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
    entry_keys = EntryKeys(recipe, tree_files, HARDWARE)
    digests: dict[str, Mapping[str, str]] = {}
    keys = {}
    for unit in plan_units(recipe, list_every_step(recipe)):
        keys[unit.name], _ = entry_keys.key_unit(unit, digests)
        digests[unit.name] = digest_unit_outputs(unit, recipe, digests)
    return keys


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
    class TestWhenALanguageLeavesEveryRunAndStaysOnTheBench:
        @pytest.fixture(scope="class")
        @classmethod
        def key_changes(cls) -> KeyChanges:
            tree_files = read_clone_tree()
            return compare_keys(
                key_every_row(RECIPE, tree_files),
                key_every_row(remove_language(RECIPE, REMOVED_LANGUAGE), tree_files),
                is_row_of_the_removed_language,
            )

        def test_should_drop_every_row_of_that_language_and_no_other(
            self, key_changes: KeyChanges
        ) -> None:
            assert key_changes.removed_rows == key_changes.rows_of_the_removed_part

        def test_should_find_rows_of_that_language_to_drop(self, key_changes: KeyChanges) -> None:
            assert key_changes.rows_of_the_removed_part != frozenset()

        def test_should_change_no_other_key(self, key_changes: KeyChanges) -> None:
            assert key_changes.changed_rows == frozenset()

    class TestWhenAConvertedVariantLeavesTheRecipe:
        @pytest.fixture(scope="class")
        @classmethod
        def key_changes(cls) -> KeyChanges:
            tree_files = read_clone_tree()
            return compare_keys(
                key_every_row(RECIPE, tree_files),
                key_every_row(remove_variant(RECIPE, REMOVED_VARIANT), tree_files),
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
