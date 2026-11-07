import modulefinder
import re
from collections.abc import Callable
from functools import partial
from pathlib import Path, PurePosixPath

import pytest

from mostt_model_builder.domain.cache.import_closure import (
    ImportClosureError,
    close_imports,
    is_project_module,
)
from mostt_model_builder.domain.containers.container_steps import (
    CORE_IMAGES,
    LOCK_ROW,
    PUBLISH_ROW,
    RUNNABLE_CORE_ROWS,
    build_clean_row,
)
from mostt_model_builder.domain.contract.recipe_contract import (
    RECIPES_DIRECTORY_NAME,
    RecipeFacts,
    judge_recipe,
)
from mostt_model_builder.host_main import (
    CLONE_ROOT,
    RECIPES,
    find_present_entry_modules,
    is_card_template_present,
    read_pins,
)

ENTRY = "mostt_model_builder.steps.entry"
ENTRY_PATH = PurePosixPath("src/mostt_model_builder/steps/entry.py")
CORE_PACKAGE_PATH = PurePosixPath("src/mostt_model_builder/__init__.py")
STEPS_PACKAGE_PATH = PurePosixPath("src/mostt_model_builder/steps/__init__.py")
DOMAIN_PACKAGE_PATH = PurePosixPath("src/mostt_model_builder/domain/__init__.py")
NAMES_PATH = PurePosixPath("src/mostt_model_builder/domain/names.py")
RULES_PATH = PurePosixPath("src/mostt_model_builder/domain/rules.py")
RECIPE_PACKAGE_PATH = PurePosixPath("recipes/family/src/mostt_recipe_family/__init__.py")
RECIPE_NAMES_PATH = PurePosixPath("recipes/family/src/mostt_recipe_family/names.py")
PACKAGES = {
    CORE_PACKAGE_PATH: b"",
    STEPS_PACKAGE_PATH: b"",
    DOMAIN_PACKAGE_PATH: b"",
    NAMES_PATH: b"NAME = 'names'\n",
    RULES_PATH: b"import json\n",
    RECIPE_PACKAGE_PATH: b"",
    RECIPE_NAMES_PATH: b"",
}
STEP_PACKAGES = (CORE_PACKAGE_PATH, STEPS_PACKAGE_PATH)
NAMES_CLOSURE = (*STEP_PACKAGES, DOMAIN_PACKAGE_PATH, NAMES_PATH)
IMPORT_FORMS = {
    "an import of a module by its full name": (
        b"import mostt_model_builder.domain.names\n",
        NAMES_CLOSURE,
    ),
    "a name taken from a module": (
        b"from mostt_model_builder.domain.names import NAME\n",
        NAMES_CLOSURE,
    ),
    "a module taken from its package": (
        b"from mostt_model_builder.domain import names\n",
        NAMES_CLOSURE,
    ),
    "an import inside a function": (
        b"def run() -> None:\n    from mostt_model_builder.domain.names import NAME\n",
        NAMES_CLOSURE,
    ),
    "an import in a try block": (
        b"try:\n    import mostt_model_builder.domain.names\nexcept ImportError:\n    pass\n",
        NAMES_CLOSURE,
    ),
    "an import read by the type checker alone": (
        b"from typing import TYPE_CHECKING\n"
        b"if TYPE_CHECKING:\n    from mostt_model_builder.domain.names import NAME\n",
        NAMES_CLOSURE,
    ),
    "a module of a recipe": (
        b"from mostt_recipe_family.names import NAME\n",
        (*STEP_PACKAGES, RECIPE_PACKAGE_PATH, RECIPE_NAMES_PATH),
    ),
    "a module that imports another": (
        b"import mostt_model_builder.domain.rules\nimport mostt_model_builder.domain.names\n",
        (*NAMES_CLOSURE, RULES_PATH),
    ),
    "a library and the standard library alone": (
        b"import json\nimport torch\nfrom onnx import helper\n",
        STEP_PACKAGES,
    ),
}
REFUSED_TREES = {
    "a relative import": (
        {**PACKAGES, ENTRY_PATH: b"from . import names\n"},
        f"{ENTRY_PATH} imports relatively, at line 1",
    ),
    "a file that does not parse": (
        {**PACKAGES, ENTRY_PATH: b"import (\n"},
        f"{ENTRY_PATH} does not parse as Python",
    ),
    "an entry module missing from the tree": (
        PACKAGES,
        f"the entry module {ENTRY} has no file in HEAD's tree",
    ),
    "an import of a module missing from the tree": (
        {**PACKAGES, ENTRY_PATH: b"import mostt_model_builder.domain.missing\n"},
        "mostt_model_builder.domain.missing is imported by the closure of "
        f"{ENTRY}, and HEAD's tree holds no file of it",
    ),
}
HOST_CPU_COUNT = 20


def list_entry_modules() -> list[str]:
    recipes = [
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
    ]
    rows = (
        *RUNNABLE_CORE_ROWS,
        LOCK_ROW,
        PUBLISH_ROW,
        *(build_clean_row(image) for image in CORE_IMAGES),
        *(row for recipe in recipes for row in recipe.rows),
    )
    return sorted({row.entry_module for row in rows})


def read_checkout_files() -> dict[PurePosixPath, bytes]:
    return {
        PurePosixPath(path.relative_to(CLONE_ROOT)): path.read_bytes()
        for directory in ("src", RECIPES_DIRECTORY_NAME)
        for path in (CLONE_ROOT / directory).rglob("*.py")
        if "__pycache__" not in path.parts
    }


def list_source_roots() -> list[str]:
    return [
        str(CLONE_ROOT / "src"),
        *(str(path) for path in sorted((CLONE_ROOT / RECIPES_DIRECTORY_NAME).glob("*/src"))),
    ]


def find_with_modulefinder(entry_module: str) -> set[PurePosixPath]:
    finder = modulefinder.ModuleFinder(path=list_source_roots())
    finder.import_hook(entry_module)
    # typeshed's modulefinder.Module leaves out the __file__ that ModuleFinder sets on each module.
    module_files = {
        name: module.__file__  # type: ignore[attr-defined]
        for name, module in finder.modules.items()
    }
    return {
        PurePosixPath(Path(module_file).relative_to(CLONE_ROOT))
        for name, module_file in module_files.items()
        if is_project_module(name) and module_file is not None
    }


ENTRY_MODULES = list_entry_modules()


class TestTheImportClosure:
    class TestTheCloseImportsFunction:
        @pytest.mark.parametrize(
            ("source", "expected_closure"),
            list(IMPORT_FORMS.values()),
            ids=list(IMPORT_FORMS.keys()),
        )
        class TestWhenTheEntryModuleImportsInOneForm:
            @pytest.fixture
            def closure(self, source: bytes) -> tuple[PurePosixPath, ...]:
                return close_imports(ENTRY, {**PACKAGES, ENTRY_PATH: source})

            def test_should_hold_the_entry_its_packages_and_every_project_module_it_reaches(
                self,
                closure: tuple[PurePosixPath, ...],
                expected_closure: tuple[PurePosixPath, ...],
            ) -> None:
                assert closure == tuple(sorted({ENTRY_PATH, *expected_closure}))

        @pytest.mark.parametrize(
            ("tree_files", "reason"), list(REFUSED_TREES.values()), ids=list(REFUSED_TREES.keys())
        )
        class TestWhenTheTreeCannotBeClosedFaithfully:
            @pytest.fixture
            def close_act(
                self, tree_files: dict[PurePosixPath, bytes]
            ) -> Callable[[], tuple[PurePosixPath, ...]]:
                return partial(close_imports, ENTRY, tree_files)

            def test_should_refuse_it_loudly_since_these_are_the_projects_own_files(
                self, close_act: Callable[[], tuple[PurePosixPath, ...]], reason: str
            ) -> None:
                with pytest.raises(ImportClosureError, match=re.escape(reason)):
                    close_act()

        class TestWhenEveryEntryModuleOfTheStepTableIsClosedOnTheCheckout:
            @pytest.fixture(scope="class")
            @classmethod
            def mismatches(cls) -> list[str]:
                files = read_checkout_files()
                return [
                    entry_module
                    for entry_module in ENTRY_MODULES
                    if set(close_imports(entry_module, files))
                    != find_with_modulefinder(entry_module)
                ]

            def test_should_check_every_entry_module_of_the_core_and_the_recipes(self) -> None:
                assert len(ENTRY_MODULES) >= len(RUNNABLE_CORE_ROWS) + len(RECIPES)

            def test_should_find_what_the_standard_librarys_modulefinder_finds(
                self, mismatches: list[str]
            ) -> None:
                assert mismatches == []
