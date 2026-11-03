import json
import shutil
import subprocess
import sys
from collections.abc import Callable
from functools import partial
from pathlib import Path

import pytest

from mostt_model_builder.bootstrap import (
    ROOT_REFUSED_EXIT_STATUS,
    find_root_ids,
    refuse_malformed_entry_module,
    refuse_root,
)

BOOTSTRAP_PATH = (
    Path(__file__).resolve().parent.parent / "src" / "mostt_model_builder" / "bootstrap.py"
)
UNPRIVILEGED_ID = 1000
ID_NAMES = (
    "real user id",
    "effective user id",
    "saved user id",
    "real group id",
    "effective group id",
    "saved group id",
)
GOOD_RECIPE_NAME = "parakeet-tdt"
RECIPE_NAMES_OUTSIDE_THE_GRAMMAR = ("Parakeet-tdt", "a" * 33, "-parakeet-tdt")
LINKED_RECIPE_NAME = "linked-recipe"
PLAIN_FILE_RECIPE_NAME = "plain-file"
PATH_PROBE_MODULE = "path_probe"
PATH_QUERY = "import json\nimport sys\n\nprint(json.dumps(sys.path))\n"
ISOLATED_RUN_TIMEOUT_S = 60
MALFORMED_ENTRY_MODULES = {
    "an empty name": "",
    "an uppercase letter": "Path_probe",
    "a leading digit": "1path_probe",
    "a dash": "path-probe",
    "a leading dot": ".path_probe",
    "a trailing dot": "path_probe.",
    "two dots in a row": "mostt_model_builder..host_main",
    "a path": "src/path_probe",
    "a trailing line feed": "path_probe\n",
    "a space": "path_probe host_main",
}


def build_ids_with_one_root(root_id_name: str) -> tuple[tuple[int, ...], tuple[int, ...]]:
    ids = tuple(0 if name == root_id_name else UNPRIVILEGED_ID for name in ID_NAMES)
    return ids[:3], ids[3:]


def build_bootstrap_root(root: Path) -> None:
    package_directory = root / "src" / "mostt_model_builder"
    package_directory.mkdir(parents=True)
    shutil.copyfile(BOOTSTRAP_PATH, package_directory / "bootstrap.py")
    (root / "src" / f"{PATH_PROBE_MODULE}.py").write_text(PATH_QUERY)
    recipes_directory = root / "recipes"
    for recipe_name in (GOOD_RECIPE_NAME, *RECIPE_NAMES_OUTSIDE_THE_GRAMMAR):
        (recipes_directory / recipe_name / "src").mkdir(parents=True)
    (recipes_directory / LINKED_RECIPE_NAME).symlink_to(recipes_directory / GOOD_RECIPE_NAME)
    (recipes_directory / PLAIN_FILE_RECIPE_NAME).write_text("")


def run_isolated_python(arguments: list[str]) -> list[str]:
    completed_run = subprocess.run(
        [sys.executable, "-I", "-S", "-B", *arguments],
        capture_output=True,
        check=True,
        text=True,
        timeout=ISOLATED_RUN_TIMEOUT_S,
    )
    parsed: object = json.loads(completed_run.stdout)
    if not isinstance(parsed, list) or not all(isinstance(entry, str) for entry in parsed):
        raise TypeError(f"the path report is not a list of strings: {completed_run.stdout!r}")
    return [str(entry) for entry in parsed]


class TestTheBootstrap:
    class TestTheRefuseRootFunction:
        @pytest.mark.parametrize("root_id_name", ID_NAMES)
        class TestWhenOneIdIsZero:
            @pytest.fixture
            def refuse_root_act(self, root_id_name: str) -> Callable[[], None]:
                user_ids, group_ids = build_ids_with_one_root(root_id_name)
                return partial(refuse_root, user_ids, group_ids)

            def test_should_exit_with_status_77_so_root_leaves_no_file_the_user_cannot_remove(
                self, refuse_root_act: Callable[[], None]
            ) -> None:
                with pytest.raises(SystemExit, match=f"^{ROOT_REFUSED_EXIT_STATUS}$"):
                    refuse_root_act()

    class TestTheFindRootIdsFunction:
        @pytest.mark.parametrize("root_id_name", ID_NAMES)
        class TestWhenOneIdIsZero:
            @pytest.fixture
            def root_ids(self, root_id_name: str) -> list[str]:
                user_ids, group_ids = build_ids_with_one_root(root_id_name)
                return find_root_ids(user_ids, group_ids)

            def test_should_name_that_id(self, root_ids: list[str], root_id_name: str) -> None:
                assert root_ids == [root_id_name]

        class TestWhenNoIdIsZero:
            @pytest.fixture
            def root_ids(self) -> list[str]:
                return find_root_ids((UNPRIVILEGED_ID,) * 3, (UNPRIVILEGED_ID,) * 3)

            def test_should_find_none_so_that_the_build_runs(self, root_ids: list[str]) -> None:
                assert root_ids == []

    class TestTheRefuseMalformedEntryModuleFunction:
        @pytest.mark.parametrize(
            "entry_module",
            list(MALFORMED_ENTRY_MODULES.values()),
            ids=list(MALFORMED_ENTRY_MODULES.keys()),
        )
        class TestWhenTheNameIsNotADottedModuleName:
            @pytest.fixture
            def refuse_act(self, entry_module: str) -> Callable[[], None]:
                return partial(refuse_malformed_entry_module, entry_module)

            def test_should_refuse_it(self, refuse_act: Callable[[], None]) -> None:
                with pytest.raises(SystemExit, match="is not a dotted module name"):
                    refuse_act()

    class TestWhenAnIsolatedPythonRunsItByPath:
        @pytest.fixture(scope="class")
        @classmethod
        def bootstrap_root(cls, tmp_path_factory: pytest.TempPathFactory) -> Path:
            root = tmp_path_factory.mktemp("bootstrap-root").resolve()
            build_bootstrap_root(root)
            return root

        @pytest.fixture(scope="class")
        @classmethod
        def base_path(cls) -> list[str]:
            return run_isolated_python(["-c", PATH_QUERY])

        @pytest.fixture(scope="class")
        @classmethod
        def entry_module_path(cls, bootstrap_root: Path) -> list[str]:
            bootstrap = bootstrap_root / "src" / "mostt_model_builder" / "bootstrap.py"
            return run_isolated_python([str(bootstrap), PATH_PROBE_MODULE])

        def test_should_append_the_source_then_each_real_well_named_recipe_source(
            self, entry_module_path: list[str], base_path: list[str], bootstrap_root: Path
        ) -> None:
            assert entry_module_path == [
                *base_path,
                str(bootstrap_root / "src"),
                str(bootstrap_root / "recipes" / GOOD_RECIPE_NAME / "src"),
            ]
