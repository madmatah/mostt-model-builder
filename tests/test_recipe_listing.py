import os
import subprocess
from pathlib import Path, PurePosixPath

import pytest

from mostt_model_builder.bootstrap import RECIPE_NAME_GRAMMAR

REPO_ROOT = Path(__file__).resolve().parent.parent
RECIPES_DIRECTORY_NAME = "recipes"
EXPECTED_RECIPE_COUNT = 0
GIT_TIMEOUT_S = 60
GIT_VARIABLE_PREFIX = "GIT_"


def list_environment_without_git_variables() -> dict[str, str]:
    return {
        name: value
        for name, value in os.environ.items()
        if not name.startswith(GIT_VARIABLE_PREFIX)
    }


def run_git(repository: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repository), *arguments],
        capture_output=True,
        check=True,
        text=True,
        env=list_environment_without_git_variables(),
        timeout=GIT_TIMEOUT_S,
    ).stdout


def list_recipe_entries(repository: Path) -> list[str]:
    tracked_paths = run_git(repository, "ls-files", "-z", "--", RECIPES_DIRECTORY_NAME)
    return sorted({PurePosixPath(path).parts[1] for path in tracked_paths.split("\0") if path})


class TestTheRecipeListing:
    class TestTheListRecipeEntriesFunction:
        class TestWhenACheckoutLeftADirectoryOfIgnoredBytecode:
            @pytest.fixture
            def recipe_entries(self, tmp_path: Path) -> list[str]:
                run_git(tmp_path, "init", "--quiet")
                (tmp_path / ".gitignore").write_text("__pycache__/\n")
                tracked_file = tmp_path / RECIPES_DIRECTORY_NAME / "tracked-recipe" / "sources.toml"
                tracked_file.parent.mkdir(parents=True)
                tracked_file.write_text("")
                bytecode_directory = (
                    tmp_path / RECIPES_DIRECTORY_NAME / "later-recipe" / "src" / "__pycache__"
                )
                bytecode_directory.mkdir(parents=True)
                (bytecode_directory / "recipe.cpython-312.pyc").write_bytes(b"")
                run_git(tmp_path, "add", "--", ".gitignore", str(tracked_file))
                return list_recipe_entries(tmp_path)

            def test_should_count_only_the_directory_that_holds_a_tracked_file(
                self, recipe_entries: list[str]
            ) -> None:
                assert recipe_entries == ["tracked-recipe"]

        class TestWhenATrackedDirectoryHasAMalformedName:
            @pytest.fixture
            def recipe_entries(self, tmp_path: Path) -> list[str]:
                run_git(tmp_path, "init", "--quiet")
                tracked_file = tmp_path / RECIPES_DIRECTORY_NAME / "Malformed_Recipe" / "README"
                tracked_file.parent.mkdir(parents=True)
                tracked_file.write_text("")
                run_git(tmp_path, "add", "--", str(tracked_file))
                return list_recipe_entries(tmp_path)

            def test_should_list_it_so_that_the_grammar_refuses_it(
                self, recipe_entries: list[str]
            ) -> None:
                assert recipe_entries == ["Malformed_Recipe"]

    class TestWhenTheRecipesDirectoryIsListed:
        @pytest.fixture
        def recipe_entries(self) -> list[str]:
            return list_recipe_entries(REPO_ROOT)

        def test_should_hold_as_many_recipes_as_the_listing_expects(
            self, recipe_entries: list[str]
        ) -> None:
            assert len(recipe_entries) == EXPECTED_RECIPE_COUNT

        def test_should_refuse_a_name_outside_the_grammar_which_the_bootstrap_would_skip(
            self, recipe_entries: list[str]
        ) -> None:
            assert [
                name for name in recipe_entries if not RECIPE_NAME_GRAMMAR.fullmatch(name)
            ] == []
