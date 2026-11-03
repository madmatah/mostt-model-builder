import os
import re
import subprocess
from pathlib import Path, PurePosixPath

import pytest

from mostt_model_builder.bootstrap import RECIPE_NAME_GRAMMAR
from mostt_model_builder.domain.containers.base_images import (
    BASE_IMAGES,
    PYTHON_BASE_IMAGE,
    UV_IMAGE,
)
from mostt_model_builder.domain.containers.container_steps import CORE_IMAGES

REPO_ROOT = Path(__file__).resolve().parent.parent
RECIPES_DIRECTORY_NAME = "recipes"
EXPECTED_RECIPE_COUNT = 0
GIT_TIMEOUT_S = 60
GIT_VARIABLE_PREFIX = "GIT_"
CORE_CONTAINERFILES = sorted((REPO_ROOT / "images").glob("*/Containerfile"))
CONTAINERFILE_IDS = [path.parent.name for path in CORE_CONTAINERFILES]
COMMENT_PREFIX = "#"
PARSER_DIRECTIVE = re.compile(r"#\s*(syntax|check|escape)\s*=", re.IGNORECASE)
BASE_IMAGE_ALTERNATIVES = "|".join(re.escape(image) for image in sorted(BASE_IMAGES))
ENVIRONMENT_VALUE = "[A-Za-z0-9._/-]+"
UV_ENVIRONMENT_NAMES = (
    "UV_FROZEN|UV_NO_CACHE|UV_LINK_MODE|UV_COMPILE_BYTECODE|UV_PYTHON|UV_PYTHON_DOWNLOADS"
    "|UV_PROJECT_ENVIRONMENT"
)
ACCEPTED_FORMS = {
    "a base image of the build, by its digest": re.compile(
        rf"FROM ({BASE_IMAGE_ALTERNATIVES})( AS [a-z][a-z0-9-]*)?"
    ),
    "a working directory": re.compile(r"WORKDIR /[A-Za-z0-9._/-]+"),
    "the variables uv needs, and never one of Python's": re.compile(
        rf"ENV ({UV_ENVIRONMENT_NAMES})={ENVIRONMENT_VALUE}"
        rf"( ({UV_ENVIRONMENT_NAMES})={ENVIRONMENT_VALUE})*"
    ),
    "the core's sources, at the repository's layout under one root": re.compile(
        r"COPY src/mostt_model_builder /opt/model-builder/src/mostt_model_builder"
    ),
    "the uv binary from the uv image, by its digest": re.compile(
        rf"COPY --from={re.escape(UV_IMAGE)} /uv /usr/local/bin/uv"
    ),
    "the build argument of the tree": re.compile(r"ARG TREE_HASH"),
    "the label of the tree": re.compile(r"LABEL mostt-model-builder\.tree=\$\{TREE_HASH\}"),
}
LABEL_LINES = ["ARG TREE_HASH", "LABEL mostt-model-builder.tree=${TREE_HASH}"]
DOCKERIGNORE_SUFFIX = ".dockerignore"
DOCKERIGNORE_REINCLUDED_ENTRIES = frozenset(
    {"!src/mostt_model_builder", "!images/pyproject.toml", "!images/uv.lock"}
)
DOCKERIGNORE_EXCLUSIONS = ["**/__pycache__", "**/*.pyc"]
CONTAINERFILE_WITH_A_DIRECTIVE = f"# syntax=docker/dockerfile:1\nFROM {PYTHON_BASE_IMAGE}\n"
CONTAINERFILE_WITH_A_LATE_DIRECTIVE = f"FROM {PYTHON_BASE_IMAGE}\n# check=skip=all\n"
CONTAINERFILE_WITH_A_DOWNLOAD = f"FROM {PYTHON_BASE_IMAGE}\nRUN curl -o /x https://example.org/x\n"


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


def read_instruction_lines(text: str) -> list[str]:
    return [line for line in text.split("\n") if line]


def find_refused_lines(text: str) -> list[str]:
    lines = read_instruction_lines(text)
    refused = [line for line in lines if PARSER_DIRECTIVE.match(line)]
    if not lines or not lines[0].startswith("FROM "):
        refused.append(lines[0] if lines else "")
    refused.extend(
        line
        for line in lines
        if not line.startswith(COMMENT_PREFIX)
        and not any(form.fullmatch(line) for form in ACCEPTED_FORMS.values())
    )
    return refused


def list_instructions(text: str) -> list[str]:
    return [line for line in read_instruction_lines(text) if not line.startswith(COMMENT_PREFIX)]


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

    class TestWhenTheCoreImagesAreListed:
        def test_should_find_one_containerfile_per_image_of_the_step_table(self) -> None:
            assert sorted(image.name for image in CORE_IMAGES) == CONTAINERFILE_IDS

    @pytest.mark.parametrize("containerfile", CORE_CONTAINERFILES, ids=CONTAINERFILE_IDS)
    class TestWhenACoreContainerfileIsRead:
        @pytest.fixture
        def text(self, containerfile: Path) -> str:
            return containerfile.read_text(encoding="utf-8")

        def test_should_hold_only_the_accepted_forms_so_nothing_else_reaches_an_image(
            self, text: str
        ) -> None:
            assert len(list_instructions(text)) > 0
            assert find_refused_lines(text) == []

        def test_should_end_with_the_label_of_the_tree_so_a_change_of_code_reuses_the_rest(
            self, text: str
        ) -> None:
            assert list_instructions(text)[-2:] == LABEL_LINES

        def test_should_run_no_command_since_neither_image_has_a_main_install(
            self, text: str
        ) -> None:
            assert [line for line in list_instructions(text) if line.startswith("RUN")] == []

    @pytest.mark.parametrize("containerfile", CORE_CONTAINERFILES, ids=CONTAINERFILE_IDS)
    class TestWhenItsDockerignoreIsRead:
        @pytest.fixture
        def entries(self, containerfile: Path) -> list[str]:
            dockerignore = containerfile.with_name(containerfile.name + DOCKERIGNORE_SUFFIX)
            return read_instruction_lines(dockerignore.read_text(encoding="utf-8"))

        def test_should_exclude_everything_first(self, entries: list[str]) -> None:
            assert entries[0] == "*"

        def test_should_include_again_only_what_an_image_copies(self, entries: list[str]) -> None:
            assert [
                entry
                for entry in entries
                if entry.startswith("!") and entry not in DOCKERIGNORE_REINCLUDED_ENTRIES
            ] == []

        def test_should_leave_out_bytecode_last(self, entries: list[str]) -> None:
            assert entries[-2:] == DOCKERIGNORE_EXCLUSIONS

    @pytest.mark.parametrize(
        ("text", "refused_line"),
        [
            pytest.param(
                CONTAINERFILE_WITH_A_DIRECTIVE,
                "# syntax=docker/dockerfile:1",
                id="a parser directive first",
            ),
            pytest.param(
                CONTAINERFILE_WITH_A_LATE_DIRECTIVE,
                "# check=skip=all",
                id="a parser directive later",
            ),
            pytest.param(
                CONTAINERFILE_WITH_A_DOWNLOAD,
                "RUN curl -o /x https://example.org/x",
                id="a download",
            ),
        ],
    )
    class TestWhenAContainerfileHoldsAFormOutsideTheTable:
        @pytest.fixture
        def refused_lines(self, text: str) -> list[str]:
            return find_refused_lines(text)

        def test_should_refuse_that_form_and_no_other_line(
            self, refused_lines: list[str], refused_line: str
        ) -> None:
            assert set(refused_lines) == {refused_line}
