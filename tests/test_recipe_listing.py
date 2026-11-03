import os
import re
import subprocess
import tomllib
from pathlib import Path, PurePosixPath
from typing import Any

import pytest

from mostt_model_builder.bootstrap import RECIPE_NAME_GRAMMAR as BOOTSTRAP_RECIPE_NAME_GRAMMAR
from mostt_model_builder.domain.containers.base_images import (
    BASE_IMAGES,
    BASE_PYTHON,
    PYTHON_BASE_IMAGE,
    UV_IMAGE,
)
from mostt_model_builder.domain.containers.container_steps import (
    CORE_IMAGES,
    find_package_of_owner,
)
from mostt_model_builder.domain.contract.recipe_contract import RECIPE_NAME_GRAMMAR
from mostt_model_builder.domain.sources.source_pins import parse_source_pins
from mostt_model_builder.host_main import RECIPES

REPO_ROOT = Path(__file__).resolve().parent.parent
RECIPES_DIRECTORY_NAME = "recipes"
RECIPES_DIRECTORY = REPO_ROOT / RECIPES_DIRECTORY_NAME
EXPECTED_RECIPE_COUNT = 1
GIT_TIMEOUT_S = 60
GIT_VARIABLE_PREFIX = "GIT_"
RECIPE_NAMES = sorted(RECIPES)
# A manifest or a lock is a tree of TOML tables, which the tests walk by their known keys.
MANIFEST: dict[str, Any] = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())
IMAGES_MANIFEST: dict[str, Any] = tomllib.loads((REPO_ROOT / "images/pyproject.toml").read_text())
IMAGES_LOCK: dict[str, Any] = tomllib.loads((REPO_ROOT / "images/uv.lock").read_text())
SOURCES_FILES = [REPO_ROOT / "sources.toml", *sorted(RECIPES_DIRECTORY.glob("*/sources.toml"))]
# The one source of the build that runs code when it loads: the .nemo of Parakeet TDT 0.6B v3,
# whose configuration and tokenizer NeMo builds the model from.
CODE_RUNNING_SOURCES = [
    (
        "recipes/parakeet-tdt/sources.toml",
        "nvidia/parakeet-tdt-0.6b-v3",
        "parakeet-tdt-0.6b-v3.nemo",
    )
]
CORE_CONTAINERFILES = sorted((REPO_ROOT / "images").glob("*/Containerfile"))
CORE_CONTAINERFILE_IDS = [path.parent.name for path in CORE_CONTAINERFILES]
BASE_PYTHON_CONTAINERFILES = [
    REPO_ROOT / str(image.containerfile) for image in CORE_IMAGES if image.python == BASE_PYTHON
]
BASE_PYTHON_CONTAINERFILE_IDS = [path.parent.name for path in BASE_PYTHON_CONTAINERFILES]
RECIPE_CONTAINERFILES = sorted(RECIPES_DIRECTORY.glob("*/images/*/Containerfile"))
CONTAINERFILES = [*CORE_CONTAINERFILES, *RECIPE_CONTAINERFILES]
CONTAINERFILE_IDS = [str(path.parent.relative_to(REPO_ROOT)) for path in CONTAINERFILES]
COMMENT_PREFIX = "#"
PARSER_DIRECTIVE = re.compile(r"#\s*(syntax|check|escape)\s*=", re.IGNORECASE)
BASE_IMAGE_ALTERNATIVES = "|".join(re.escape(image) for image in sorted(BASE_IMAGES))
ENVIRONMENT_VALUE = "[A-Za-z0-9._/-]+"
UV_ENVIRONMENT_NAMES = (
    "UV_FROZEN|UV_NO_CACHE|UV_LINK_MODE|UV_COMPILE_BYTECODE|UV_PYTHON|UV_PYTHON_DOWNLOADS"
    "|UV_PROJECT_ENVIRONMENT"
)
NAME = "[a-z0-9]+(?:-[a-z0-9]+)*"
FILE = "[A-Za-z0-9._-]+"
STAGE = "[a-z][a-z0-9-]*"
DOWNLOADS = "/build/downloads"
ACCEPTED_FORMS = {
    "a base image of the build, by its digest": re.compile(
        rf"FROM ({BASE_IMAGE_ALTERNATIVES})( AS {STAGE})?"
    ),
    "a working directory": re.compile(r"WORKDIR /[A-Za-z0-9._/-]+"),
    "the variables uv needs, and never one of Python's": re.compile(
        rf"ENV ({UV_ENVIRONMENT_NAMES})={ENVIRONMENT_VALUE}"
        rf"( ({UV_ENVIRONMENT_NAMES})={ENVIRONMENT_VALUE})*"
    ),
    "the core's sources, at the repository's layout under one root": re.compile(
        r"COPY src/mostt_model_builder /opt/model-builder/src/mostt_model_builder"
    ),
    "a recipe's sources, at the repository's layout under one root": re.compile(
        rf"COPY recipes/(?P<recipe>{NAME})/src/(?P<package>mostt_recipe_[a-z0-9_]+) "
        r"/opt/model-builder/recipes/(?P=recipe)/src/(?P=package)"
    ),
    "the images' manifest and lock": re.compile(
        r"COPY images/pyproject\.toml images/uv\.lock /opt/model-builder/images/"
    ),
    "the uv binary from the uv image, by its digest": re.compile(
        rf"COPY --from={re.escape(UV_IMAGE)} /uv /usr/local/bin/uv"
    ),
    "the main install, frozen and without a build, of one extra": re.compile(
        rf"RUN uv sync --frozen --no-dev --no-install-project --no-build --extra {NAME}"
        rf"( --no-install-package {NAME})*"
    ),
    "the build requirements of an sdist, into its build stage": re.compile(
        rf"COPY recipes/{NAME}/images/{NAME}/{NAME}-build-requirements\.txt /build/"
    ),
    "the build requirements downloaded as wheels, by their hashes": re.compile(
        rf"RUN pip download --no-deps --only-binary :all: --require-hashes "
        rf"-r /build/{NAME}-build-requirements\.txt -d {DOWNLOADS}"
    ),
    "the sdist fetched by BuildKit against its hash": re.compile(
        rf"ADD --checksum=sha256:[0-9a-f]{{64}} "
        rf"https://files\.pythonhosted\.org/packages/[0-9a-f/]+/{FILE}\.tar\.gz {DOWNLOADS}/"
    ),
    "the wheel built without network or isolation": re.compile(
        rf"RUN --network=none python -m venv /build/venv && /build/venv/bin/pip install "
        rf"--no-index --no-deps {DOWNLOADS}/{FILE}\.whl && SOURCE_DATE_EPOCH=[0-9]+ "
        rf"/build/venv/bin/pip wheel --no-build-isolation --no-deps --no-index -w /build/wheels "
        rf"{DOWNLOADS}/{FILE}\.tar\.gz"
    ),
    "the built wheel, from its build stage": re.compile(
        rf"COPY --from={STAGE} /build/wheels/{FILE}\.whl /opt/wheels/"
    ),
    "the reviewed hash of the built wheel": re.compile(
        rf"COPY recipes/{NAME}/images/{NAME}/built-wheels\.sha256 /opt/wheels/"
    ),
    "the built wheel checked against its hash, then installed offline": re.compile(
        r"RUN sha256sum -c /opt/wheels/built-wheels\.sha256 && uv pip install "
        rf"--python /opt/venv/bin/python --no-deps --no-index --offline /opt/wheels/{FILE}\.whl "
        r"&& rm -rf /opt/wheels"
    ),
    "the build argument of the tree": re.compile(r"ARG TREE_HASH"),
    "the label of the tree": re.compile(r"LABEL mostt-model-builder\.tree=\$\{TREE_HASH\}"),
}
LABEL_LINES = ["ARG TREE_HASH", "LABEL mostt-model-builder.tree=${TREE_HASH}"]
STAGE_DECLARATION = re.compile(rf"FROM \S+ AS (?P<stage>{STAGE})")
STAGE_COPY = re.compile(rf"COPY --from=(?P<stage>{STAGE}) ")
SDIST_ADDITION = re.compile(r"ADD --checksum=(?P<hash>sha256:[0-9a-f]{64}) (?P<url>\S+) ")
MAIN_INSTALL = re.compile(rf"RUN uv sync .* --extra (?P<extra>{NAME})(?P<excluded>.*)")
EXCLUDED_PACKAGE = re.compile(rf"--no-install-package (?P<package>{NAME})")
COPIED_SOURCES = re.compile(r"COPY (?!--from=)(?P<sources>.+) \S+")
BUILT_WHEEL_INSTALL = re.compile(rf"RUN sha256sum -c .* /opt/wheels/(?P<wheel>{FILE}\.whl) ")
BUILT_WHEEL_HASH_COPY = re.compile(r"COPY \S+/built-wheels\.sha256 /opt/wheels/")
BUILT_WHEELS_FILE_NAME = "built-wheels.sha256"
DOCKERIGNORE_SUFFIX = ".dockerignore"
DOCKERIGNORE_EXCLUSIONS = ["**/__pycache__", "**/*.pyc"]
CONTAINERFILE_WITH_A_DIRECTIVE = f"# syntax=docker/dockerfile:1\nFROM {PYTHON_BASE_IMAGE}\n"
CONTAINERFILE_WITH_A_LATE_DIRECTIVE = f"FROM {PYTHON_BASE_IMAGE}\n# check=skip=all\n"
CONTAINERFILE_WITH_A_DOWNLOAD = f"FROM {PYTHON_BASE_IMAGE}\nRUN curl -o /x https://example.org/x\n"
CONTAINERFILE_WITH_AN_UNCHECKED_ADDITION = (
    f"FROM {PYTHON_BASE_IMAGE}\nADD https://files.pythonhosted.org/packages/a/x.tar.gz "
    "/build/downloads/\n"
)
CONTAINERFILE_WITH_A_BUILD = (
    f"FROM {PYTHON_BASE_IMAGE}\n"
    "RUN uv sync --frozen --no-dev --no-install-project --extra parakeet-tdt-export\n"
)
CONTAINERFILE_WITH_AN_INSTALL_FROM_AN_INDEX = f"FROM {PYTHON_BASE_IMAGE}\nRUN pip install torch\n"
HEAVY_LIBRARIES_CONTRACT = "The libraries of the models load only in the steps that need them"
CORE_LAYER_MODULES = (
    "mostt_model_builder.bootstrap",
    "mostt_model_builder.domain",
    "mostt_model_builder.application",
    "mostt_model_builder.adapters",
    "mostt_model_builder.steps",
)
HOST_ENTRY_MODULE = "mostt_model_builder.host_main"


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


def list_strings(entries: object) -> list[str]:
    if isinstance(entries, str):
        return [entries]
    if not isinstance(entries, list):
        return []
    return [str(entry) for entry in entries]


def find_contracts_of(source_module: str) -> list[dict[str, Any]]:
    return [
        contract
        for contract in MANIFEST["tool"]["importlinter"]["contracts"]
        if source_module in list_strings(contract.get("source_modules"))
    ]


def list_forbidden_modules(source_module: str) -> set[str]:
    return {
        module
        for contract in find_contracts_of(source_module)
        for module in list_strings(contract.get("forbidden_modules"))
    }


def list_built_packages() -> set[str]:
    return {
        str(entry["name"]) for entry in IMAGES_MANIFEST["tool"]["uv"].get("dependency-metadata", [])
    }


def find_sdists(package_names: set[str]) -> set[tuple[str, str]]:
    return {
        (str(package["sdist"]["url"]), str(package["sdist"]["hash"]))
        for package in IMAGES_LOCK["package"]
        if package["name"] in package_names and "sdist" in package
    }


def read_reviewed_wheels(containerfile: Path) -> set[str]:
    hash_file = containerfile.parent / BUILT_WHEELS_FILE_NAME
    if not hash_file.exists():
        return set()
    return {
        line.split()[-1].rsplit("/", 1)[-1]
        for line in read_instruction_lines(hash_file.read_text(encoding="utf-8"))
    }


def find_undeclared_stages(instructions: list[str]) -> list[str]:
    declared_stages = {
        match["stage"] for line in instructions if (match := STAGE_DECLARATION.fullmatch(line))
    }
    copied_stages = [match["stage"] for line in instructions if (match := STAGE_COPY.match(line))]
    return [stage for stage in copied_stages if stage not in declared_stages]


def find_unchecked_wheels(containerfile: Path, instructions: list[str]) -> list[str]:
    reviewed_wheels = read_reviewed_wheels(containerfile)
    first_hash_copy = next(
        (index for index, line in enumerate(instructions) if BUILT_WHEEL_HASH_COPY.fullmatch(line)),
        len(instructions),
    )
    return [
        match["wheel"]
        for index, line in enumerate(instructions)
        if (match := BUILT_WHEEL_INSTALL.match(line))
        and (match["wheel"] not in reviewed_wheels or index < first_hash_copy)
    ]


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

        def test_should_give_each_one_to_the_command_line_under_its_directorys_name(
            self, recipe_entries: list[str]
        ) -> None:
            assert recipe_entries == RECIPE_NAMES

    class TestWhenEverySourcesFileIsRead:
        @pytest.fixture
        def marked_sources(self) -> list[tuple[str, str, str]]:
            return [
                (str(sources_file.relative_to(REPO_ROOT)), pin.repository, pin.path)
                for sources_file in SOURCES_FILES
                for pin in parse_source_pins(sources_file.read_text(encoding="utf-8"))
                if pin.runs_code_when_loaded
            ]

        def test_should_read_the_cores_and_each_recipes(self) -> None:
            assert len(SOURCES_FILES) == 1 + EXPECTED_RECIPE_COUNT

        def test_should_mark_exactly_one_file_as_running_code_when_it_loads_the_v3_nemo(
            self, marked_sources: list[tuple[str, str, str]]
        ) -> None:
            assert marked_sources == CODE_RUNNING_SOURCES

    class TestTheRecipeNameGrammar:
        def test_should_be_the_one_under_which_the_bootstrap_appends_a_recipe(self) -> None:
            assert RECIPE_NAME_GRAMMAR.pattern == BOOTSTRAP_RECIPE_NAME_GRAMMAR.pattern

    @pytest.mark.parametrize("recipe_name", RECIPE_NAMES)
    class TestWhenARecipesConfigurationIsRead:
        @pytest.fixture
        def package(self, recipe_name: str) -> str:
            return find_package_of_owner(recipe_name)

        def test_should_give_import_linter_its_package_as_a_root(self, package: str) -> None:
            assert package in MANIFEST["tool"]["importlinter"]["root_packages"]

        def test_should_keep_every_layer_of_the_core_from_importing_it(self, package: str) -> None:
            assert [
                module
                for module in CORE_LAYER_MODULES
                if package not in list_forbidden_modules(module)
            ] == []

        def test_should_keep_the_host_from_importing_its_steps(self, package: str) -> None:
            assert f"{package}.steps" in list_forbidden_modules(HOST_ENTRY_MODULE)

        def test_should_keep_its_domain_and_its_contract_off_the_outer_layers_and_the_outside(
            self, package: str
        ) -> None:
            assert {
                "mostt_model_builder.application",
                "mostt_model_builder.adapters",
                "mostt_model_builder.host_main",
                f"{package}.steps",
                "subprocess",
                "socket",
            } <= list_forbidden_modules(f"{package}.domain") & list_forbidden_modules(
                f"{package}.recipe"
            )

        def test_should_keep_its_steps_off_the_application_the_adapters_and_the_host(
            self, package: str
        ) -> None:
            assert {
                "mostt_model_builder.application",
                "mostt_model_builder.adapters",
                "mostt_model_builder.host_main",
            } <= list_forbidden_modules(f"{package}.steps")

        def test_should_keep_the_libraries_of_the_models_out_of_its_domain_and_its_contract(
            self, package: str
        ) -> None:
            assert [
                module
                for module in (f"{package}.domain", f"{package}.recipe")
                if not any(
                    str(contract["name"]).startswith(HEAVY_LIBRARIES_CONTRACT)
                    for contract in find_contracts_of(module)
                )
            ] == []

        def test_should_give_its_sources_to_ruff_mypy_and_pytest(self, recipe_name: str) -> None:
            source = f"recipes/{recipe_name}/src"
            assert [
                tool
                for tool, paths in (
                    ("ruff", MANIFEST["tool"]["ruff"]["src"]),
                    ("mypy's path", MANIFEST["tool"]["mypy"]["mypy_path"]),
                    ("mypy's files", MANIFEST["tool"]["mypy"]["files"]),
                    ("pytest", MANIFEST["tool"]["pytest"]["ini_options"]["pythonpath"]),
                )
                if source not in paths
            ] == []

        def test_should_give_its_tests_to_mypy_and_pytest(self, recipe_name: str) -> None:
            tests = f"recipes/{recipe_name}/tests"
            assert [
                tool
                for tool, paths in (
                    ("mypy", MANIFEST["tool"]["mypy"]["files"]),
                    ("pytest", MANIFEST["tool"]["pytest"]["ini_options"]["testpaths"]),
                )
                if tests not in paths
            ] == []

        def test_should_hold_at_least_one_test(self, recipe_name: str) -> None:
            assert len(list((RECIPES_DIRECTORY / recipe_name / "tests").glob("test_*.py"))) > 0

    class TestWhenTheImagesAreListed:
        def test_should_find_one_containerfile_per_image_of_the_core(self) -> None:
            assert sorted(image.name for image in CORE_IMAGES) == CORE_CONTAINERFILE_IDS

        def test_should_find_one_containerfile_per_image_of_each_recipe(self) -> None:
            assert sorted(
                f"{contract.name}/{image.name}"
                for contract in RECIPES.values()
                for image in contract.images
            ) == [f"{path.parents[2].name}/{path.parent.name}" for path in RECIPE_CONTAINERFILES]

    @pytest.mark.parametrize("containerfile", CONTAINERFILES, ids=CONTAINERFILE_IDS)
    class TestWhenAContainerfileIsRead:
        @pytest.fixture
        def instructions(self, containerfile: Path) -> list[str]:
            return list_instructions(containerfile.read_text(encoding="utf-8"))

        def test_should_hold_only_the_accepted_forms_so_nothing_else_reaches_an_image(
            self, containerfile: Path, instructions: list[str]
        ) -> None:
            assert len(instructions) > 0
            assert find_refused_lines(containerfile.read_text(encoding="utf-8")) == []

        def test_should_end_with_the_label_of_the_tree_so_a_change_of_code_reuses_the_rest(
            self, instructions: list[str]
        ) -> None:
            assert instructions[-2:] == LABEL_LINES

        def test_should_copy_from_no_stage_it_does_not_declare(
            self, instructions: list[str]
        ) -> None:
            assert find_undeclared_stages(instructions) == []

        def test_should_fetch_only_the_sdists_that_the_lock_gives_for_a_package_a_stage_builds(
            self, instructions: list[str]
        ) -> None:
            additions = {
                (match["url"], match["hash"])
                for line in instructions
                if (match := SDIST_ADDITION.match(line))
            }
            assert additions <= find_sdists(list_built_packages())

        def test_should_leave_out_of_its_main_install_exactly_the_packages_a_stage_builds(
            self, instructions: list[str]
        ) -> None:
            excluded_packages = {
                package_match["package"]
                for line in instructions
                if (match := MAIN_INSTALL.fullmatch(line))
                for package_match in EXCLUDED_PACKAGE.finditer(match["excluded"])
            }
            has_build_stage = any(SDIST_ADDITION.match(line) for line in instructions)
            assert excluded_packages == (list_built_packages() if has_build_stage else set())

        def test_should_check_each_built_wheel_against_its_reviewed_hash_before_installing_it(
            self, containerfile: Path, instructions: list[str]
        ) -> None:
            assert find_unchecked_wheels(containerfile, instructions) == []

    @pytest.mark.parametrize(
        "containerfile", BASE_PYTHON_CONTAINERFILES, ids=BASE_PYTHON_CONTAINERFILE_IDS
    )
    class TestWhenTheContainerfileOfACoreImageWithoutAnEnvironmentIsRead:
        @pytest.fixture
        def instructions(self, containerfile: Path) -> list[str]:
            return list_instructions(containerfile.read_text(encoding="utf-8"))

        def test_should_run_no_command_since_its_steps_run_on_the_bases_python(
            self, instructions: list[str]
        ) -> None:
            assert [line for line in instructions if line.startswith("RUN")] == []

    class TestWhenTheMainInstallsAreListed:
        @pytest.fixture
        def installed_extras(self) -> list[str]:
            return sorted(
                match["extra"]
                for containerfile in CONTAINERFILES
                for line in list_instructions(containerfile.read_text(encoding="utf-8"))
                if (match := MAIN_INSTALL.fullmatch(line))
            )

        def test_should_install_each_extra_of_the_images_environment_in_exactly_one_image(
            self, installed_extras: list[str]
        ) -> None:
            assert installed_extras == sorted(
                IMAGES_MANIFEST["project"].get("optional-dependencies", {})
            )

        def test_should_declare_in_the_images_environment_the_extra_of_each_recipe_image(
            self, installed_extras: list[str]
        ) -> None:
            assert [
                image.extra
                for contract in RECIPES.values()
                for image in contract.images
                if image.extra is not None and image.extra not in installed_extras
            ] == []

    @pytest.mark.parametrize("containerfile", CONTAINERFILES, ids=CONTAINERFILE_IDS)
    class TestWhenItsDockerignoreIsRead:
        @pytest.fixture
        def entries(self, containerfile: Path) -> list[str]:
            dockerignore = containerfile.with_name(containerfile.name + DOCKERIGNORE_SUFFIX)
            return read_instruction_lines(dockerignore.read_text(encoding="utf-8"))

        def test_should_exclude_everything_first(self, entries: list[str]) -> None:
            assert entries[0] == "*"

        def test_should_include_again_exactly_what_its_image_copies(
            self, containerfile: Path, entries: list[str]
        ) -> None:
            copied_sources = {
                f"!{source}"
                for line in list_instructions(containerfile.read_text(encoding="utf-8"))
                if (match := COPIED_SOURCES.fullmatch(line))
                for source in match["sources"].split()
            }
            assert {entry for entry in entries if entry.startswith("!")} == copied_sources

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
            pytest.param(
                CONTAINERFILE_WITH_AN_UNCHECKED_ADDITION,
                "ADD https://files.pythonhosted.org/packages/a/x.tar.gz /build/downloads/",
                id="an addition without a hash",
            ),
            pytest.param(
                CONTAINERFILE_WITH_A_BUILD,
                "RUN uv sync --frozen --no-dev --no-install-project --extra parakeet-tdt-export",
                id="a main install that may build",
            ),
            pytest.param(
                CONTAINERFILE_WITH_AN_INSTALL_FROM_AN_INDEX,
                "RUN pip install torch",
                id="an install from an index",
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
