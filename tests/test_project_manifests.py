import re
import tomllib
from pathlib import Path

import pytest

from mostt_model_builder.domain.containers.base_images import UV_VERSION
from mostt_model_builder.domain.lock.declared_requirements import (
    DeclaredRequirement,
    read_declared_requirements,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
HOST_MANIFEST_PATH = REPO_ROOT / "pyproject.toml"
IMAGES_MANIFEST_PATH = REPO_ROOT / "images" / "pyproject.toml"
IMAGES_UV_KEYS = {"package", "no-build", "environments", "exclude-newer", "required-version"}
IMAGES_REQUIREMENT_COUNT = 0
IMAGES_ENVIRONMENTS = ["sys_platform == 'linux' and platform_machine == 'x86_64'"]
EXCLUDE_NEWER_GRAMMAR = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z")
HOST_UV_SETTINGS = {"package": False, "no-build": True}
DEVELOPMENT_GROUP = "dev"
EXACT_PIN_GRAMMAR = re.compile(r"[a-z0-9]+(-[a-z0-9]+)*==[0-9]+(\.[0-9]+)*")


def read_table(table: dict[str, object], dotted_key: str) -> dict[str, object]:
    current_table = table
    for key in dotted_key.split("."):
        nested_table = current_table[key]
        if not isinstance(nested_table, dict):
            raise TypeError(f"{dotted_key} is not a table: {nested_table!r}")
        current_table = nested_table
    return current_table


def read_string_list(table: dict[str, object], key: str) -> list[str]:
    entries = table[key]
    if not isinstance(entries, list) or not all(isinstance(entry, str) for entry in entries):
        raise TypeError(f"{key} is not a list of strings: {entries!r}")
    return [str(entry) for entry in entries]


def list_unpinned_requirements(requirements: list[str]) -> list[str]:
    return [
        requirement for requirement in requirements if not EXACT_PIN_GRAMMAR.fullmatch(requirement)
    ]


class TestTheProjectManifests:
    class TestWhenTheHostManifestIsRead:
        @pytest.fixture
        def host_manifest(self) -> dict[str, object]:
            return tomllib.loads(HOST_MANIFEST_PATH.read_text())

        def test_should_set_exactly_no_package_and_no_build_so_uv_builds_nothing_on_the_host(
            self, host_manifest: dict[str, object]
        ) -> None:
            assert read_table(host_manifest, "tool.uv") == HOST_UV_SETTINGS

        def test_should_declare_no_dependency_so_the_host_installs_only_the_development_tools(
            self, host_manifest: dict[str, object]
        ) -> None:
            assert read_string_list(read_table(host_manifest, "project"), "dependencies") == []

        def test_should_pin_every_development_dependency_exactly(
            self, host_manifest: dict[str, object]
        ) -> None:
            development_requirements = read_string_list(
                read_table(host_manifest, "dependency-groups"), DEVELOPMENT_GROUP
            )
            assert len(development_requirements) > 0
            assert list_unpinned_requirements(development_requirements) == []

    class TestWhenTheImagesManifestIsRead:
        @pytest.fixture
        def images_manifest(self) -> dict[str, object]:
            return tomllib.loads(IMAGES_MANIFEST_PATH.read_text())

        def test_should_target_the_python_of_the_images(
            self, images_manifest: dict[str, object]
        ) -> None:
            assert read_table(images_manifest, "project")["requires-python"] == "==3.12.*"

        def test_should_declare_no_dependency_outside_an_extra(
            self, images_manifest: dict[str, object]
        ) -> None:
            assert read_string_list(read_table(images_manifest, "project"), "dependencies") == []

        def test_should_declare_no_group_so_the_lock_records_none(
            self, images_manifest: dict[str, object]
        ) -> None:
            assert "dependency-groups" not in images_manifest

        def test_should_hold_exactly_the_uv_settings_of_its_shape(
            self, images_manifest: dict[str, object]
        ) -> None:
            assert set(read_table(images_manifest, "tool.uv")) == IMAGES_UV_KEYS

        def test_should_install_no_project_and_allow_a_build_only_in_the_lock_container(
            self, images_manifest: dict[str, object]
        ) -> None:
            uv_settings = read_table(images_manifest, "tool.uv")
            assert (uv_settings["package"], uv_settings["no-build"]) == (False, False)

        def test_should_resolve_for_linux_on_x86_64_alone(
            self, images_manifest: dict[str, object]
        ) -> None:
            assert read_table(images_manifest, "tool.uv")["environments"] == IMAGES_ENVIRONMENTS

        def test_should_exclude_every_upload_after_a_fixed_instant(
            self, images_manifest: dict[str, object]
        ) -> None:
            exclude_newer = read_table(images_manifest, "tool.uv")["exclude-newer"]
            assert isinstance(exclude_newer, str)
            assert EXCLUDE_NEWER_GRAMMAR.fullmatch(exclude_newer)

        def test_should_require_the_uv_of_the_uv_image_so_any_other_uv_refuses_the_project(
            self, images_manifest: dict[str, object]
        ) -> None:
            assert read_table(images_manifest, "tool.uv")["required-version"] == f"=={UV_VERSION}"

    class TestWhenTheImagesRequirementsAreRead:
        @pytest.fixture
        def requirements(self) -> tuple[DeclaredRequirement, ...]:
            images_manifest = tomllib.loads(IMAGES_MANIFEST_PATH.read_text())
            return read_declared_requirements(
                read_table(images_manifest, "project").get("optional-dependencies")
            )

        def test_should_read_each_one_as_an_exact_pin_through_the_strict_parser(
            self, requirements: tuple[DeclaredRequirement, ...]
        ) -> None:
            assert len(requirements) == IMAGES_REQUIREMENT_COUNT
