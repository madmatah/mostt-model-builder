import re
import tomllib
from pathlib import Path

import pytest

HOST_MANIFEST_PATH = Path(__file__).resolve().parent.parent / "pyproject.toml"
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
