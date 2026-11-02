import json
import subprocess
import sys
import sysconfig
from pathlib import Path

import pytest

PACKAGE_NAME = "mostt_model_builder"
SOURCE_DIRECTORY = Path(__file__).resolve().parent.parent / "src"
PACKAGE_DIRECTORY = SOURCE_DIRECTORY / PACKAGE_NAME
STANDARD_LIBRARY_DIRECTORIES = (
    Path(sysconfig.get_paths()["stdlib"]).resolve(),
    Path(sysconfig.get_paths()["platstdlib"]).resolve(),
)
ORIGINS_WITHOUT_A_FILE = ("built-in", "frozen")
SCRIPT_MODULE_NAME = "__main__"
LOADED_MODULES_QUERY = f"""
import json
import sys

sys.path.append(sys.argv[1])
import {PACKAGE_NAME}

origins = {{}}
for name, module in sys.modules.items():
    spec = getattr(module, "__spec__", None)
    origins[name] = None if spec is None else spec.origin
print(json.dumps(origins))
"""
ISOLATED_IMPORT_TIMEOUT_S = 60


def parse_module_origins(report: str) -> dict[str, str | None]:
    parsed: object = json.loads(report)
    if not isinstance(parsed, dict):
        raise TypeError(f"the module report is not an object: {report!r}")
    origins: dict[str, str | None] = {}
    for name, origin in parsed.items():
        if not isinstance(name, str) or not (origin is None or isinstance(origin, str)):
            raise TypeError(f"the module report holds a malformed entry: {name!r}: {origin!r}")
        origins[name] = origin
    return origins


def is_trusted_origin(origin: str | None) -> bool:
    if origin in ORIGINS_WITHOUT_A_FILE:
        return True
    if origin is None:
        return False
    origin_path = Path(origin).resolve()
    trusted_directories = (*STANDARD_LIBRARY_DIRECTORIES, PACKAGE_DIRECTORY)
    return any(origin_path.is_relative_to(directory) for directory in trusted_directories)


def list_foreign_modules(origins: dict[str, str | None]) -> dict[str, str | None]:
    return {
        name: origin
        for name, origin in origins.items()
        if name != SCRIPT_MODULE_NAME and not is_trusted_origin(origin)
    }


class TestTheMosttModelBuilderPackage:
    class TestTheImport:
        class TestWhenAPythonWithoutItsEnvironmentAndSitePackagesImportsIt:
            @pytest.fixture(scope="class")
            @classmethod
            def module_origins(cls) -> dict[str, str | None]:
                completed_import = subprocess.run(
                    [
                        sys.executable,
                        "-I",
                        "-S",
                        "-B",
                        "-c",
                        LOADED_MODULES_QUERY,
                        str(SOURCE_DIRECTORY),
                    ],
                    capture_output=True,
                    check=True,
                    text=True,
                    timeout=ISOLATED_IMPORT_TIMEOUT_S,
                )
                return parse_module_origins(completed_import.stdout)

            def test_should_load_the_package(self, module_origins: dict[str, str | None]) -> None:
                assert PACKAGE_NAME in module_origins

            def test_should_load_only_the_standard_library_and_itself_so_no_third_party_code_runs(
                self, module_origins: dict[str, str | None]
            ) -> None:
                assert list_foreign_modules(module_origins) == {}
