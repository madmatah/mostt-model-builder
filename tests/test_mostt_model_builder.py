import json
import subprocess
import sys
import sysconfig
from pathlib import Path

import pytest

PACKAGE_NAME = "mostt_model_builder"
HOST_ENTRY_MODULE = f"{PACKAGE_NAME}.host_main"
SOURCE_DIRECTORY = Path(__file__).resolve().parent.parent / "src"
PACKAGE_DIRECTORY = SOURCE_DIRECTORY / PACKAGE_NAME
STANDARD_LIBRARY_DIRECTORIES = (
    Path(sysconfig.get_paths()["stdlib"]).resolve(),
    Path(sysconfig.get_paths()["platstdlib"]).resolve(),
)
ORIGINS_WITHOUT_A_FILE = ("built-in", "frozen")
SCRIPT_MODULE_NAME = "__main__"
PACKAGE_SEPARATOR = "."
STANDARD_LIBRARY_STEPS = tuple(
    f"{PACKAGE_NAME}.steps.{step}"
    for step in (
        "fleurs_audio",
        "fleurs_check",
        "clean_output",
        "output_modes",
        "control_group_peaks",
        "lock_resolution",
    )
)
ISOLATED_MODULES = (HOST_ENTRY_MODULE, *STANDARD_LIBRARY_STEPS)
LOADED_MODULES_QUERY = """
import importlib
import json
import sys

sys.path.append(sys.argv[1])
importlib.import_module(sys.argv[2])

origins = {}
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


def is_trusted_module(name: str, origins: dict[str, str | None]) -> bool:
    origin = origins[name]
    parent_name, _, _ = name.rpartition(PACKAGE_SEPARATOR)
    # Python 3.12's typing registers typing.io and typing.re in memory, without a spec: a module
    # that no file holds is only as trusted as the package that made it.
    if origin is None and parent_name in origins:
        return is_trusted_module(parent_name, origins)
    return is_trusted_origin(origin)


def list_foreign_modules(origins: dict[str, str | None]) -> dict[str, str | None]:
    return {
        name: origin
        for name, origin in origins.items()
        if name != SCRIPT_MODULE_NAME and not is_trusted_module(name, origins)
    }


class TestTheMosttModelBuilderPackage:
    class TestTheImport:
        @pytest.mark.parametrize("module_name", ISOLATED_MODULES, scope="class")
        class TestWhenAPythonWithoutItsEnvironmentAndSitePackagesImportsAModuleOfTheHostOrAStep:
            @pytest.fixture(scope="class")
            @classmethod
            def module_origins(cls, module_name: str) -> dict[str, str | None]:
                completed_import = subprocess.run(
                    [
                        sys.executable,
                        "-I",
                        "-S",
                        "-B",
                        "-c",
                        LOADED_MODULES_QUERY,
                        str(SOURCE_DIRECTORY),
                        module_name,
                    ],
                    capture_output=True,
                    check=True,
                    text=True,
                    timeout=ISOLATED_IMPORT_TIMEOUT_S,
                )
                return parse_module_origins(completed_import.stdout)

            def test_should_load_the_module(
                self, module_origins: dict[str, str | None], module_name: str
            ) -> None:
                assert module_name in module_origins

            def test_should_load_only_the_standard_library_and_itself_so_no_third_party_code_runs(
                self, module_origins: dict[str, str | None]
            ) -> None:
                assert list_foreign_modules(module_origins) == {}
