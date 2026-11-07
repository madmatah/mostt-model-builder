import ast
from collections.abc import Mapping
from pathlib import PurePosixPath

from mostt_model_builder.domain.containers.container_steps import (
    CORE_PACKAGE,
    IMAGES_DIRECTORY_NAME,
    RECIPE_PACKAGE_PREFIX,
)

MODULE_SEPARATOR = "."
PYTHON_SOURCE_SUFFIX = ".py"
PACKAGE_FILE_NAME = "__init__.py"
SOURCE_DIRECTORY_NAME = "src"
RECIPES_DIRECTORY_NAME = "recipes"
BOOTSTRAP_PATH = PurePosixPath(SOURCE_DIRECTORY_NAME, CORE_PACKAGE, "bootstrap.py")
# The code that a step runs and the files that its image is built from live under these directories
# of the tree, so the keys read HEAD's tree there alone.
KEYED_TREE_DIRECTORIES = frozenset(
    {SOURCE_DIRECTORY_NAME, RECIPES_DIRECTORY_NAME, IMAGES_DIRECTORY_NAME}
)


class ImportClosureError(ValueError):
    pass


def is_project_module(module: str) -> bool:
    package = module.split(MODULE_SEPARATOR, 1)[0]
    return package == CORE_PACKAGE or package.startswith(RECIPE_PACKAGE_PREFIX)


def find_source_root(path: PurePosixPath) -> int | None:
    parts = path.parts
    if len(parts) > 1 and parts[0] == SOURCE_DIRECTORY_NAME:
        return 1
    if len(parts) > 3 and parts[0] == RECIPES_DIRECTORY_NAME and parts[2] == SOURCE_DIRECTORY_NAME:
        return 3
    return None


def map_project_modules(files: Mapping[PurePosixPath, bytes]) -> dict[str, PurePosixPath]:
    modules: dict[str, PurePosixPath] = {}
    for path in files:
        root_length = find_source_root(path)
        if root_length is None or path.suffix != PYTHON_SOURCE_SUFFIX:
            continue
        module_parts = path.parts[root_length:]
        if module_parts[-1] == PACKAGE_FILE_NAME:
            module_parts = module_parts[:-1]
        else:
            module_parts = (*module_parts[:-1], PurePosixPath(module_parts[-1]).stem)
        module = MODULE_SEPARATOR.join(module_parts)
        if module_parts and is_project_module(module):
            modules[module] = path
    return modules


def is_namespace_package(module: str, modules: Mapping[str, PurePosixPath]) -> bool:
    prefix = module + MODULE_SEPARATOR
    return any(other.startswith(prefix) for other in modules)


def list_parent_modules(module: str) -> list[str]:
    parts = module.split(MODULE_SEPARATOR)
    return [MODULE_SEPARATOR.join(parts[:length]) for length in range(1, len(parts) + 1)]


def list_imported_modules(
    source: bytes, path: PurePosixPath, modules: Mapping[str, PurePosixPath]
) -> list[str]:
    try:
        tree = ast.parse(source, filename=str(path))
    except (SyntaxError, ValueError) as error:
        raise ImportClosureError(f"{path} does not parse as Python: {error}") from error
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                raise ImportClosureError(
                    f"{path} imports relatively, at line {node.lineno}, where the project names "
                    "every module in full"
                )
            if node.module is None:
                continue
            imported.append(node.module)
            imported.extend(
                submodule
                for alias in node.names
                if (submodule := f"{node.module}{MODULE_SEPARATOR}{alias.name}") in modules
            )
    return [module for module in imported if is_project_module(module)]


# Every import counts, those nested in a function, a try block or a check of TYPE_CHECKING
# included: a closure larger than what runs only reruns a step more often, never less.
def close_imports(
    entry_module: str, files: Mapping[PurePosixPath, bytes]
) -> tuple[PurePosixPath, ...]:
    modules = map_project_modules(files)
    if entry_module not in modules:
        raise ImportClosureError(f"the entry module {entry_module} has no file in HEAD's tree")
    closure: set[PurePosixPath] = set()
    pending = list_parent_modules(entry_module)
    seen = set(pending)
    while pending:
        module = pending.pop()
        path = modules.get(module)
        if path is None:
            if is_namespace_package(module, modules):
                continue
            raise ImportClosureError(
                f"{module} is imported by the closure of {entry_module}, and HEAD's tree holds "
                "no file of it"
            )
        closure.add(path)
        for imported in list_imported_modules(files[path], path, modules):
            for parent in list_parent_modules(imported):
                if parent not in seen:
                    seen.add(parent)
                    pending.append(parent)
    return tuple(sorted(closure))


def collect_code_files(
    entry_module: str, files: Mapping[PurePosixPath, bytes]
) -> dict[PurePosixPath, bytes]:
    return {path: files[path] for path in close_imports(entry_module, files)}
