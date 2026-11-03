"""Run by path, before any package of the project is on the path, so it may import none of them."""

import os
import re
import runpy
import stat
import sys
from collections.abc import Sequence
from pathlib import Path

ROOT_REFUSED_EXIT_STATUS = 77
USER_ID_NAMES = ("real user id", "effective user id", "saved user id")
GROUP_ID_NAMES = ("real group id", "effective group id", "saved group id")
RECIPE_NAME_GRAMMAR = re.compile(r"[a-z0-9][a-z0-9-]{0,31}")
ENTRY_MODULE_GRAMMAR = re.compile(r"[a-z_][a-z0-9_]*(\.[a-z_][a-z0-9_]*)*")
SOURCE_DIRECTORY_NAME = "src"
RECIPES_DIRECTORY_NAME = "recipes"
ROOT_DEPTH_UNDER_THE_BOOTSTRAP = 2


def find_root_ids(user_ids: Sequence[int], group_ids: Sequence[int]) -> list[str]:
    named_ids = zip((*USER_ID_NAMES, *GROUP_ID_NAMES), (*user_ids, *group_ids), strict=True)
    return [name for name, process_id in named_ids if process_id == 0]


def refuse_root(user_ids: Sequence[int], group_ids: Sequence[int]) -> None:
    root_ids = find_root_ids(user_ids, group_ids)
    if root_ids:
        print(
            f"model-build: refusing to run as root: the {', '.join(root_ids)} is 0",
            file=sys.stderr,
        )
        raise SystemExit(ROOT_REFUSED_EXIT_STATUS)


def is_real_directory(path: Path) -> bool:
    try:
        return stat.S_ISDIR(path.lstat().st_mode)
    except FileNotFoundError:
        return False


def list_package_directories(root: Path) -> list[Path]:
    package_directories = [root / SOURCE_DIRECTORY_NAME]
    recipes_directory = root / RECIPES_DIRECTORY_NAME
    if not is_real_directory(recipes_directory):
        return package_directories
    for recipe_name in sorted(os.listdir(recipes_directory)):
        recipe_directory = recipes_directory / recipe_name
        recipe_source_directory = recipe_directory / SOURCE_DIRECTORY_NAME
        if (
            RECIPE_NAME_GRAMMAR.fullmatch(recipe_name)
            and is_real_directory(recipe_directory)
            and is_real_directory(recipe_source_directory)
        ):
            package_directories.append(recipe_source_directory)
    return package_directories


def refuse_malformed_entry_module(entry_module: str) -> None:
    if not ENTRY_MODULE_GRAMMAR.fullmatch(entry_module):
        raise SystemExit(
            f"model-build: the entry module {entry_module!a} is not a dotted module name"
        )


def main() -> None:
    refuse_root(os.getresuid(), os.getresgid())
    root = Path(__file__).resolve().parents[ROOT_DEPTH_UNDER_THE_BOOTSTRAP]
    sys.path.extend(str(directory) for directory in list_package_directories(root))
    if len(sys.argv) < 2:
        raise SystemExit("model-build: the bootstrap needs the entry module to run")
    entry_module, *entry_arguments = sys.argv[1:]
    refuse_malformed_entry_module(entry_module)
    sys.argv[:] = [entry_module, *entry_arguments]
    runpy.run_module(entry_module, run_name="__main__", alter_sys=True)


if __name__ == "__main__":
    main()
