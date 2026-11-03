import os
import shutil
from pathlib import Path

from mostt_model_builder.domain.containers.container_steps import OUTPUT_IN_THE_CONTAINER

REMOVABLE_DIRECTORY_MODE = 0o700


def make_tree_removable(directory: Path) -> None:
    os.chmod(directory, REMOVABLE_DIRECTORY_MODE)
    for parent, directory_names, _ in os.walk(directory):
        for name in directory_names:
            child = Path(parent, name)
            if not child.is_symlink():
                os.chmod(child, REMOVABLE_DIRECTORY_MODE)


def empty_directory(directory: Path) -> None:
    for entry in directory.iterdir():
        if entry.is_dir() and not entry.is_symlink():
            make_tree_removable(entry)
            shutil.rmtree(entry)
        else:
            entry.unlink()


def main() -> None:
    empty_directory(Path(OUTPUT_IN_THE_CONTAINER))


if __name__ == "__main__":
    main()
