from pathlib import Path

from mostt_model_builder.domain.containers.container_paths import SOURCES_IN_THE_CONTAINER

# A step names a source by its repository and its path alone: only the host reads the revision
# that the source's pin gives.
REVISION_WILDCARD = "*"


class BoundSourceError(Exception):
    pass


def find_bound_source(
    repository: str, path: str, sources_directory: Path = Path(SOURCES_IN_THE_CONTAINER)
) -> Path:
    matches = sorted((sources_directory / repository).glob(f"{REVISION_WILDCARD}/{path}"))
    if len(matches) != 1:
        raise BoundSourceError(
            f"{len(matches)} files are bound for {path} of {repository}, where its step expects one"
        )
    return matches[0]
