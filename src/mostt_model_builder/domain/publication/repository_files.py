from collections.abc import Collection, Sequence

from mostt_model_builder.domain.publication.published_tree import HUB_ATTRIBUTES_FILE_NAME
from mostt_model_builder.domain.sources.source_pins import REPOSITORY_GRAMMAR


class RepositoryFilesError(ValueError):
    pass


def judge_repository_name(repository: str) -> None:
    if not REPOSITORY_GRAMMAR.fullmatch(repository):
        raise RepositoryFilesError(
            f"the repository {repository!a} is not <namespace>/<name> under "
            f"{REPOSITORY_GRAMMAR.pattern}"
        )


# Any other file belongs to another repository, or would stay beside the build's own.
def judge_repository_files(listed_files: Sequence[str], published_names: Collection[str]) -> None:
    allowed = {HUB_ATTRIBUTES_FILE_NAME, *published_names}
    foreign = sorted(name for name in listed_files if name not in allowed)
    if foreign:
        raise RepositoryFilesError(
            f"the repository holds {len(foreign)} files that are not the build's, such as "
            f"{foreign[0]!a}: publish into a repository that holds only the Hub's "
            f"{HUB_ATTRIBUTES_FILE_NAME} or an earlier publication of this build"
        )


# A slip on the command line would send the files to another repository.
def judge_repository_confirmation(repository: str, typed_repository: str) -> None:
    if typed_repository != repository:
        raise RepositoryFilesError(
            f"the repository typed again differs from {repository}: nothing was published"
        )
