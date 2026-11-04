from collections.abc import Sequence

from mostt_model_builder.domain.publication.publication_request import (
    PublicationRequest,
    PublishedFile,
)


class RequestedFilesError(ValueError):
    pass


def judge_requested_files(
    files: Sequence[PublishedFile], request: PublicationRequest, subject: str
) -> None:
    expected = {published.name: published for published in request.files}
    given = {published.name: published for published in files}
    if len(given) != len(files):
        raise RequestedFilesError(f"{subject} names one file twice")
    for name in sorted(set(given) - set(expected)):
        raise RequestedFilesError(f"{subject} names {name}, which the request does not")
    for name in sorted(set(expected) - set(given)):
        raise RequestedFilesError(f"{subject} lacks {name}, which the request names")
    for name, published in sorted(given.items()):
        if published != expected[name]:
            raise RequestedFilesError(
                f"{subject} gives {published.size_bytes} bytes of SHA-256 {published.sha256} "
                f"for {name}, where the verification accepted {expected[name].size_bytes} bytes "
                f"of SHA-256 {expected[name].sha256}"
            )
