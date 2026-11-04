from mostt_model_builder.domain.sources.source_pins import RepositoryType, SourcePin

HUGGING_FACE_URL = "https://huggingface.co"
REPOSITORY_PREFIXES = {RepositoryType.MODEL: "", RepositoryType.DATASET: "/datasets"}


def build_source_url(pin: SourcePin) -> str:
    repository_prefix = REPOSITORY_PREFIXES[pin.repository_type]
    return (
        f"{HUGGING_FACE_URL}{repository_prefix}/{pin.repository}/resolve/{pin.revision}/{pin.path}"
    )


def build_tree_url(repository: str, revision: str) -> str:
    return f"{HUGGING_FACE_URL}/api/models/{repository}/tree/{revision}"
