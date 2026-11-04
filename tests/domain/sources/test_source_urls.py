import pytest

from mostt_model_builder.domain.sources.source_pins import RepositoryType, SourcePin
from mostt_model_builder.domain.sources.source_urls import build_source_url, build_tree_url

REVISION = "70bb2e84b976b7e960aa89f1c648e09c59f894dd"
SHA256 = "0" * 64
DATASET_PIN = SourcePin(
    repository_type=RepositoryType.DATASET,
    repository="google/fleurs",
    revision=REVISION,
    path="data/fr_fr/test.tsv",
    size_bytes=1,
    sha256=SHA256,
)
MODEL_PIN = SourcePin(
    repository_type=RepositoryType.MODEL,
    repository="moondream/parakeet-ultra",
    revision=REVISION,
    path="model.safetensors",
    size_bytes=1,
    sha256=SHA256,
)


class TestTheSourceUrls:
    class TestTheBuildSourceUrlFunction:
        class TestWhenThePinIsADatasetFile:
            @pytest.fixture
            def source_url(self) -> str:
                return build_source_url(DATASET_PIN)

            def test_should_resolve_the_file_at_its_revision_among_the_datasets(
                self, source_url: str
            ) -> None:
                assert source_url == (
                    f"https://huggingface.co/datasets/google/fleurs/resolve/{REVISION}/"
                    "data/fr_fr/test.tsv"
                )

        class TestWhenThePinIsAModelFile:
            @pytest.fixture
            def source_url(self) -> str:
                return build_source_url(MODEL_PIN)

            def test_should_resolve_the_file_at_its_revision_among_the_models(
                self, source_url: str
            ) -> None:
                assert source_url == (
                    f"https://huggingface.co/moondream/parakeet-ultra/resolve/{REVISION}/"
                    "model.safetensors"
                )

    class TestTheBuildTreeUrlFunction:
        class TestWhenGivenAModelRepositoryAndARevision:
            @pytest.fixture
            def tree_url(self) -> str:
                return build_tree_url("example-user/parakeet-ultra-onnx", REVISION)

            def test_should_list_the_files_of_the_revision_on_the_hub(self, tree_url: str) -> None:
                assert tree_url == (
                    "https://huggingface.co/api/models/example-user/parakeet-ultra-onnx/tree/"
                    f"{REVISION}"
                )
