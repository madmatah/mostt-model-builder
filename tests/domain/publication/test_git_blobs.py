import pytest

from mostt_model_builder.domain.publication.git_blobs import compute_git_blob_id


class TestTheGitBlobs:
    class TestTheComputeGitBlobIdFunction:
        @pytest.mark.parametrize(
            ("content", "expected_blob_id"),
            [
                pytest.param(b"", "e69de29bb2d1d6434b8b29ae775ad8c2e48c5391", id="an-empty-file"),
                pytest.param(
                    b"hello world\n",
                    "3b18e512dba79e4c8300dd08aeb37f8e728b8dad",
                    id="a-short-text",
                ),
            ],
        )
        class TestWhenGivenBytesThatGitHashObjectNames:
            @pytest.fixture
            def blob_id(self, content: bytes) -> str:
                return compute_git_blob_id(content)

            def test_should_give_the_blob_id_that_git_gives(
                self, blob_id: str, expected_blob_id: str
            ) -> None:
                assert blob_id == expected_blob_id
