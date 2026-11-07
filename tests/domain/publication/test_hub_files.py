import pytest

from mostt_model_builder.domain.publication.hub_files import (
    COMPARED_CONTENT_MAX_SIZE_BYTES,
    FileDecision,
    HubFile,
    decide_file,
)
from mostt_model_builder.domain.publication.publication_request import PublishedFile

SHA256 = "1" * 64
OTHER_SHA256 = "2" * 64
SMALL_FILE = PublishedFile("vocab.txt", 9412, SHA256)
LARGE_FILE = PublishedFile("encoder-model.onnx.data", COMPARED_CONTENT_MAX_SIZE_BYTES + 1, SHA256)


class TestTheHubFiles:
    class TestTheDecideFileFunction:
        @pytest.mark.parametrize(
            ("requested", "listed", "expected_decision"),
            [
                pytest.param(
                    LARGE_FILE,
                    HubFile(LARGE_FILE.name, LARGE_FILE.size_bytes, SHA256),
                    FileDecision.KEPT,
                    id="the-same-size-and-large-file-storage-sha256-is-kept",
                ),
                pytest.param(
                    LARGE_FILE,
                    HubFile(LARGE_FILE.name, LARGE_FILE.size_bytes, OTHER_SHA256),
                    FileDecision.UPLOAD,
                    id="another-large-file-storage-sha256-is-uploaded",
                ),
                pytest.param(
                    SMALL_FILE,
                    HubFile(SMALL_FILE.name, SMALL_FILE.size_bytes, None),
                    FileDecision.COMPARE_CONTENT,
                    id="no-large-file-storage-sha256-within-16-mib-has-its-content-compared",
                ),
                pytest.param(
                    PublishedFile(SMALL_FILE.name, COMPARED_CONTENT_MAX_SIZE_BYTES, SHA256),
                    HubFile(SMALL_FILE.name, COMPARED_CONTENT_MAX_SIZE_BYTES, None),
                    FileDecision.COMPARE_CONTENT,
                    id="exactly-16-mib-has-its-content-compared",
                ),
                pytest.param(
                    LARGE_FILE,
                    HubFile(LARGE_FILE.name, LARGE_FILE.size_bytes, None),
                    FileDecision.UPLOAD,
                    id="no-large-file-storage-sha256-beyond-16-mib-is-uploaded-unread",
                ),
                pytest.param(
                    SMALL_FILE,
                    None,
                    FileDecision.UPLOAD,
                    id="a-file-missing-on-the-hub-is-uploaded",
                ),
                pytest.param(
                    SMALL_FILE,
                    HubFile(SMALL_FILE.name, SMALL_FILE.size_bytes + 1, SHA256),
                    FileDecision.UPLOAD,
                    id="another-size-is-uploaded",
                ),
            ],
        )
        class TestWhenTheHubListsAFile:
            @pytest.fixture
            def decision(self, requested: PublishedFile, listed: HubFile | None) -> FileDecision:
                return decide_file(requested, listed)

            def test_should_decide_whether_it_goes_up(
                self, decision: FileDecision, expected_decision: FileDecision
            ) -> None:
                assert decision is expected_decision
