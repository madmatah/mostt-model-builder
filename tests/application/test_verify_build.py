import hashlib
import json
import re
from collections.abc import Callable
from dataclasses import dataclass, replace
from functools import partial

import pytest
from accepted_build import (
    ENCODER,
    ENCODER_NAME,
    ENCODER_PARITY,
    ENTRIES_DIRECTORY,
    FALLEN_BACK_PLACEMENT,
    FLEURS_REVISION,
    INDEXES,
    REFUSED_INSPECTION_REPORT,
    REPORT_PATH,
    SAME_TOKENS_EVERYWHERE,
    SHARD_LINES,
    VOCABULARY,
    WORK_ROOT,
    build_files,
    build_request,
)
from run_step_fakes import (
    USER_ID,
    FakeStepFiles,
    build_file_facts,
)

from mostt_model_builder.application.verify_build import (
    BuildRejectedError,
    VerifiedBuild,
    refuse_rejected_build,
    verify_build,
)
from mostt_model_builder.domain.acceptance.verification import (
    CandidateDigest,
    VerificationError,
)
from mostt_model_builder.domain.containers.container_files import (
    ContainerFileError,
    FileFacts,
    FileKind,
)
from mostt_model_builder.domain.containers.container_steps import INSPECT_STEP_NAME
from mostt_model_builder.domain.containers.step_peaks import PEAKS_FILE_NAME
from mostt_model_builder.domain.formats.file_formats import FileFormat
from mostt_model_builder.domain.onnx.graph_rules import GraphRuleError
from mostt_model_builder.domain.onnx.node_placement import (
    NodePlacementError,
)


@dataclass(frozen=True)
class CompletedVerification:
    verified: VerifiedBuild
    files: FakeStepFiles


@dataclass(frozen=True)
class RefusedVerification:
    verify: Callable[[], VerifiedBuild]
    message: str


CANDIDATE_DIRECTORY = WORK_ROOT / ENTRIES_DIRECTORY / "candidate"
NOT_THE_HOSTS_COPY = (
    f"the candidate file {ENCODER_NAME} is not the host's own copy: a regular file of the user's, "
    "in mode 0444, with a single link, of at most 4096 bytes"
)


class TestTheBuildVerification:
    class TestTheVerifyBuildFunction:
        class TestWhenEveryRunAndParityIsInPlace:
            @pytest.fixture
            def verification(self) -> CompletedVerification:
                files = build_files()
                return CompletedVerification(verify_build(build_request(), files), files)

            def test_should_measure_each_run_across_its_shards_on_each_language(
                self, verification: CompletedVerification
            ) -> None:
                assert [
                    (measure.run, measure.language, measure.utterance_count, measure.error_count)
                    for measure in verification.verified.verification.runs
                ] == [
                    ("float32-cpu", "fr_fr", 2, 0),
                    ("float32-cpu", "en_us", 2, 3),
                    ("float32-gpu", "fr_fr", 2, 0),
                    ("float32-gpu", "en_us", 2, 3),
                ]

            def test_should_compare_the_runs_by_their_token_ids(
                self, verification: CompletedVerification
            ) -> None:
                assert [
                    (measure.language, measure.identical_count, measure.compared_count)
                    for measure in verification.verified.verification.comparisons
                ] == [("en_us", 2, 2), ("fr_fr", 1, 2)]

            def test_should_hold_each_parity_of_the_recipe(
                self, verification: CompletedVerification
            ) -> None:
                assert [
                    (measure.report.name, measure.failure)
                    for measure in verification.verified.verification.parities
                ] == [(ENCODER_PARITY.name, None)]

            def test_should_write_its_report_beside_the_outputs_of_the_run(
                self, verification: CompletedVerification
            ) -> None:
                assert verification.verified.report_path == REPORT_PATH

            def test_should_record_the_size_and_sha256_of_each_candidate_file_it_accepts(
                self, verification: CompletedVerification
            ) -> None:
                assert verification.verified.verification.candidate == (
                    CandidateDigest(
                        ENCODER_NAME,
                        FileFormat.ONNX,
                        ("float32",),
                        len(ENCODER),
                        hashlib.sha256(ENCODER).hexdigest(),
                    ),
                    CandidateDigest(
                        "vocab.txt",
                        FileFormat.TEXT,
                        ("float32",),
                        len(VOCABULARY),
                        hashlib.sha256(VOCABULARY).hexdigest(),
                    ),
                )

            def test_should_hash_each_file_through_the_candidate_directory_it_judged(
                self, verification: CompletedVerification
            ) -> None:
                assert verification.files.digested_paths == [
                    CANDIDATE_DIRECTORY / ENCODER_NAME,
                    CANDIDATE_DIRECTORY / "vocab.txt",
                ]

            def test_should_write_the_hashes_into_its_report(
                self, verification: CompletedVerification
            ) -> None:
                report = json.loads(verification.files.contents[WORK_ROOT / REPORT_PATH])
                assert [(digest["name"], digest["sha256"]) for digest in report["candidate"]] == [
                    (ENCODER_NAME, hashlib.sha256(ENCODER).hexdigest()),
                    ("vocab.txt", hashlib.sha256(VOCABULARY).hexdigest()),
                ]

            def test_should_record_in_its_report_that_the_build_is_accepted(
                self, verification: CompletedVerification
            ) -> None:
                report = json.loads(verification.files.contents[WORK_ROOT / REPORT_PATH])
                assert report["is_accepted"] is True

            def test_should_read_the_inspection_the_lines_the_placements_and_the_parity(
                self, verification: CompletedVerification
            ) -> None:
                assert sorted(
                    str(path.relative_to(WORK_ROOT / ENTRIES_DIRECTORY))
                    for path in verification.files.read_paths
                    if path.name != PEAKS_FILE_NAME
                    and not path.is_relative_to(WORK_ROOT / ENTRIES_DIRECTORY / "candidate")
                    and not path.is_relative_to(WORK_ROOT / "sources")
                ) == [
                    "export/encoder-parity.json",
                    "inspect/inspection.json",
                    "transcribe-float32-cpu-en_us-0/lines.jsonl",
                    "transcribe-float32-cpu-en_us-1/lines.jsonl",
                    "transcribe-float32-cpu-fr_fr-0/lines.jsonl",
                    "transcribe-float32-cpu-fr_fr-1/lines.jsonl",
                    "transcribe-float32-gpu-en_us-0/lines.jsonl",
                    "transcribe-float32-gpu-en_us-0/placement.json",
                    "transcribe-float32-gpu-fr_fr-0/lines.jsonl",
                    "transcribe-float32-gpu-fr_fr-0/placement.json",
                ]

        class TestWhenTheInspectionRefusedAGraphOfTheCandidate:
            @pytest.fixture
            def verify_act(self) -> Callable[[], VerifiedBuild]:
                files = build_files(inspection_report=REFUSED_INSPECTION_REPORT)
                return partial(verify_build, build_request(), files)

            def test_should_refuse_the_build_since_its_inspection_refused_the_candidate(
                self, verify_act: Callable[[], VerifiedBuild]
            ) -> None:
                with pytest.raises(
                    GraphRuleError,
                    match=re.escape(
                        f"the graph of {ENCODER_NAME} has 3 nodes that hold metadata_props, which "
                        "may name paths of the container that exported it"
                    ),
                ):
                    verify_act()

        class TestWhenTheCandidateWasNeverInspected:
            @pytest.fixture
            def verify_act(self) -> Callable[[], VerifiedBuild]:
                return partial(verify_build, build_request(), build_files(inspection_report=None))

            def test_should_refuse_the_build_and_name_step_inspect(
                self, verify_act: Callable[[], VerifiedBuild]
            ) -> None:
                with pytest.raises(
                    ContainerFileError,
                    match=re.escape(f"{ENTRIES_DIRECTORY}/{INSPECT_STEP_NAME} is missing"),
                ):
                    verify_act()

        class TestWhenAFileOfTheCandidateIsNotTheHostsCopy:
            @pytest.fixture
            def verify_act(self) -> Callable[[], VerifiedBuild]:
                return partial(verify_build, build_request(), build_files(encoder_copy_mode=0o644))

            def test_should_refuse_it_since_it_may_have_changed_since_its_inspection(
                self, verify_act: Callable[[], VerifiedBuild]
            ) -> None:
                with pytest.raises(
                    ContainerFileError,
                    match=re.escape(
                        f"the candidate file {ENCODER_NAME} is not the host's own copy: a regular "
                        "file of the user's, in mode 0444, with a single link, of at most 4096 "
                        "bytes"
                    ),
                ):
                    verify_act()

        class TestWhenACandidateFileIsSwappedBeforeItIsHashed:
            @pytest.fixture(
                params=[
                    pytest.param(
                        (FileFacts(FileKind.LINK, USER_ID, 0o777, 12, 1), NOT_THE_HOSTS_COPY),
                        id="a link",
                    ),
                    pytest.param(
                        (build_file_facts(len(ENCODER), 0o444, owner_id=0), NOT_THE_HOSTS_COPY),
                        id="a file of another user",
                    ),
                    pytest.param(
                        (build_file_facts(len(ENCODER), 0o644), NOT_THE_HOSTS_COPY),
                        id="a file in another mode than the copies",
                    ),
                    pytest.param(
                        (
                            FileFacts(FileKind.REGULAR, USER_ID, 0o444, len(ENCODER), 2),
                            NOT_THE_HOSTS_COPY,
                        ),
                        id="a file with a second link",
                    ),
                ]
            )
            def refused_verification(self, request: pytest.FixtureRequest) -> RefusedVerification:
                hashing_facts, message = request.param
                files = build_files()
                files.hashing_facts[CANDIDATE_DIRECTORY / ENCODER_NAME] = hashing_facts
                return RefusedVerification(partial(verify_build, build_request(), files), message)

            def test_should_refuse_it_by_the_facts_of_the_descriptor_it_hashes(
                self, refused_verification: RefusedVerification
            ) -> None:
                with pytest.raises(
                    ContainerFileError, match=re.escape(refused_verification.message)
                ):
                    refused_verification.verify()

        class TestWhenAComparisonBreaksItsRule:
            @pytest.fixture
            def refuse_act(self) -> Callable[[], None]:
                verified = verify_build(build_request(SAME_TOKENS_EVERYWHERE), build_files())
                return partial(refuse_rejected_build, verified)

            def test_should_refuse_the_build_and_name_its_report(
                self, refuse_act: Callable[[], None]
            ) -> None:
                with pytest.raises(
                    BuildRejectedError,
                    match=re.escape(
                        f"the build fails 1 of its rules, as {REPORT_PATH} records: float32-gpu "
                        "against float32-cpu, fr_fr: 1 of 2 utterances have identical token ids"
                    ),
                ):
                    refuse_act()

        class TestWhenAnIndexOfTheBenchNoLongerMatchesItsPin:
            @pytest.fixture
            def verify_act(self) -> Callable[[], VerifiedBuild]:
                files = build_files(indexes={**INDEXES, "en_us": INDEXES["en_us"][:-1] + b"X"})
                return partial(verify_build, build_request(), files)

            def test_should_refuse_to_measure_against_it(
                self, verify_act: Callable[[], VerifiedBuild]
            ) -> None:
                with pytest.raises(
                    VerificationError,
                    match=re.escape(
                        f"sources/google/fleurs/{FLEURS_REVISION}/data/en_us/test.tsv no longer "
                        "matches its pin"
                    ),
                ):
                    verify_act()

        class TestWhenAShardOfARunNeverRan:
            @pytest.fixture
            def verify_act(self) -> Callable[[], VerifiedBuild]:
                shard_lines = {
                    name: lines
                    for name, lines in SHARD_LINES.items()
                    if name != "transcribe-float32-cpu-fr_fr-1"
                }
                return partial(verify_build, build_request(), build_files(shard_lines=shard_lines))

            def test_should_refuse_its_missing_outputs(
                self, verify_act: Callable[[], VerifiedBuild]
            ) -> None:
                with pytest.raises(
                    ContainerFileError,
                    match=re.escape("transcribe-float32-cpu-fr_fr-1 is missing"),
                ):
                    verify_act()

        class TestWhenARunOnTheGpuFellBackToTheCpu:
            @pytest.fixture
            def verify_act(self) -> Callable[[], VerifiedBuild]:
                return partial(
                    verify_build, build_request(), build_files(placement=FALLEN_BACK_PLACEMENT)
                )

            def test_should_refuse_its_placement(
                self, verify_act: Callable[[], VerifiedBuild]
            ) -> None:
                with pytest.raises(NodePlacementError, match="MatMul"):
                    verify_act()

        class TestWhenAParityExceedsItsTolerance:
            @pytest.fixture
            def refuse_act(self) -> Callable[[], None]:
                request = replace(
                    build_request(),
                    parities=(replace(ENCODER_PARITY, absolute_difference_tolerance=1e-6),),
                )
                return partial(refuse_rejected_build, verify_build(request, build_files()))

            def test_should_refuse_the_build(self, refuse_act: Callable[[], None]) -> None:
                with pytest.raises(
                    BuildRejectedError,
                    match=re.escape(
                        "the parity encoder-against-reference: the maximum absolute difference, "
                        "4e-06, exceeds the tolerance of 1e-06"
                    ),
                ):
                    refuse_act()
