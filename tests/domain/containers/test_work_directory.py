import re
from collections.abc import Callable, Mapping
from functools import partial
from pathlib import PurePosixPath

import pytest

from mostt_model_builder.domain.containers.container_steps import (
    SourceReference,
    StepPlacement,
)
from mostt_model_builder.domain.containers.work_directory import (
    DirectoryFacts,
    EntryKind,
    WorkDirectory,
    WorkDirectoryError,
    find_next_lock_number,
    judge_work_directory,
    locate_image_build_log,
    locate_referenced_sources,
    locate_source,
    place_clean,
    place_lock,
    place_step,
    require_absolute_home,
)
from mostt_model_builder.domain.sources.source_pins import RepositoryType, SourcePin

USER_ID = 1000
COMMIT = "e949fa3a2eb8bf218e96c2c03625bbd36e4342b1"
OTHER_USER_ID = 1001
WORK_PATH = "/data/builds/work"
HOME = "/home/user"
ACCOUNT_HOME = "/srv/accounts/user"
CLONE_ROOT = "/opt/clones/mostt-model-builder"
REVISION = "70bb2e84b976b7e960aa89f1c648e09c59f894dd"
FLEURS_INDEX_PIN = SourcePin(
    repository_type=RepositoryType.DATASET,
    repository="google/fleurs",
    revision=REVISION,
    path="data/fr_fr/test.tsv",
    size_bytes=1,
    sha256="0" * 64,
)


def build_facts(
    canonical_path: str,
    kind: EntryKind = EntryKind.DIRECTORY,
    owner_id: int = USER_ID,
    permission_bits: int = 0o700,
) -> DirectoryFacts:
    return DirectoryFacts(kind, canonical_path, owner_id, permission_bits)


GOOD_FACTS = build_facts(WORK_PATH)


def match_exactly(message: str) -> str:
    return f"^{re.escape(message)}$"


MALFORMED_PATHS = {
    "a relative path": "builds/work",
    "the root directory": "/",
    "a tilde": "~/work",
    "a space": "/data/build work",
    "a line feed": "/data/work\n",
    "a colon": "/data/work:ro",
    "a character beyond ascii": "/data/étude",
}
REFUSED_DIRECTORIES = {
    "a link": (
        WORK_PATH,
        build_facts(WORK_PATH, kind=EntryKind.LINK),
        f"the work directory {WORK_PATH} is a link",
    ),
    "a missing directory": (
        WORK_PATH,
        build_facts(WORK_PATH, kind=EntryKind.MISSING),
        f"the work directory {WORK_PATH} does not exist",
    ),
    "a regular file": (
        WORK_PATH,
        build_facts(WORK_PATH, kind=EntryKind.OTHER),
        f"the work directory {WORK_PATH} is not a directory",
    ),
    "a path through a linked parent": (
        WORK_PATH,
        build_facts("/mnt/disk/builds/work"),
        f"the work directory {WORK_PATH} is not canonical: it resolves to '/mnt/disk/builds/work'",
    ),
    "a path with a trailing slash": (
        f"{WORK_PATH}/",
        build_facts(WORK_PATH),
        f"the work directory {WORK_PATH}/ is not canonical: it resolves to '{WORK_PATH}'",
    ),
    "a path with a dot segment": (
        "/data/./builds/work",
        build_facts(WORK_PATH),
        f"the work directory /data/./builds/work is not canonical: it resolves to '{WORK_PATH}'",
    ),
    "another owner": (
        WORK_PATH,
        build_facts(WORK_PATH, owner_id=OTHER_USER_ID),
        f"the work directory {WORK_PATH} is owned by the user id {OTHER_USER_ID}, not by {USER_ID}",
    ),
    **{
        f"the mode {mode:04o}": (
            WORK_PATH,
            build_facts(WORK_PATH, permission_bits=mode),
            f"the work directory {WORK_PATH} has the mode {mode:04o}, not 0700",
        )
        for mode in (0o755, 0o750, 0o770, 0o600, 0o2700, 0o1700)
    },
    **{
        f"{home_name} itself": (
            home,
            build_facts(home),
            f"the work directory {home} is the home directory",
        )
        for home_name, home in (("$HOME", HOME), ("the account's home", ACCOUNT_HOME))
    },
    **{
        f"a directory under {home_name}": (
            f"{home}/builds",
            build_facts(f"{home}/builds"),
            f"the work directory {home}/builds lies under the home directory '{home}'",
        )
        for home_name, home in (("$HOME", HOME), ("the account's home", ACCOUNT_HOME))
    },
    **{
        f"a directory around {home_name}": (
            around,
            build_facts(around),
            f"the work directory {around} holds the home directory '{home}'",
        )
        for home_name, home, around in (
            ("$HOME", HOME, "/home"),
            ("the account's home", ACCOUNT_HOME, "/srv"),
        )
    },
    "the clone": (
        CLONE_ROOT,
        build_facts(CLONE_ROOT),
        f"the work directory {CLONE_ROOT} lies in the clone '{CLONE_ROOT}'",
    ),
    "a directory in the clone": (
        f"{CLONE_ROOT}/work",
        build_facts(f"{CLONE_ROOT}/work"),
        f"the work directory {CLONE_ROOT}/work lies in the clone '{CLONE_ROOT}'",
    ),
    "a directory around the clone": (
        "/opt/clones",
        build_facts("/opt/clones"),
        f"the work directory /opt/clones holds the clone '{CLONE_ROOT}'",
    ),
}
ACCEPTED_DIRECTORIES = {
    "a directory beside $HOME": "/home/other",
    "a directory whose name starts like the home's": "/home/user-builds",
    "a directory beside the clone": "/opt/clones/work",
}


def judge(path: str, facts: DirectoryFacts) -> WorkDirectory:
    return judge_work_directory(path, facts, USER_ID, (HOME, ACCOUNT_HOME), CLONE_ROOT)


class TestTheWorkDirectory:
    class TestTheJudgeWorkDirectoryFunction:
        class TestWhenTheDirectoryIsTheUsersOwnInMode0700OutsideTheHomeAndTheClone:
            @pytest.fixture
            def work_directory(self) -> WorkDirectory:
                return judge(WORK_PATH, GOOD_FACTS)

            def test_should_accept_it(self, work_directory: WorkDirectory) -> None:
                assert work_directory == WorkDirectory(WORK_PATH)

        @pytest.mark.parametrize(
            "path", list(MALFORMED_PATHS.values()), ids=list(MALFORMED_PATHS.keys())
        )
        class TestWhenThePathFallsOutsideTheGrammar:
            @pytest.fixture
            def judge_act(self, path: str) -> Callable[[], WorkDirectory]:
                return partial(judge, path, build_facts(path))

            def test_should_refuse_it_before_anything_else(
                self, judge_act: Callable[[], WorkDirectory], path: str
            ) -> None:
                with pytest.raises(
                    WorkDirectoryError,
                    match=match_exactly(
                        f"the work directory {path!a} does not match /[A-Za-z0-9._/-]+"
                    ),
                ):
                    judge_act()

        @pytest.mark.parametrize(
            ("path", "facts", "message"),
            list(REFUSED_DIRECTORIES.values()),
            ids=list(REFUSED_DIRECTORIES.keys()),
        )
        class TestWhenTheDirectoryBreaksARule:
            @pytest.fixture
            def judge_act(self, path: str, facts: DirectoryFacts) -> Callable[[], WorkDirectory]:
                return partial(judge, path, facts)

            def test_should_refuse_it_and_say_why(
                self, judge_act: Callable[[], WorkDirectory], message: str
            ) -> None:
                with pytest.raises(WorkDirectoryError, match=match_exactly(message)):
                    judge_act()

        @pytest.mark.parametrize(
            "path", list(ACCEPTED_DIRECTORIES.values()), ids=list(ACCEPTED_DIRECTORIES.keys())
        )
        class TestWhenTheDirectoryOnlyLiesNearTheHomeOrTheClone:
            @pytest.fixture
            def work_directory(self, path: str) -> WorkDirectory:
                return judge(path, build_facts(path))

            def test_should_accept_it(self, work_directory: WorkDirectory, path: str) -> None:
                assert work_directory == WorkDirectory(path)

    class TestTheRequireAbsoluteHomeFunction:
        @pytest.mark.parametrize(
            ("home", "message"),
            [
                pytest.param(
                    None,
                    "$HOME is missing, so the home directory cannot be avoided",
                    id="a missing home",
                ),
                pytest.param(
                    "", "$HOME is missing, so the home directory cannot be avoided", id="empty"
                ),
                pytest.param(
                    "home/user",
                    "$HOME is 'home/user', which is not an absolute path",
                    id="relative",
                ),
            ],
        )
        class TestWhenTheHomeCannotBeKnown:
            @pytest.fixture
            def require_act(self, home: str | None) -> Callable[[], str]:
                return partial(require_absolute_home, home, "$HOME")

            def test_should_refuse_it_since_the_work_directory_could_not_avoid_it(
                self, require_act: Callable[[], str], message: str
            ) -> None:
                with pytest.raises(WorkDirectoryError, match=match_exactly(message)):
                    require_act()

        class TestWhenTheHomeIsAbsolute:
            @pytest.fixture
            def home(self) -> str:
                return require_absolute_home(HOME, "$HOME")

            def test_should_give_it_back(self, home: str) -> None:
                assert home == HOME

    class TestTheLocateSourceFunction:
        class TestWhenGivenAPin:
            @pytest.fixture
            def source_path(self) -> PurePosixPath:
                return locate_source(FLEURS_INDEX_PIN)

            def test_should_place_it_under_the_sources_by_namespace_name_and_revision(
                self, source_path: PurePosixPath
            ) -> None:
                assert source_path == PurePosixPath(
                    f"sources/google/fleurs/{REVISION}/data/fr_fr/test.tsv"
                )

    class TestTheLocateReferencedSourcesFunction:
        class TestWhenAPinGivesTheReferencedFile:
            @pytest.fixture
            def source_paths(self) -> Mapping[SourceReference, PurePosixPath]:
                return locate_referenced_sources(
                    [SourceReference("google/fleurs", "data/fr_fr/test.tsv")], [FLEURS_INDEX_PIN]
                )

            def test_should_place_it_where_the_fetch_put_it(
                self, source_paths: Mapping[SourceReference, PurePosixPath]
            ) -> None:
                assert source_paths == {
                    SourceReference("google/fleurs", "data/fr_fr/test.tsv"): PurePosixPath(
                        f"sources/google/fleurs/{REVISION}/data/fr_fr/test.tsv"
                    )
                }

        class TestWhenNoPinGivesTheReferencedFile:
            @pytest.fixture
            def locate_act(self) -> Callable[[], Mapping[SourceReference, PurePosixPath]]:
                return partial(
                    locate_referenced_sources,
                    [SourceReference("google/fleurs", "data/en_us/test.tsv")],
                    [FLEURS_INDEX_PIN],
                )

            def test_should_refuse_the_step(
                self, locate_act: Callable[[], Mapping[SourceReference, PurePosixPath]]
            ) -> None:
                with pytest.raises(
                    WorkDirectoryError,
                    match=re.escape("no pin gives data/en_us/test.tsv of google/fleurs"),
                ):
                    locate_act()

    class TestThePlaceStepFunction:
        class TestWhenACoreStepRunsAtACommit:
            @pytest.fixture
            def placement(self) -> StepPlacement:
                return place_step("core", COMMIT, "fleurs-check")

            def test_should_keep_its_outputs_request_and_log_apart_under_the_commit(
                self, placement: StepPlacement
            ) -> None:
                assert placement == StepPlacement(
                    output_directory=PurePosixPath(f"runs/core/{COMMIT}/fleurs-check"),
                    request_directory=PurePosixPath(f"runs/core/{COMMIT}/requests/fleurs-check"),
                    log_path=PurePosixPath(f"runs/core/{COMMIT}/logs/fleurs-check.log"),
                )

    class TestThePlaceCleanFunction:
        class TestWhenAStepIsCleaned:
            @pytest.fixture
            def placement(self) -> StepPlacement:
                return place_clean("core", COMMIT, "fleurs-check")

            def test_should_mount_the_steps_outputs_alone_and_log_apart(
                self, placement: StepPlacement
            ) -> None:
                assert placement == StepPlacement(
                    output_directory=PurePosixPath(f"runs/core/{COMMIT}/fleurs-check"),
                    request_directory=None,
                    log_path=PurePosixPath(f"runs/core/{COMMIT}/logs/clean-fleurs-check.log"),
                )

    class TestThePlaceLockFunction:
        class TestWhenALockRuns:
            @pytest.fixture
            def placement(self) -> StepPlacement:
                return place_lock(3)

            def test_should_give_it_a_request_an_output_and_a_log_of_its_own(
                self, placement: StepPlacement
            ) -> None:
                assert placement == StepPlacement(
                    output_directory=PurePosixPath("lock/3/output"),
                    request_directory=PurePosixPath("lock/3/request"),
                    log_path=PurePosixPath("lock/3/logs/lock.log"),
                )

    class TestTheFindNextLockNumberFunction:
        @pytest.mark.parametrize(
            ("entry_names", "lock_number"),
            [
                pytest.param([], 1, id="no lock yet"),
                pytest.param(["1", "2", "10"], 11, id="numbered locks"),
                pytest.param(["2", "notes", "03", "-1"], 3, id="entries that are not numbers"),
            ],
        )
        class TestWhenEarlierLocksExist:
            @pytest.fixture
            def next_number(self, entry_names: list[str]) -> int:
                return find_next_lock_number(entry_names)

            def test_should_number_the_new_one_after_the_last(
                self, next_number: int, lock_number: int
            ) -> None:
                assert next_number == lock_number

    class TestTheLocateImageBuildLogFunction:
        class TestWhenAnImageIsBuilt:
            @pytest.fixture
            def log_path(self) -> PurePosixPath:
                return locate_image_build_log("mostt-model-builder-core-bench")

            def test_should_log_it_under_the_image_builds(self, log_path: PurePosixPath) -> None:
                assert log_path == PurePosixPath("image-builds/mostt-model-builder-core-bench.log")
