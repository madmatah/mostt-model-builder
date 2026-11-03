import re
from collections.abc import Callable
from functools import partial

import pytest

from mostt_model_builder.domain.bench.fleurs_index import (
    FleursIndexError,
    FleursUtterance,
    locate_archive_member,
    locate_bench_files,
    parse_fleurs_index,
)

FIRST_LINE = "1829\t7105431834829365765.wav\tLes voyageurs.\tles voyageurs\tl e s\t150720\tFEMALE"
SECOND_LINE = "1829\t8412394508646602569.wav\tLes voyageurs.\tles voyageurs\tl e s\t96000\tMALE"
INDEX = f"{FIRST_LINE}\n{SECOND_LINE}\n"
REFUSED_INDEXES = {
    "no final line feed": (FIRST_LINE, "the index does not end with a line feed"),
    "an empty line": (f"{FIRST_LINE}\n\n", "line 2 of the index holds 1 fields, not 7"),
    "a field too few": (
        "\t".join(FIRST_LINE.split("\t")[:6]) + "\n",
        "line 1 of the index holds 6 fields, not 7",
    ),
    "a sentence id that is not a number": (
        FIRST_LINE.replace("1829", "x", 1) + "\n",
        "the sentence id 'x' is not a number",
    ),
    "a recording that is a path": (
        FIRST_LINE.replace("7105431834829365765.wav", "../7105.wav") + "\n",
        "the recording '../7105.wav' does not match",
    ),
    "a count of samples that is not a number": (
        FIRST_LINE.replace("150720", "-1") + "\n",
        "the count of samples '-1' is not a number",
    ),
    "an empty transcription": (
        FIRST_LINE.replace("\tles voyageurs\t", "\t\t") + "\n",
        "the transcription is empty",
    ),
    "a recording indexed twice": (
        f"{FIRST_LINE}\n{FIRST_LINE}\n",
        "the recording 7105431834829365765.wav is indexed twice",
    ),
}


class TestTheFleursIndex:
    class TestTheParseFleursIndexFunction:
        class TestWhenTheIndexHoldsTwoRecordingsOfOneSentence:
            @pytest.fixture
            def utterances(self) -> tuple[FleursUtterance, ...]:
                return parse_fleurs_index(INDEX)

            def test_should_read_each_recording_with_its_transcriptions_and_samples(
                self, utterances: tuple[FleursUtterance, ...]
            ) -> None:
                assert utterances == (
                    FleursUtterance(
                        "1829",
                        "7105431834829365765.wav",
                        "Les voyageurs.",
                        "les voyageurs",
                        150_720,
                    ),
                    FleursUtterance(
                        "1829", "8412394508646602569.wav", "Les voyageurs.", "les voyageurs", 96_000
                    ),
                )

        @pytest.mark.parametrize(
            ("text", "reason"), list(REFUSED_INDEXES.values()), ids=list(REFUSED_INDEXES.keys())
        )
        class TestWhenTheIndexIsMalformed:
            @pytest.fixture
            def parse_act(self, text: str) -> Callable[[], tuple[FleursUtterance, ...]]:
                return partial(parse_fleurs_index, text)

            def test_should_refuse_it_and_say_why(
                self, parse_act: Callable[[], tuple[FleursUtterance, ...]], reason: str
            ) -> None:
                with pytest.raises(FleursIndexError, match=re.escape(reason)):
                    parse_act()

    class TestTheLocateBenchFilesFunction:
        class TestWhenGivenALanguage:
            @pytest.fixture
            def bench_files(self) -> tuple[str, str]:
                return locate_bench_files("fr_fr")

            def test_should_give_its_index_and_its_audio_in_the_dataset(
                self, bench_files: tuple[str, str]
            ) -> None:
                assert bench_files == ("data/fr_fr/test.tsv", "data/fr_fr/audio/test.tar.gz")

    class TestTheLocateArchiveMemberFunction:
        class TestWhenGivenARecording:
            @pytest.fixture
            def member_name(self) -> str:
                return locate_archive_member("7105.wav")

            def test_should_find_it_in_the_test_directory_of_its_archive(
                self, member_name: str
            ) -> None:
                assert member_name == "test/7105.wav"
