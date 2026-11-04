import random
import re
import string
from collections.abc import Callable
from functools import partial

import pytest

from mostt_model_builder.domain.publication.hub_token import (
    HubToken,
    HubTokenError,
    parse_hub_token,
)

# Placeholders hold fewer than 30 characters after hf_, so that no secret scan takes them for a
# token.
PLACEHOLDER_TOKEN = "hf_placeholder0123456789"
PROPERTY_SEED = 20261104
PROPERTY_CASE_COUNT = 500
GENERATED_CHARACTERS = string.ascii_letters + string.digits + "_-:\"' \n\té\N{LINE SEPARATOR}"
MIN_GENERATED_LENGTH = 8
MAX_GENERATED_LENGTH = 40
ACCEPTED_TOKENS = {
    "the shortest token": "hf_abcdefgh",
    "letters and digits": PLACEHOLDER_TOKEN,
    "the longest token": "hf_" + "a" * 256,
}
REFUSED_TOKENS = {
    "an empty text": "",
    "no prefix": "placeholder0123456789",
    "another prefix": "hg_placeholder0123456789",
    "a prefix in upper case": "HF_placeholder0123456789",
    "too short a body": "hf_abcdefg",
    "too long a body": "hf_" + "a" * 257,
    "an underscore in the body": "hf_place_holder0123",
    "a trailing line feed": PLACEHOLDER_TOKEN + "\n",
    "a leading space": " " + PLACEHOLDER_TOKEN,
    "a letter outside ASCII": "hf_placéholder0123",
    "a quote": 'hf_place"holder0123',
}


def generate_refused_texts(generator: random.Random) -> list[str]:
    texts: list[str] = []
    while len(texts) < PROPERTY_CASE_COUNT:
        text = "".join(
            generator.choice(GENERATED_CHARACTERS)
            for _ in range(generator.randint(MIN_GENERATED_LENGTH, MAX_GENERATED_LENGTH))
        )
        if not re.fullmatch(r"hf_[A-Za-z0-9]{8,256}", text):
            texts.append(text)
    return texts


def find_quoting_refusals(texts: list[str]) -> list[str]:
    quoting = []
    for text in texts:
        try:
            parse_hub_token(text)
        except HubTokenError as error:
            if text in str(error) or ascii(text) in str(error):
                quoting.append(text)
    return quoting


class TestTheHubToken:
    class TestTheParseHubTokenFunction:
        @pytest.mark.parametrize(
            "text", list(ACCEPTED_TOKENS.values()), ids=list(ACCEPTED_TOKENS.keys())
        )
        class TestWhenTheTextIsAToken:
            @pytest.fixture
            def hub_token(self, text: str) -> HubToken:
                return parse_hub_token(text)

            def test_should_keep_it_whole(self, hub_token: HubToken, text: str) -> None:
                assert hub_token.secret == text

        @pytest.mark.parametrize(
            "text", list(REFUSED_TOKENS.values()), ids=list(REFUSED_TOKENS.keys())
        )
        class TestWhenTheTextIsNoToken:
            @pytest.fixture
            def parse_act(self, text: str) -> Callable[[], HubToken]:
                return partial(parse_hub_token, text)

            def test_should_refuse_it_and_say_what_a_token_is(
                self, parse_act: Callable[[], HubToken]
            ) -> None:
                with pytest.raises(
                    HubTokenError,
                    match=re.escape(
                        "the token is not hf_ followed by 8 to 256 ASCII letters and digits"
                    ),
                ):
                    parse_act()

        class TestWhenGivenAnyTextThatIsNoToken:
            def test_should_never_quote_it_in_its_refusal(self) -> None:
                texts = generate_refused_texts(random.Random(PROPERTY_SEED))
                assert (find_quoting_refusals(texts), len(texts)) == (
                    [],
                    PROPERTY_CASE_COUNT,
                ), f"seed {PROPERTY_SEED}"

    class TestTheHubTokenType:
        class TestWhenAParsedTokenIsShown:
            @pytest.fixture
            def shown_forms(self) -> tuple[str, str, str]:
                hub_token = parse_hub_token(PLACEHOLDER_TOKEN)
                return repr(hub_token), str(hub_token), f"{[hub_token]}"

            def test_should_hide_the_secret_in_every_form(
                self, shown_forms: tuple[str, str, str]
            ) -> None:
                assert shown_forms == (
                    "HubToken(<hidden>)",
                    "HubToken(<hidden>)",
                    "[HubToken(<hidden>)]",
                )

        class TestWhenItIsGivenToAContainer:
            @pytest.fixture
            def line(self) -> bytes:
                return parse_hub_token(PLACEHOLDER_TOKEN).encode_line()

            def test_should_be_one_line_of_ascii(self, line: bytes) -> None:
                assert line == PLACEHOLDER_TOKEN.encode("ascii") + b"\n"
