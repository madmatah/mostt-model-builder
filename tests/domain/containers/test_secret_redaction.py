import itertools
import random

import pytest

from mostt_model_builder.domain.containers.secret_redaction import (
    REDACTED_SECRET,
    SecretRedaction,
)

PROPERTY_SEED = 20261104
PROPERTY_CASE_COUNT = 300
# The third secret ends as it begins, so two of its occurrences can overlap.
SECRETS = (b"hf_placeholder0123", b"hf_abcdefgh", b"hf_abcdefgh" + b"f_abcdefgh")
FILLER_BYTES = b"ahf_\n\x00\xff "
MAX_FILLER_LENGTH = 12
MAX_PIECE_COUNT = 8


def generate_stream(generator: random.Random, secret: bytes) -> bytes:
    pieces = []
    for _ in range(generator.randint(0, MAX_PIECE_COUNT)):
        kind = generator.choice(("filler", "secret", "prefix", "overlap"))
        if kind == "filler":
            pieces.append(
                bytes(
                    generator.choice(FILLER_BYTES)
                    for _ in range(generator.randint(0, MAX_FILLER_LENGTH))
                )
            )
        elif kind == "secret":
            pieces.append(secret)
        elif kind == "prefix":
            pieces.append(secret[: generator.randint(1, len(secret) - 1)])
        else:
            pieces.append(secret + secret[generator.randint(1, len(secret) - 1) :])
    return b"".join(pieces)


def redact_in_chunks(secret: bytes, stream: bytes, cuts: list[int]) -> bytes:
    redaction = SecretRedaction(secret)
    bounds = [0, *sorted(cuts), len(stream)]
    redacted = b"".join(
        redaction.redact(stream[start:end]) for start, end in itertools.pairwise(bounds)
    )
    return redacted + redaction.finish()


def find_mismatches(generator: random.Random) -> list[tuple[bytes, bytes, list[int]]]:
    mismatches = []
    for _ in range(PROPERTY_CASE_COUNT):
        secret = generator.choice(SECRETS)
        stream = generate_stream(generator, secret)
        for cut in range(len(stream) + 1):
            for cuts in ([cut], sorted({cut, generator.randint(0, len(stream))})):
                redacted = redact_in_chunks(secret, stream, cuts)
                if redacted != stream.replace(secret, REDACTED_SECRET) or secret in redacted:
                    mismatches.append((secret, stream, cuts))
    return mismatches


class TestTheSecretRedaction:
    class TestTheRedactMethod:
        class TestWhenGivenAnyStreamCutAtAnyBoundary:
            def test_should_replace_every_occurrence_and_never_let_the_secret_through(
                self,
            ) -> None:
                mismatches = find_mismatches(random.Random(PROPERTY_SEED))
                assert mismatches == [], f"seed {PROPERTY_SEED}"

        class TestWhenTheSecretStraddlesTwoChunks:
            @pytest.fixture
            def redacted(self) -> bytes:
                redaction = SecretRedaction(SECRETS[0])
                return (
                    redaction.redact(b"token: " + SECRETS[0][:5])
                    + redaction.redact(SECRETS[0][5:] + b"\n")
                    + redaction.finish()
                )

            def test_should_still_replace_it(self, redacted: bytes) -> None:
                assert redacted == b"token: " + REDACTED_SECRET + b"\n"
