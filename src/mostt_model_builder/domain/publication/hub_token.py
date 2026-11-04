import re
from dataclasses import dataclass

# The grammar keeps a line break, a quote or a byte outside ASCII out of the one line that the
# publication's container reads.
HUB_TOKEN_GRAMMAR = re.compile(r"hf_[A-Za-z0-9]{8,256}")
MAX_HUB_TOKEN_BYTES = 512
TOKEN_ENCODING = "ascii"
HIDDEN_REPRESENTATION = "HubToken(<hidden>)"
LINE_FEED = "\n"


class HubTokenError(ValueError):
    pass


@dataclass(frozen=True, slots=True, repr=False)
class HubToken:
    secret: str

    def __repr__(self) -> str:
        return HIDDEN_REPRESENTATION

    def __str__(self) -> str:
        return HIDDEN_REPRESENTATION

    def encode_line(self) -> bytes:
        return (self.secret + LINE_FEED).encode(TOKEN_ENCODING)

    def encode_secret(self) -> bytes:
        return self.secret.encode(TOKEN_ENCODING)


# A refusal never quotes the text it refused, which may be a token typed with a mistake.
def parse_hub_token(text: str) -> HubToken:
    if not HUB_TOKEN_GRAMMAR.fullmatch(text):
        raise HubTokenError(
            "the token is not hf_ followed by 8 to 256 ASCII letters and digits, as a Hugging "
            "Face token is"
        )
    return HubToken(text)
