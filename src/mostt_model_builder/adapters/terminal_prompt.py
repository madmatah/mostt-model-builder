import os
import termios
from collections.abc import Sequence
from typing import TextIO

from mostt_model_builder.domain.publication.hub_token import (
    MAX_HUB_TOKEN_BYTES,
    TOKEN_ENCODING,
    HubToken,
    HubTokenError,
    parse_hub_token,
)

LINE_FEED = b"\n"
MAX_REPOSITORY_LINE_BYTES = 512
READ_CHUNK_BYTES = 1024
LOCAL_MODES_INDEX = 3
REPOSITORY_QUESTION = "type the repository again to confirm it: "
TOKEN_QUESTION = (
    "the Hugging Face token, fine-grained, with read and write access to this repository alone "
    "(it is not shown): "
)


class TerminalError(Exception):
    pass


def read_line(descriptor: int, max_size_bytes: int) -> bytes:
    line = b""
    while not line.endswith(LINE_FEED):
        chunk = os.read(descriptor, READ_CHUNK_BYTES)
        if not chunk:
            raise TerminalError("the terminal closed before a line ended")
        line += chunk
        if len(line) > max_size_bytes + len(LINE_FEED):
            raise TerminalError(f"the line holds more than {max_size_bytes} bytes")
    return line.removesuffix(LINE_FEED)


class TerminalPrompt:
    def __init__(self, descriptor: int, output: TextIO) -> None:
        self.descriptor = descriptor
        self.output = output

    def require_terminal(self) -> None:
        if not os.isatty(self.descriptor):
            raise TerminalError("the standard input is not a terminal, where the token is typed")

    def announce(self, lines: Sequence[str]) -> None:
        for line in lines:
            print(line, file=self.output, flush=True)

    def ask_repository(self) -> str:
        self.require_terminal()
        print(REPOSITORY_QUESTION, end="", file=self.output, flush=True)
        line = read_line(self.descriptor, MAX_REPOSITORY_LINE_BYTES)
        try:
            return line.decode(TOKEN_ENCODING)
        except UnicodeDecodeError:
            return ""

    # Echo goes off as getpass turns it off, and comes back in a finally, an interruption
    # included; TCSAFLUSH drops whatever was typed ahead, on the way in and out.
    def ask_token(self) -> HubToken:
        self.require_terminal()
        print(TOKEN_QUESTION, end="", file=self.output, flush=True)
        saved_modes = termios.tcgetattr(self.descriptor)
        silent_modes = list(saved_modes)
        silent_modes[LOCAL_MODES_INDEX] = saved_modes[LOCAL_MODES_INDEX] & ~termios.ECHO
        termios.tcsetattr(self.descriptor, termios.TCSAFLUSH, silent_modes)
        try:
            line = read_line(self.descriptor, MAX_HUB_TOKEN_BYTES)
        finally:
            termios.tcsetattr(self.descriptor, termios.TCSAFLUSH, saved_modes)
            print(file=self.output, flush=True)
        try:
            text = line.decode(TOKEN_ENCODING)
        except UnicodeDecodeError:
            raise HubTokenError("the token holds a byte outside ASCII") from None
        return parse_hub_token(text)
