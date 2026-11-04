import contextlib
import io
import os
import re
import termios
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from functools import partial

import pytest

from mostt_model_builder.adapters import terminal_prompt
from mostt_model_builder.adapters.terminal_prompt import (
    TerminalError,
    TerminalPrompt,
    read_line,
)
from mostt_model_builder.domain.publication.hub_token import HubToken, HubTokenError

PLACEHOLDER_TOKEN = b"hf_placeholder0123456789"
NON_ASCII_TOKEN = b"hf_placeholder\xe9123456789"
TYPED_REPOSITORY = b"example-user/parakeet-ultra-onnx"
NON_ASCII_REPOSITORY = b"example-user/parakeet-ultra-onnx\xc3\xa9"
INTERRUPTION = "the maintainer pressed Ctrl-C"
LOCAL_MODES_INDEX = 3


@dataclass
class Terminal:
    controller: int
    descriptor: int
    echo_while_reading: list[bool] = field(default_factory=list)


@dataclass(frozen=True)
class TokenReading:
    hub_token: HubToken
    echo_while_reading: list[bool]
    modes_before: list[object]
    modes_after: list[object]


@dataclass(frozen=True)
class InterruptedReading:
    modes_before: list[object]
    modes_after: list[object]


def interrupt(descriptor: int, max_size_bytes: int) -> bytes:  # noqa: ARG001  # the stand-in is interrupted before it reads
    raise KeyboardInterrupt(INTERRUPTION)


def has_echo(descriptor: int) -> bool:
    return bool(termios.tcgetattr(descriptor)[LOCAL_MODES_INDEX] & termios.ECHO)


class TestTheTerminalPrompt:
    @pytest.fixture
    def terminal(self) -> Iterator[Terminal]:
        controller, descriptor = os.openpty()
        yield Terminal(controller, descriptor)
        os.close(controller)
        os.close(descriptor)

    class TestTheAskRepositoryMethod:
        class TestWhenTheMaintainerTypesTheRepository:
            @pytest.fixture
            def typed_repository(self, terminal: Terminal) -> str:
                os.write(terminal.controller, TYPED_REPOSITORY + b"\n")
                return TerminalPrompt(terminal.descriptor, io.StringIO()).ask_repository()

            def test_should_give_the_line_typed(self, typed_repository: str) -> None:
                assert typed_repository == TYPED_REPOSITORY.decode()

        class TestWhenTheMaintainerTypesAByteOutsideAscii:
            @pytest.fixture
            def typed_repository(self, terminal: Terminal) -> str:
                os.write(terminal.controller, NON_ASCII_REPOSITORY + b"\n")
                return TerminalPrompt(terminal.descriptor, io.StringIO()).ask_repository()

            def test_should_give_an_empty_repository_which_no_confirmation_matches(
                self, typed_repository: str
            ) -> None:
                assert typed_repository == ""

    class TestTheAskTokenMethod:
        class TestWhenTheStandardInputIsAPipe:
            @pytest.fixture
            def ask_act(self) -> Iterator[Callable[[], HubToken]]:
                read_end, write_end = os.pipe()
                yield TerminalPrompt(read_end, io.StringIO()).ask_token
                os.close(read_end)
                os.close(write_end)

            def test_should_refuse_it_since_the_token_is_typed_at_a_terminal(
                self, ask_act: Callable[[], HubToken]
            ) -> None:
                with pytest.raises(TerminalError, match=re.escape("is not a terminal")):
                    ask_act()

        class TestWhenTheMaintainerTypesTheToken:
            @pytest.fixture
            def token_reading(
                self, terminal: Terminal, monkeypatch: pytest.MonkeyPatch
            ) -> TokenReading:
                def read_while_watching_echo(descriptor: int, max_size_bytes: int) -> bytes:  # noqa: ARG001  # the stand-in reads nothing, it only watches the terminal
                    terminal.echo_while_reading.append(has_echo(descriptor))
                    return PLACEHOLDER_TOKEN

                monkeypatch.setattr(terminal_prompt, "read_line", read_while_watching_echo)
                modes_before = termios.tcgetattr(terminal.descriptor)
                hub_token = TerminalPrompt(terminal.descriptor, io.StringIO()).ask_token()
                return TokenReading(
                    hub_token,
                    terminal.echo_while_reading,
                    modes_before,
                    termios.tcgetattr(terminal.descriptor),
                )

            def test_should_read_it_with_echo_off(self, token_reading: TokenReading) -> None:
                assert token_reading.echo_while_reading == [False]

            def test_should_restore_the_terminals_settings(
                self, token_reading: TokenReading
            ) -> None:
                assert token_reading.modes_after == token_reading.modes_before

            def test_should_give_the_token(self, token_reading: TokenReading) -> None:
                assert token_reading.hub_token.encode_secret() == PLACEHOLDER_TOKEN

        class TestWhenTheTokenHoldsAByteOutsideAscii:
            @pytest.fixture
            def ask_act(
                self, terminal: Terminal, monkeypatch: pytest.MonkeyPatch
            ) -> Callable[[], HubToken]:
                def read_non_ascii_token(descriptor: int, max_size_bytes: int) -> bytes:  # noqa: ARG001  # the stand-in gives a typed line without reading the terminal
                    return NON_ASCII_TOKEN

                monkeypatch.setattr(terminal_prompt, "read_line", read_non_ascii_token)
                return TerminalPrompt(terminal.descriptor, io.StringIO()).ask_token

            def test_should_refuse_it_without_quoting_it(
                self, ask_act: Callable[[], HubToken]
            ) -> None:
                with pytest.raises(
                    HubTokenError, match=f"^{re.escape('the token holds a byte outside ASCII')}$"
                ):
                    ask_act()

        class TestWhenTheReadingIsInterrupted:
            @pytest.fixture
            def ask_act(
                self, terminal: Terminal, monkeypatch: pytest.MonkeyPatch
            ) -> Callable[[], HubToken]:
                monkeypatch.setattr(terminal_prompt, "read_line", interrupt)
                return TerminalPrompt(terminal.descriptor, io.StringIO()).ask_token

            def test_should_let_the_interruption_through(
                self, ask_act: Callable[[], HubToken]
            ) -> None:
                with pytest.raises(KeyboardInterrupt, match=INTERRUPTION):
                    ask_act()

        class TestWhenTheReadingIsInterruptedAndTheInterruptionCaught:
            @pytest.fixture
            def interrupted_reading(
                self, terminal: Terminal, monkeypatch: pytest.MonkeyPatch
            ) -> InterruptedReading:
                monkeypatch.setattr(terminal_prompt, "read_line", interrupt)
                modes_before = termios.tcgetattr(terminal.descriptor)
                with contextlib.suppress(KeyboardInterrupt):
                    TerminalPrompt(terminal.descriptor, io.StringIO()).ask_token()
                return InterruptedReading(modes_before, termios.tcgetattr(terminal.descriptor))

            def test_should_restore_the_terminals_settings_on_the_way_out(
                self, interrupted_reading: InterruptedReading
            ) -> None:
                assert interrupted_reading.modes_after == interrupted_reading.modes_before

    class TestTheReadLineFunction:
        class TestWhenALineEnds:
            @pytest.fixture
            def line(self) -> Iterator[bytes]:
                read_end, write_end = os.pipe()
                os.write(write_end, b"example-user/parakeet-ultra-onnx\n")
                yield read_line(read_end, 512)
                os.close(read_end)
                os.close(write_end)

            def test_should_give_it_without_its_line_feed(self, line: bytes) -> None:
                assert line == b"example-user/parakeet-ultra-onnx"

        class TestWhenTheTerminalClosesBeforeTheLineEnds:
            @pytest.fixture
            def read_act(self) -> Iterator[Callable[[], bytes]]:
                read_end, write_end = os.pipe()
                os.write(write_end, b"example-user")
                os.close(write_end)
                yield partial(read_line, read_end, 512)
                os.close(read_end)

            def test_should_refuse_the_unfinished_line(self, read_act: Callable[[], bytes]) -> None:
                with pytest.raises(
                    TerminalError, match=re.escape("the terminal closed before a line ended")
                ):
                    read_act()

        class TestWhenALineExceedsItsBound:
            @pytest.fixture
            def read_act(self) -> Iterator[Callable[[], bytes]]:
                read_end, write_end = os.pipe()
                os.write(write_end, b"a" * 40 + b"\n")
                yield partial(read_line, read_end, 16)
                os.close(read_end)
                os.close(write_end)

            def test_should_refuse_it(self, read_act: Callable[[], bytes]) -> None:
                with pytest.raises(TerminalError, match=re.escape("holds more than 16 bytes")):
                    read_act()
