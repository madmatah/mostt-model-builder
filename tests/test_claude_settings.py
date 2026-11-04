import fnmatch
import json
from pathlib import Path

import pytest

SETTINGS_PATH = Path(__file__).resolve().parent.parent / ".claude" / "settings.json"
BASH_RULE_PREFIX = "Bash("
BASH_RULE_SUFFIX = ")"
DENIED_RULES = [
    "Bash(*model-build.sh * publish)",
    "Bash(*model-build.sh * publish *)",
    "Bash(*mostt_model_builder.host_main * publish)",
    "Bash(*mostt_model_builder.host_main * publish *)",
]
PUBLISHING_COMMANDS = {
    "the launcher with its options": (
        "./model-build.sh --work /data/model-build --recipe parakeet-tdt publish "
        "--repository example-user/parakeet-ultra-onnx"
    ),
    "the launcher bare": "./model-build.sh --work /data/model-build --recipe parakeet-tdt publish",
    "the launcher after a change of directory": (
        "cd /data/clone && ./model-build.sh --work /data/w --recipe parakeet-tdt publish"
    ),
    "the module of the command line run through the bootstrap": (
        "python3 -I -S -B src/mostt_model_builder/bootstrap.py "
        "mostt_model_builder.host_main --work /data/w --recipe parakeet-tdt publish "
        "--repository example-user/parakeet-ultra-onnx"
    ),
}
COMMANDS_THAT_PUBLISH_NOTHING = {
    "the check of a published revision": (
        "./model-build.sh --work /data/w --recipe parakeet-tdt check-published "
        "--repository example-user/parakeet-ultra-onnx --revision "
        "0123456789abcdef0123456789abcdef01234567"
    ),
    "the card for review": "./model-build.sh --work /data/w --recipe parakeet-tdt card",
    "the build": "./model-build.sh --work /data/w --recipe parakeet-tdt build",
}


def read_denied_rules() -> list[str]:
    settings = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
    return list(settings["permissions"]["deny"])


# The rules' `*` matches any text, spaces included, as fnmatch's does.
def is_denied(command: str, rules: list[str]) -> bool:
    return any(
        fnmatch.fnmatchcase(
            command, rule.removeprefix(BASH_RULE_PREFIX).removesuffix(BASH_RULE_SUFFIX)
        )
        for rule in rules
    )


class TestTheClaudeSettings:
    class TestWhenTheDeniedCommandsAreRead:
        @pytest.fixture(scope="class")
        @classmethod
        def denied_rules(cls) -> list[str]:
            return read_denied_rules()

        def test_should_deny_agents_the_publish_command_through_either_entry_point(
            self, denied_rules: list[str]
        ) -> None:
            assert denied_rules == DENIED_RULES

        @pytest.mark.parametrize(
            "command", list(PUBLISHING_COMMANDS.values()), ids=list(PUBLISHING_COMMANDS.keys())
        )
        class TestWhenACommandPublishes:
            def test_should_deny_it(self, denied_rules: list[str], command: str) -> None:
                assert is_denied(command, denied_rules)

        @pytest.mark.parametrize(
            "command",
            list(COMMANDS_THAT_PUBLISH_NOTHING.values()),
            ids=list(COMMANDS_THAT_PUBLISH_NOTHING.keys()),
        )
        class TestWhenACommandPublishesNothing:
            def test_should_let_it_run_since_publish_needs_a_space_before_it(
                self, denied_rules: list[str], command: str
            ) -> None:
                assert not is_denied(command, denied_rules)
