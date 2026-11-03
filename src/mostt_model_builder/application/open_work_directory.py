from dataclasses import dataclass
from typing import Protocol

from mostt_model_builder.domain.containers.work_directory import (
    DirectoryFacts,
    WorkDirectory,
    judge_work_directory,
    refuse_malformed_work_directory_path,
    require_absolute_home,
)

HOME_VARIABLE_ORIGIN = "$HOME"
ACCOUNT_HOME_ORIGIN = "the account's home directory"


class DirectoryInspector(Protocol):
    def inspect_directory(self, path: str) -> DirectoryFacts: ...

    def canonicalize(self, path: str) -> str: ...


@dataclass(frozen=True, slots=True)
class WorkDirectoryRequest:
    path: str
    home_variable: str | None
    account_home: str
    clone_root: str
    user_id: int


def open_work_directory(
    request: WorkDirectoryRequest, inspector: DirectoryInspector
) -> WorkDirectory:
    refuse_malformed_work_directory_path(request.path)
    homes = (
        require_absolute_home(request.home_variable, HOME_VARIABLE_ORIGIN),
        require_absolute_home(request.account_home, ACCOUNT_HOME_ORIGIN),
    )
    facts = inspector.inspect_directory(request.path)
    return judge_work_directory(
        request.path,
        facts,
        request.user_id,
        tuple(inspector.canonicalize(home) for home in homes),
        inspector.canonicalize(request.clone_root),
    )
