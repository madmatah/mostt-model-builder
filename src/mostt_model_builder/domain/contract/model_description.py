from dataclasses import dataclass, field
from pathlib import PurePosixPath


@dataclass(frozen=True, slots=True)
class VariantDescription:
    variant: str
    name: str
    role_line: str
    note: str | None = field(default=None, kw_only=True)
    abbreviation: str | None = field(default=None, kw_only=True)


@dataclass(frozen=True, slots=True)
class ReferenceRunDescription:
    run: str
    title: str


@dataclass(frozen=True, slots=True)
class PinnedRevision:
    placeholder: str
    repository: str


@dataclass(frozen=True, slots=True)
class ModelDescription:
    title: str
    template_path: PurePosixPath
    measuring_library_names: tuple[str, ...]
    variants: tuple[VariantDescription, ...]
    reference_runs: tuple[ReferenceRunDescription, ...] = ()
    pinned_revisions: tuple[PinnedRevision, ...] = ()
