import re
from collections.abc import Callable
from functools import partial

import pytest

from mostt_model_builder.domain.lock.lock_freshness import LockFreshnessError, check_lock_freshness

ROOT = "mostt-model-builder-images"
MARKERS = "[\"platform_machine == 'x86_64' and sys_platform == 'linux'\"]"
EXCLUDE_NEWER = "2026-11-02T16:10:00Z"
MANIFEST = f"""
[project]
name = "{ROOT}"
version = "0.1.0"
requires-python = "==3.12.*"
dependencies = []

[project.optional-dependencies]
core-onnx-inspect = ["onnx==1.23.2"]
parakeet-tdt-export = ["nemo-toolkit[asr]==3.0.0", "torch==2.15.0"]

[tool.uv]
package = false
no-build = false
environments = ["sys_platform == 'linux' and platform_machine == 'x86_64'"]
exclude-newer = "{EXCLUDE_NEWER}"
required-version = "==0.13.7"
conflicts = [[{{ extra = "core-onnx-inspect" }}, {{ extra = "parakeet-tdt-export" }}]]

[[tool.uv.dependency-metadata]]
name = "antlr4-python3-runtime"
version = "4.9.3"
requires-dist = []
"""
EXPORT_MARKER = "marker = \"extra == 'parakeet-tdt-export'\""
INSPECT_MARKER = "marker = \"extra == 'core-onnx-inspect'\""
REQUIRES_DIST = [
    f'{{ name = "nemo-toolkit", extras = ["asr"], {EXPORT_MARKER}, specifier = "==3.0.0" }}',
    f'{{ name = "onnx", {INSPECT_MARKER}, specifier = "==1.23.2" }}',
    f'{{ name = "torch", {EXPORT_MARKER}, specifier = "==2.15.0" }}',
]
RECORDED_EXTRAS = ("core-onnx-inspect", "parakeet-tdt-export")
PROVIDES_EXTRAS = "[" + ", ".join(f'"{extra}"' for extra in reversed(RECORDED_EXTRAS)) + "]"
CONFLICTS = (
    f'[[{{ package = "{ROOT}", extra = "core-onnx-inspect" }}, '
    f'{{ package = "{ROOT}", extra = "parakeet-tdt-export" }}]]'
)
DEPENDENCY_METADATA = (
    '[[manifest.dependency-metadata]]\nname = "antlr4-python3-runtime"\nversion = "4.9.3"\n'
)


def build_lock(
    requires_python: str = '"==3.12.*"',
    supported_markers: str = MARKERS,
    conflicts: str = CONFLICTS,
    dependency_metadata: str = DEPENDENCY_METADATA,
    exclude_newer: str = EXCLUDE_NEWER,
    requires_dist: list[str] = REQUIRES_DIST,
    provides_extras: str = PROVIDES_EXTRAS,
    root_lines: str = "",
    metadata_lines: str = "",
) -> str:
    return f"""version = 1
revision = 3
requires-python = {requires_python}
supported-markers = {supported_markers}
conflicts = {conflicts}

[options]
exclude-newer = "{exclude_newer}"

[manifest]

{dependency_metadata}
[[package]]
name = "{ROOT}"
version = "0.1.0"
source = {{ virtual = "." }}
{root_lines}
[package.metadata]
requires-dist = [{", ".join(requires_dist)}]
provides-extras = {provides_extras}
{metadata_lines}
"""


STALE_LOCKS = {
    "a requirement added to the manifest and not to the lock": (
        build_lock(requires_dist=REQUIRES_DIST[:2]),
        "requires-dist: the lock records",
    ),
    "a requirement removed from the manifest": (
        build_lock(
            requires_dist=[
                *REQUIRES_DIST,
                f'{{ name = "numpy", {INSPECT_MARKER}, specifier = "==2.5.4" }}',
            ]
        ),
        "requires-dist: the lock records",
    ),
    "a requirement changed in its pin": (
        build_lock(
            requires_dist=[
                REQUIRES_DIST[0],
                REQUIRES_DIST[1].replace("1.23.2", "1.23.1"),
                REQUIRES_DIST[2],
            ]
        ),
        "requires-dist: the lock records",
    ),
    "a requirement changed in its extras": (
        build_lock(
            requires_dist=[REQUIRES_DIST[0].replace('["asr"]', '["all"]'), *REQUIRES_DIST[1:]]
        ),
        "requires-dist: the lock records",
    ),
    "an extra added": (
        build_lock(provides_extras='["core-onnx-inspect"]'),
        "provides-extras: the lock records ['core-onnx-inspect']",
    ),
    "an extra removed": (
        build_lock(provides_extras='["core-bench", "core-onnx-inspect", "parakeet-tdt-export"]'),
        "provides-extras: the lock records ['core-bench',",
    ),
    "an entry of requires-dist without an extra marker": (
        build_lock(requires_dist=[*REQUIRES_DIST, '{ name = "six", specifier = "==1.0" }']),
        "the root's requires-dist holds {'name': 'six', 'specifier': '==1.0'}, "
        "which no extra declares",
    ),
    "a requires-dev on the root": (
        build_lock(metadata_lines="[package.metadata.requires-dev]\ndev = []"),
        "the root's metadata holds the unknown key 'requires-dev'",
    ),
    "dev dependencies on the root": (
        build_lock(root_lines="dev-dependencies = { dev = [] }"),
        "the root holds the unknown key 'dev-dependencies'",
    ),
    "another requires-python": (
        build_lock(requires_python='">=3.12"'),
        "requires-python: the lock records '>=3.12', the manifest declares '==3.12.*'",
    ),
    "other supported markers": (
        build_lock(supported_markers="[\"sys_platform == 'linux'\"]"),
        "supported-markers: the lock records",
    ),
    "other dependency metadata": (
        build_lock(dependency_metadata=DEPENDENCY_METADATA.replace("4.9.3", "4.13.2")),
        "dependency-metadata: the lock records",
    ),
    "other conflicts": (build_lock(conflicts="[]"), "conflicts: the lock records []"),
    "another exclude-newer": (
        build_lock(exclude_newer="2026-12-01T00:00:00Z"),
        "exclude-newer: the lock records '2026-12-01T00:00:00Z'",
    ),
    "an unknown key in the root's entries": (
        build_lock(root_lines='sdist = { url = "https://files.pythonhosted.org/x.tar.gz" }'),
        "the root holds the unknown key 'sdist'",
    ),
}


class TestTheLockFreshness:
    class TestTheCheckLockFreshnessFunction:
        class TestWhenTheLockRecordsWhatTheManifestDeclares:
            @pytest.fixture
            def fresh_extras(self) -> tuple[str, ...]:
                return check_lock_freshness(MANIFEST, build_lock(), ROOT)

            def test_should_accept_it_and_give_the_extras_the_lock_records_sorted(
                self, fresh_extras: tuple[str, ...]
            ) -> None:
                assert fresh_extras == RECORDED_EXTRAS

        class TestWhenTheLockNestsAnArray3000Deep:
            @pytest.fixture
            def check_act(self) -> Callable[[], object]:
                return partial(check_lock_freshness, MANIFEST, "value = " + "[" * 3000, ROOT)

            def test_should_refuse_it_as_invalid_toml(
                self, check_act: Callable[[], object]
            ) -> None:
                with pytest.raises(
                    LockFreshnessError, match=r"^the manifest or the lock is not valid TOML: "
                ):
                    check_act()

        @pytest.mark.parametrize(
            ("lock_text", "reason"), list(STALE_LOCKS.values()), ids=list(STALE_LOCKS.keys())
        )
        class TestWhenTheLockDriftsFromTheManifest:
            @pytest.fixture
            def check_act(self, lock_text: str) -> Callable[[], object]:
                return partial(check_lock_freshness, MANIFEST, lock_text, ROOT)

            def test_should_refuse_it_and_say_what_differs(
                self, check_act: Callable[[], object], reason: str
            ) -> None:
                with pytest.raises(LockFreshnessError, match=re.escape(reason)):
                    check_act()
