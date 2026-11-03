import re
from collections.abc import Callable
from functools import partial

import pytest

from mostt_model_builder.domain.lock.declared_requirements import (
    DeclaredRequirement,
    DeclaredRequirementError,
    parse_declared_requirement,
    read_declared_requirements,
)

EXTRA = "parakeet-tdt-export"
ACCEPTED_REQUIREMENTS = {
    "a name and a version": ("torch==2.15.0", DeclaredRequirement(EXTRA, "torch", (), "2.15.0")),
    "an extra": (
        "nemo-toolkit[asr]==3.0.0",
        DeclaredRequirement(EXTRA, "nemo-toolkit", ("asr",), "3.0.0"),
    ),
    "two extras": (
        "onnxruntime-gpu[cudnn,cuda]==1.31.0",
        DeclaredRequirement(EXTRA, "onnxruntime-gpu", ("cuda", "cudnn"), "1.31.0"),
    ),
}
REFUSED_REQUIREMENTS = {
    "no specifier": "torch",
    "a lower bound": "onnx>=1.21",
    "a wildcard": "torch==2.*",
    "a url": "torch @ https://download.pytorch.org/torch.whl",
    "a marker": "torch==2.15.0; sys_platform == 'linux'",
    "whitespace": "torch == 2.15.0",
    "a name not normalized": "ml_dtypes==0.6.0",
    "an uppercase name": "Torch==2.15.0",
}


class TestTheDeclaredRequirements:
    class TestTheParseDeclaredRequirementFunction:
        @pytest.mark.parametrize(
            ("text", "requirement"),
            list(ACCEPTED_REQUIREMENTS.values()),
            ids=list(ACCEPTED_REQUIREMENTS.keys()),
        )
        class TestWhenTheRequirementPinsAnExactVersion:
            @pytest.fixture
            def parsed(self, text: str) -> DeclaredRequirement:
                return parse_declared_requirement(text, EXTRA)

            def test_should_read_its_name_extras_and_version(
                self, parsed: DeclaredRequirement, requirement: DeclaredRequirement
            ) -> None:
                assert parsed == requirement

        @pytest.mark.parametrize(
            "text", list(REFUSED_REQUIREMENTS.values()), ids=list(REFUSED_REQUIREMENTS.keys())
        )
        class TestWhenTheRequirementIsNotAnExactPin:
            @pytest.fixture
            def parse_act(self, text: str) -> Callable[[], DeclaredRequirement]:
                return partial(parse_declared_requirement, text, EXTRA)

            def test_should_refuse_it(
                self, parse_act: Callable[[], DeclaredRequirement], text: str
            ) -> None:
                with pytest.raises(DeclaredRequirementError, match=re.escape(f"{text!a}")):
                    parse_act()

    class TestTheReadDeclaredRequirementsFunction:
        class TestWhenTheManifestDeclaresNoExtra:
            @pytest.fixture
            def requirements(self) -> tuple[DeclaredRequirement, ...]:
                return read_declared_requirements(None)

            def test_should_find_none(self, requirements: tuple[DeclaredRequirement, ...]) -> None:
                assert requirements == ()

        class TestWhenAnExtraIsNotNormalized:
            @pytest.fixture
            def read_act(self) -> Callable[[], tuple[DeclaredRequirement, ...]]:
                return partial(read_declared_requirements, {"Core_Bench": ["torch==2.15.0"]})

            def test_should_refuse_it(
                self, read_act: Callable[[], tuple[DeclaredRequirement, ...]]
            ) -> None:
                with pytest.raises(
                    DeclaredRequirementError, match="'Core_Bench' is not a normalized"
                ):
                    read_act()
