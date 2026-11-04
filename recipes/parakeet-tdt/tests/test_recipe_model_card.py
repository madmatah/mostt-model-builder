import tomllib
from pathlib import Path

import pytest

from mostt_model_builder.domain.acceptance.paired_comparison import PairedComparison
from mostt_model_builder.domain.acceptance.parity_reports import ParityReport
from mostt_model_builder.domain.acceptance.verification import (
    CandidateDigest,
    ComparisonMeasure,
    RunMeasure,
    Verification,
    measure_parity,
)
from mostt_model_builder.domain.contract.recipe_contract import (
    JudgedRecipe,
    RecipeFacts,
    judge_recipe,
)
from mostt_model_builder.domain.lock.declared_requirements import (
    DeclaredRequirement,
    read_declared_requirements,
)
from mostt_model_builder.domain.publication.measuring_libraries import select_measuring_libraries
from mostt_model_builder.domain.publication.model_card import CardInputs, render_model_card
from mostt_model_builder.host_main import CLONE_ROOT, find_present_entry_modules, read_pins
from mostt_recipe_parakeet_tdt.domain.publication.model_description import MODEL_DESCRIPTION
from mostt_recipe_parakeet_tdt.recipe import RECIPE

RECIPE_DIRECTORY = Path(__file__).resolve().parent.parent
IMAGES_MANIFEST_PATH = CLONE_ROOT / "images" / "environment.toml"
EXPECTED_CARD_PATH = RECIPE_DIRECTORY / "tests" / "fixtures" / "model-card.md"
HOST_CPU_COUNT = 64
BUILD_COMMIT = "89abcdef0123456789abcdef0123456789abcdef"
LICENSE_LINK = "[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)"


def judge_the_recipe() -> JudgedRecipe:
    return judge_recipe(
        RECIPE,
        RecipeFacts(
            core_pins=read_pins(CLONE_ROOT),
            recipe_pins=read_pins(RECIPE_DIRECTORY),
            present_entry_modules=find_present_entry_modules(RECIPE),
            host_cpu_count=HOST_CPU_COUNT,
        ),
    )


def build_verification(recipe: JudgedRecipe) -> Verification:
    return Verification(
        runs=tuple(
            RunMeasure(run.name, language, 350, 10_000, 450, 520, 30, 1000.0)
            for run in recipe.runs
            for language in run.languages
        ),
        comparisons=tuple(
            ComparisonMeasure(
                run=comparison.run,
                reference_run=comparison.reference_run,
                language=language,
                rule=comparison.rule,
                identical_count=349,
                compared_count=350,
                paired=PairedComparison(0.01, -0.05, 0.07),
                failures=(),
            )
            for comparison in recipe.comparisons
            for run in recipe.runs
            if run.name == comparison.run
            for language in run.languages
        ),
        parities=tuple(
            measure_parity(
                ParityReport(parity.name, 1e-6, 20), parity.absolute_difference_tolerance
            )
            for parity in recipe.parities
        ),
        candidate=tuple(
            CandidateDigest(
                candidate_file.name,
                candidate_file.file_format,
                candidate_file.variants,
                1,
                "e" * 64,
            )
            for candidate_file in recipe.candidate
        ),
    )


def read_manifest_requirements() -> tuple[DeclaredRequirement, ...]:
    manifest = tomllib.loads(IMAGES_MANIFEST_PATH.read_text(encoding="utf-8"))
    return read_declared_requirements(manifest["project"]["optional-dependencies"])


def render_the_card() -> str:
    recipe = judge_the_recipe()
    card = render_model_card(
        CardInputs(
            template=(RECIPE_DIRECTORY / MODEL_DESCRIPTION.template_path).read_text(
                encoding="utf-8"
            ),
            description=MODEL_DESCRIPTION,
            pins=recipe.pins,
            verification=build_verification(recipe),
            runs=recipe.runs,
            conversions=recipe.conversions,
            measuring_libraries=select_measuring_libraries(
                MODEL_DESCRIPTION.measuring_library_names,
                read_manifest_requirements(),
                recipe.derived_rows,
            ),
            build_commit=BUILD_COMMIT,
        )
    )
    return card.decode("utf-8")


class TestTheModelCardOfTheParakeetTdtRecipe:
    class TestWhenRenderedFromFixedMeasures:
        @pytest.fixture(scope="class")
        @classmethod
        def card(cls) -> str:
            return render_the_card()

        def test_should_give_the_reviewed_card(self, card: str) -> None:
            assert card == EXPECTED_CARD_PATH.read_text(encoding="utf-8")

        def test_should_link_the_code_that_built_the_files_at_its_commit(self, card: str) -> None:
            assert (
                "[mostt-model-builder](https://github.com/madmatah/mostt-model-builder/tree/"
                f"{BUILD_COMMIT}), at the commit `{BUILD_COMMIT}`" in card
            )

        def test_should_credit_both_works_under_their_license(self, card: str) -> None:
            attribution = card.partition("## License and attribution")[2]
            assert (
                f"| Moondream | `73175eb7aeb0d82f1e2a6b53b3aabc10a90bcd0b` | {LICENSE_LINK} |"
                in attribution,
                f"| NVIDIA | `541d1f99c6b0c3cd0b11a95167540bb8edefd82b` | {LICENSE_LINK} |"
                in attribution,
            ) == (True, True)
