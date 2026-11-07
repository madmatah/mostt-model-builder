from dataclasses import replace

import pytest
from cache_arrangement import (
    EXPORT_ROW,
    GPU_HARDWARE,
    GPU_IMAGE,
    HARDWARE_WITHOUT_GPU,
    TREE_FILES,
    build_recipe,
)

from mostt_model_builder.application.cache_entries import EntryKeys, KeySkip, SkipCause
from mostt_model_builder.domain.cache.build_units import BuildUnit, UnitKind, plan_row
from mostt_model_builder.domain.cache.output_keys import PlainData

GPU_STEP_ROW = replace(EXPORT_ROW, name="describe", image=GPU_IMAGE, uses_gpu=True)
GPU_STEP_UNIT = plan_row("describe", UnitKind.PLAIN, GPU_STEP_ROW, {})


class TestTheEntryKeys:
    class TestTheKeyUnitMethod:
        class TestWhenAPlainStepRunsOnTheGpu:
            @pytest.fixture
            def document(self) -> dict[str, PlainData]:
                _, document = EntryKeys(build_recipe(), TREE_FILES, GPU_HARDWARE).key_unit(
                    GPU_STEP_UNIT, {}
                )
                return document

            def test_should_key_it_by_the_gpu_it_runs_on(
                self, document: dict[str, PlainData]
            ) -> None:
                assert document["hardware"] == {
                    "cpu_models": list(GPU_HARDWARE.cpu_models),
                    "gpu_models": list(GPU_HARDWARE.gpu_models),
                    "driver_version": GPU_HARDWARE.driver_version,
                }

    class TestTheFindKeySkipMethod:
        @pytest.mark.parametrize(
            "unit",
            [
                pytest.param(GPU_STEP_UNIT, id="a plain step"),
                pytest.param(
                    plan_row(
                        "transcribe",
                        UnitKind.RUN,
                        replace(GPU_STEP_ROW, name="transcribe-float32-gpu-fr_fr-0"),
                        {},
                    ),
                    id="a run",
                ),
            ],
        )
        class TestWhenAStepOnTheGpuFindsNoGpu:
            @pytest.fixture
            def skip(self, unit: BuildUnit) -> KeySkip | None:
                return EntryKeys(build_recipe(), TREE_FILES, HARDWARE_WITHOUT_GPU).find_key_skip(
                    unit, {}, ()
                )

            def test_should_skip_it_for_want_of_a_gpu(self, skip: KeySkip | None) -> None:
                assert skip == KeySkip(SkipCause.NO_GPU)
