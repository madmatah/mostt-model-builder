import pytest

from mostt_model_builder.domain.containers.base_images import BASE_PYTHON, ENVIRONMENT_PYTHON
from mostt_model_builder.domain.containers.container_steps import (
    ContainerImage,
    Device,
    StepLimits,
    StepRow,
)
from mostt_model_builder.domain.contract.derived_rows import DerivedRow, ShardRequest
from mostt_model_builder.domain.contract.recipe_contract import REFERENCE_VARIANT
from mostt_model_builder.domain.lock.declared_requirements import DeclaredRequirement
from mostt_model_builder.domain.publication.measuring_libraries import (
    MeasuringLibrary,
    select_measuring_libraries,
)

LIMITS = StepLimits(memory_mib=1024, process_count=64, cpu_count=2, temporary_directory_mib=64)
NAMES = ("onnx-asr", "onnxruntime", "onnxruntime-gpu")
CPU_IMAGE = ContainerImage("family", "transcribe-cpu", Device.CPU, ENVIRONMENT_PYTHON)
GPU_IMAGE = ContainerImage("family", "transcribe-gpu", Device.GPU, ENVIRONMENT_PYTHON)
REFERENCE_IMAGE = ContainerImage("family", "reference", Device.CPU, ENVIRONMENT_PYTHON)
PLAIN_IMAGE = ContainerImage("family", "plain", Device.CPU, BASE_PYTHON)
REQUIREMENTS = (
    DeclaredRequirement("family-transcribe-cpu", "numpy", (), "2.5.4"),
    DeclaredRequirement("family-transcribe-cpu", "onnx-asr", (), "0.12.0"),
    DeclaredRequirement("family-transcribe-cpu", "onnxruntime", (), "1.31.0"),
    DeclaredRequirement("family-transcribe-gpu", "numpy", (), "2.5.4"),
    DeclaredRequirement("family-transcribe-gpu", "onnx-asr", (), "0.12.0"),
    DeclaredRequirement("family-transcribe-gpu", "onnxruntime-gpu", ("cuda", "cudnn"), "1.31.0"),
    DeclaredRequirement("family-reference", "onnx-asr", (), "0.11.0"),
    DeclaredRequirement("family-reference", "torch", (), "2.15.0"),
    DeclaredRequirement("family-plain", "onnx-asr", (), "0.10.0"),
)


def build_derived_row(run: str, variant: str, image: ContainerImage) -> DerivedRow:
    return DerivedRow(
        step="transcribe",
        run=run,
        row=StepRow(
            name=f"transcribe-{run}-0",
            image=image,
            entry_module="mostt_recipe_family.steps.transcribe",
            sources=(),
            has_request=True,
            outputs=(),
            limits=LIMITS,
        ),
        request=ShardRequest(run, variant, image.device, ("fr_fr",), 0, 1),
    )


# The GPU's row comes first, so that the order of the libraries is the selection's own.
DERIVED_ROWS = (
    build_derived_row("float32-gpu", "float32", GPU_IMAGE),
    build_derived_row("float32-cpu", "float32", CPU_IMAGE),
    build_derived_row("nemo", REFERENCE_VARIANT, REFERENCE_IMAGE),
    build_derived_row("float16-cpu", "float16", PLAIN_IMAGE),
)


class TestTheMeasuringLibraries:
    class TestTheSelectMeasuringLibrariesFunction:
        class TestWhenTheVariantsRunOnBothDevices:
            @pytest.fixture
            def libraries(self) -> tuple[MeasuringLibrary, ...]:
                return select_measuring_libraries(NAMES, REQUIREMENTS, DERIVED_ROWS)

            def test_should_keep_each_named_library_by_device_the_cpu_first_in_the_names_order(
                self, libraries: tuple[MeasuringLibrary, ...]
            ) -> None:
                assert libraries == (
                    MeasuringLibrary(Device.CPU, "onnx-asr", "0.12.0"),
                    MeasuringLibrary(Device.CPU, "onnxruntime", "1.31.0"),
                    MeasuringLibrary(Device.GPU, "onnx-asr", "0.12.0"),
                    MeasuringLibrary(Device.GPU, "onnxruntime-gpu", "1.31.0"),
                )
