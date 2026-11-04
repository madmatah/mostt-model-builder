from collections.abc import Sequence
from dataclasses import dataclass

from mostt_model_builder.domain.containers.base_images import ENVIRONMENT_PYTHON
from mostt_model_builder.domain.containers.container_steps import Device
from mostt_model_builder.domain.contract.derived_rows import DerivedRow
from mostt_model_builder.domain.contract.recipe_contract import REFERENCE_VARIANT
from mostt_model_builder.domain.lock.declared_requirements import DeclaredRequirement

DEVICE_ORDER = (Device.CPU, Device.GPU)


@dataclass(frozen=True, slots=True)
class MeasuringLibrary:
    device: Device
    name: str
    version: str


def select_measuring_libraries(
    names: Sequence[str],
    requirements: Sequence[DeclaredRequirement],
    derived_rows: Sequence[DerivedRow],
) -> tuple[MeasuringLibrary, ...]:
    extras_by_device: dict[Device, set[str]] = {}
    for derived_row in derived_rows:
        image = derived_row.row.image
        if derived_row.request.variant == REFERENCE_VARIANT or image.python != ENVIRONMENT_PYTHON:
            continue
        extras_by_device.setdefault(derived_row.request.device, set()).add(
            f"{image.owner}-{image.name}"
        )
    versions: dict[tuple[Device, str], set[str]] = {}
    for requirement in requirements:
        for device, extras in extras_by_device.items():
            if requirement.name in names and requirement.extra in extras:
                versions.setdefault((device, requirement.name), set()).add(requirement.version)
    return tuple(
        MeasuringLibrary(device, name, version)
        for device in DEVICE_ORDER
        for name in names
        for version in sorted(versions.get((device, name), ()))
    )
