from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from mostt_model_builder.domain.containers.container_steps import (
    ContainerImage,
    DeclaredOutput,
    EarlierOutput,
    SourceReference,
    StepLimits,
    StepRow,
)
from mostt_model_builder.domain.containers.devices import Device
from mostt_model_builder.domain.contract.shard_request import ShardRequest


@dataclass(frozen=True, slots=True)
class ShardedRun:
    name: str
    variant: str
    device: Device
    languages: tuple[str, ...]
    shard_count: int


@dataclass(frozen=True, slots=True)
class DeviceRow:
    image: ContainerImage
    limits: StepLimits
    outputs: tuple[DeclaredOutput, ...]


@dataclass(frozen=True, slots=True)
class StepTemplate:
    name: str
    entry_module: str
    sources: tuple[SourceReference, ...]
    earlier_outputs: tuple[EarlierOutput, ...]
    binds_candidate: bool
    outputs: tuple[DeclaredOutput, ...]
    devices: Mapping[Device, DeviceRow]


@dataclass(frozen=True, slots=True)
class DerivedRow:
    step: str
    run: str
    row: StepRow
    request: ShardRequest


def name_derived_row(step_name: str, run_name: str, shard: int, is_single_row: bool) -> str:
    if is_single_row:
        return step_name
    return f"{step_name}-{run_name}-{shard}"


def derive_rows(template: StepTemplate, runs: Sequence[ShardedRun]) -> tuple[DerivedRow, ...]:
    is_single_row = sum(run.shard_count for run in runs) == 1
    return tuple(
        derive_row(template, run, shard, is_single_row)
        for run in runs
        for shard in range(run.shard_count)
    )


def derive_row(
    template: StepTemplate, run: ShardedRun, shard: int, is_single_row: bool
) -> DerivedRow:
    device_row = template.devices[run.device]
    return DerivedRow(
        step=template.name,
        run=run.name,
        row=StepRow(
            name=name_derived_row(template.name, run.name, shard, is_single_row),
            image=device_row.image,
            entry_module=template.entry_module,
            sources=template.sources,
            has_request=True,
            outputs=(*device_row.outputs, *template.outputs),
            limits=device_row.limits,
            uses_gpu=run.device is Device.GPU,
            earlier_outputs=template.earlier_outputs,
            binds_candidate=template.binds_candidate,
        ),
        request=ShardRequest(
            run=run.name,
            variant=run.variant,
            device=run.device,
            languages=run.languages,
            shard=shard,
            shard_count=run.shard_count,
        ),
    )
