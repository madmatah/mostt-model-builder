from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from mostt_model_builder.domain.bench.fleurs_index import FLEURS_REPOSITORY, locate_bench_files
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
    candidate_files: tuple[str, ...] = ()


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
    binds_bench: bool = False


@dataclass(frozen=True, slots=True)
class DerivedRow:
    step: str
    run: str
    language: str
    row: StepRow
    request: ShardRequest


def name_derived_row(step_name: str, run_name: str, language: str, shard: int) -> str:
    return f"{step_name}-{run_name}-{language}-{shard}"


def locate_bench_sources(language: str) -> tuple[SourceReference, ...]:
    return tuple(SourceReference(FLEURS_REPOSITORY, path) for path in locate_bench_files(language))


def derive_rows(template: StepTemplate, runs: Sequence[ShardedRun]) -> tuple[DerivedRow, ...]:
    return tuple(
        derive_row(template, run, language, shard)
        for run in runs
        for language in run.languages
        for shard in range(run.shard_count)
    )


def derive_row(template: StepTemplate, run: ShardedRun, language: str, shard: int) -> DerivedRow:
    device_row = template.devices[run.device]
    candidate_files = run.candidate_files if template.binds_candidate else ()
    bench_sources = locate_bench_sources(language) if template.binds_bench else ()
    return DerivedRow(
        step=template.name,
        run=run.name,
        language=language,
        row=StepRow(
            name=name_derived_row(template.name, run.name, language, shard),
            image=device_row.image,
            entry_module=template.entry_module,
            sources=(*template.sources, *bench_sources),
            has_request=True,
            outputs=(*device_row.outputs, *template.outputs),
            limits=device_row.limits,
            uses_gpu=run.device is Device.GPU,
            earlier_outputs=template.earlier_outputs,
            candidate_files=candidate_files,
        ),
        request=ShardRequest(
            run=run.name,
            variant=run.variant,
            device=run.device,
            languages=(language,),
            shard=shard,
            shard_count=run.shard_count,
            files=candidate_files,
        ),
    )
