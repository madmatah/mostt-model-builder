from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import PurePosixPath

from mostt_model_builder.application.run_step import (
    ContainerEngine,
    HostIdentity,
    StepFiles,
    StepOutputs,
    StepRunRequest,
    run_step,
)
from mostt_model_builder.domain.containers.container_steps import StepPlacement, StepRow
from mostt_model_builder.domain.onnx.conversion_report import (
    CONVERSION_REPORT_FILE_NAME,
    ConversionReport,
    parse_conversion_report,
)
from mostt_model_builder.domain.onnx.conversion_request import (
    CONVERSION_REQUEST_FILE_NAME,
    ConversionRequest,
    render_conversion_request,
)

CONVERSION_PARSED_NAMES = frozenset({CONVERSION_REPORT_FILE_NAME})


@dataclass(frozen=True, slots=True)
class ConversionRun:
    row: StepRow
    request: ConversionRequest
    placement: StepPlacement
    earlier_output_directories: Mapping[str, PurePosixPath]
    work_root: PurePosixPath
    head_tree: str
    host: HostIdentity


@dataclass(frozen=True, slots=True)
class ConversionOutcome:
    request: ConversionRequest
    outputs: StepOutputs
    report: ConversionReport


def convert_variants(
    run: ConversionRun, engine: ContainerEngine, files: StepFiles
) -> ConversionOutcome:
    outputs = run_step(
        StepRunRequest(
            row=run.row,
            placement=run.placement,
            request_files={CONVERSION_REQUEST_FILE_NAME: render_conversion_request(run.request)},
            source_paths={},
            work_root=run.work_root,
            head_tree=run.head_tree,
            host=run.host,
            earlier_output_directories=run.earlier_output_directories,
            parsed_output_names=CONVERSION_PARSED_NAMES,
        ),
        engine,
        files,
    )
    return read_conversion(run.request, outputs)


def read_conversion(request: ConversionRequest, outputs: StepOutputs) -> ConversionOutcome:
    report = parse_conversion_report(outputs.contents[CONVERSION_REPORT_FILE_NAME], request)
    return ConversionOutcome(request, outputs, report)
