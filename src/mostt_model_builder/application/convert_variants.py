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
from mostt_model_builder.domain.containers.container_steps import StepRow
from mostt_model_builder.domain.containers.work_directory import place_step
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


@dataclass(frozen=True, slots=True)
class ConversionRun:
    row: StepRow
    request: ConversionRequest
    recipe_name: str
    commit: str
    work_root: PurePosixPath
    head_tree: str
    host: HostIdentity


@dataclass(frozen=True, slots=True)
class ConversionOutcome:
    outputs: StepOutputs
    report: ConversionReport


def convert_variants(
    run: ConversionRun, engine: ContainerEngine, files: StepFiles
) -> ConversionOutcome:
    outputs = run_step(
        StepRunRequest(
            row=run.row,
            placement=place_step(run.recipe_name, run.commit, run.row.name),
            request_files={CONVERSION_REQUEST_FILE_NAME: render_conversion_request(run.request)},
            source_paths={},
            work_root=run.work_root,
            head_tree=run.head_tree,
            host=run.host,
            earlier_output_directories={
                earlier_output.step: place_step(
                    run.recipe_name, run.commit, earlier_output.step
                ).output_directory
                for earlier_output in run.row.earlier_outputs
            },
            parsed_output_names=frozenset({CONVERSION_REPORT_FILE_NAME}),
        ),
        engine,
        files,
    )
    report = parse_conversion_report(outputs.contents[CONVERSION_REPORT_FILE_NAME], run.request)
    return ConversionOutcome(outputs, report)
