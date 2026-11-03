import re
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from pathlib import PurePosixPath

import pytest
from run_step_fakes import (
    GROUP_ID,
    USER_ID,
    FakeContainerEngine,
    FakeStepFiles,
    build_directory_facts,
    build_step_output,
)

from mostt_model_builder.application.convert_variants import (
    ConversionOutcome,
    ConversionRun,
    convert_variants,
)
from mostt_model_builder.application.run_step import HostIdentity, ImageFacts
from mostt_model_builder.domain.containers.account_ranges import HostAccounts
from mostt_model_builder.domain.containers.container_files import (
    ContainerFileError,
    FileFacts,
    FileKind,
)
from mostt_model_builder.domain.containers.container_steps import (
    CONTAINER_USER_ID,
    ONNX_CONVERT_IMAGE,
    PEAKS_FILE_NAME,
    BindMount,
    ContainerRun,
    DeclaredOutput,
    EarlierOutput,
    build_convert_row,
)
from mostt_model_builder.domain.containers.step_peaks import StepPeaks, render_step_peaks
from mostt_model_builder.domain.onnx.conversion_report import (
    CONVERSION_REPORT_FILE_NAME,
    ConversionReport,
    ConversionReportError,
    ConvertedFile,
    render_conversion_report,
)
from mostt_model_builder.domain.onnx.conversion_request import (
    CONVERSION_REQUEST_FILE_NAME,
    ConversionKind,
    ConversionRequest,
    EightBitBlocksConversion,
    FloatConversion,
    parse_conversion_request,
)

WORK_ROOT = PurePosixPath("/data/builds/work")
HEAD_TREE = "3f1d1ffe79bc3336eb583331cfbf9b189fd45f01"
COMMIT = "0123456789abcdef0123456789abcdef01234567"
IMAGE_ID = "sha256:" + "c" * 64
RECIPE_NAME = "family"
RUN_DIRECTORY = PurePosixPath("runs/family", COMMIT)
EXPORT_DIRECTORY = WORK_ROOT / RUN_DIRECTORY / "export"
ENCODER_NAME = "encoder-model.onnx"
ENCODER_DATA_NAME = "encoder-model.onnx.data"
FLOAT16_NAME = "encoder-model.fp16.onnx"
EIGHT_BIT_NAME = "encoder-model.int8.onnx"
ENCODER = b"\x08\x07\x3a\x00"
ENCODER_DATA = bytes(4096)
EXPORT_OUTPUT_NAMES = (ENCODER_NAME, ENCODER_DATA_NAME, PEAKS_FILE_NAME)
PEAKS = StepPeaks(memory_peak_bytes=12 * 2**30, process_count_peak=9)
CONVERT_ROW = build_convert_row(
    (
        EarlierOutput("export", DeclaredOutput(ENCODER_NAME, 2**20), EXPORT_OUTPUT_NAMES),
        EarlierOutput("export", DeclaredOutput(ENCODER_DATA_NAME, 2**31), EXPORT_OUTPUT_NAMES),
    ),
    (DeclaredOutput(FLOAT16_NAME, 2**31), DeclaredOutput(EIGHT_BIT_NAME, 2**31)),
)
REQUEST = ConversionRequest(
    "export",
    ENCODER_NAME,
    (
        FloatConversion(ConversionKind.FLOAT16, FLOAT16_NAME),
        EightBitBlocksConversion(
            EIGHT_BIT_NAME,
            block_size=64,
            is_symmetric=True,
            accuracy_level=4,
            quantized_product_count=265,
        ),
    ),
)
FLOAT16_FILE = ConvertedFile(
    kind=ConversionKind.FLOAT16,
    output=FLOAT16_NAME,
    node_count=1648,
    quantized_product_count=0,
    kept_float32_operators=(),
    clamped_initializer_count=75,
    below_min_positive_count=95,
    above_max_finite_count=0,
    realigned_cast_count=5,
)
EIGHT_BIT_FILE = ConvertedFile(
    kind=ConversionKind.EIGHT_BIT_BLOCKS,
    output=EIGHT_BIT_NAME,
    node_count=1643,
    quantized_product_count=265,
    kept_float32_operators=(),
    clamped_initializer_count=0,
    below_min_positive_count=0,
    above_max_finite_count=0,
    realigned_cast_count=0,
)
REPORT = ConversionReport((FLOAT16_FILE, EIGHT_BIT_FILE))
HOST = HostIdentity(USER_ID, GROUP_ID, 20, HostAccounts(False, (), ()))
RUN = ConversionRun(
    row=CONVERT_ROW,
    request=REQUEST,
    recipe_name=RECIPE_NAME,
    commit=COMMIT,
    work_root=WORK_ROOT,
    head_tree=HEAD_TREE,
    host=HOST,
)
REQUEST_PATH = WORK_ROOT / RUN_DIRECTORY / "requests/convert" / CONVERSION_REQUEST_FILE_NAME


@dataclass(frozen=True)
class CompletedConversion:
    outcome: ConversionOutcome
    runs: list[ContainerRun]
    files: FakeStepFiles


def build_files(encoder_facts: FileFacts | None = None) -> FakeStepFiles:
    files = FakeStepFiles()
    files.add_directory(WORK_ROOT)
    files.add_directory(EXPORT_DIRECTORY, build_directory_facts(0o2770))
    encoder, encoder_content = build_step_output(ENCODER)
    files.add_file(EXPORT_DIRECTORY / ENCODER_NAME, encoder_facts or encoder, encoder_content)
    files.add_file(EXPORT_DIRECTORY / ENCODER_DATA_NAME, *build_step_output(ENCODER_DATA))
    files.add_file(EXPORT_DIRECTORY / PEAKS_FILE_NAME, *build_step_output(b"{}"))
    return files


def build_engine(files: FakeStepFiles, report: bytes) -> FakeContainerEngine:
    return FakeContainerEngine(
        files=files,
        images={ONNX_CONVERT_IMAGE.tag: ImageFacts(IMAGE_ID, HEAD_TREE)},
        step_outputs={
            FLOAT16_NAME: build_step_output(b"\x08\x07\x3a\x00"),
            EIGHT_BIT_NAME: build_step_output(b"\x08\x07\x3a\x00"),
            CONVERSION_REPORT_FILE_NAME: build_step_output(report),
            PEAKS_FILE_NAME: build_step_output(render_step_peaks(PEAKS)),
        },
    )


class TestTheConversionOfTheVariants:
    class TestTheConvertVariantsFunction:
        class TestWhenTheConversionsWriteWhatTheRequestAsked:
            @pytest.fixture
            def conversion(self) -> CompletedConversion:
                files = build_files()
                engine = build_engine(files, render_conversion_report(REPORT))
                return CompletedConversion(convert_variants(RUN, engine, files), engine.runs, files)

            def test_should_write_the_request_of_every_conversion_for_the_container(
                self, conversion: CompletedConversion
            ) -> None:
                assert parse_conversion_request(conversion.files.contents[REQUEST_PATH]) == REQUEST

            def test_should_run_the_cores_conversion_once(
                self, conversion: CompletedConversion
            ) -> None:
                assert [run.entry_module for run in conversion.runs] == [CONVERT_ROW.entry_module]

            def test_should_bind_the_input_and_its_external_data_side_by_side_read_only(
                self, conversion: CompletedConversion
            ) -> None:
                assert [
                    mount
                    for run in conversion.runs
                    for mount in run.mounts
                    if mount.is_written_by_a_step
                ] == [
                    BindMount(
                        RUN_DIRECTORY / "export" / name,
                        PurePosixPath("/inputs/export") / name,
                        is_read_only=True,
                        is_file=True,
                        is_written_by_a_step=True,
                    )
                    for name in (ENCODER_NAME, ENCODER_DATA_NAME)
                ]

            def test_should_give_back_the_report_of_each_conversion(
                self, conversion: CompletedConversion
            ) -> None:
                assert conversion.outcome.report == REPORT

            def test_should_give_back_the_size_of_each_converted_file(
                self, conversion: CompletedConversion
            ) -> None:
                assert {
                    name: conversion.outcome.outputs.sizes_bytes[name]
                    for name in (FLOAT16_NAME, EIGHT_BIT_NAME)
                } == {FLOAT16_NAME: 4, EIGHT_BIT_NAME: 4}

            def test_should_read_no_converted_file_since_no_parser_of_the_host_reads_them(
                self, conversion: CompletedConversion
            ) -> None:
                assert [path.name for path in conversion.files.read_paths] == [
                    CONVERSION_REPORT_FILE_NAME,
                    PEAKS_FILE_NAME,
                ]

            def test_should_give_back_the_peaks_of_the_conversions(
                self, conversion: CompletedConversion
            ) -> None:
                assert conversion.outcome.outputs.peaks == PEAKS

        class TestWhenTheInputHasASecondLink:
            @pytest.fixture
            def convert_act(self) -> Callable[[], ConversionOutcome]:
                files = build_files(
                    FileFacts(FileKind.REGULAR, CONTAINER_USER_ID, 0o640, len(ENCODER), 2)
                )
                engine = build_engine(files, render_conversion_report(REPORT))
                return partial(convert_variants, RUN, engine, files)

            def test_should_refuse_it_before_any_container_reads_it(
                self, convert_act: Callable[[], ConversionOutcome]
            ) -> None:
                with pytest.raises(
                    ContainerFileError,
                    match=re.escape("the output encoder-model.onnx has 2 links, not one"),
                ):
                    convert_act()

        class TestWhenTheReportQuantizedFewerProductsThanDeclared:
            @pytest.fixture
            def convert_act(self) -> Callable[[], ConversionOutcome]:
                files = build_files()
                report = ConversionReport(
                    (
                        FLOAT16_FILE,
                        ConvertedFile(
                            kind=ConversionKind.EIGHT_BIT_BLOCKS,
                            output=EIGHT_BIT_NAME,
                            node_count=1643,
                            quantized_product_count=217,
                            kept_float32_operators=(),
                            clamped_initializer_count=0,
                            below_min_positive_count=0,
                            above_max_finite_count=0,
                            realigned_cast_count=0,
                        ),
                    )
                )
                engine = build_engine(files, render_conversion_report(report))
                return partial(convert_variants, RUN, engine, files)

            def test_should_refuse_the_conversion_and_say_why(
                self, convert_act: Callable[[], ConversionOutcome]
            ) -> None:
                with pytest.raises(
                    ConversionReportError,
                    match=re.escape(
                        "the conversion to encoder-model.int8.onnx quantized 217 products to "
                        "8-bit blocks, not 265"
                    ),
                ):
                    convert_act()
