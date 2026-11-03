from pathlib import Path

import onnx

from mostt_model_builder.domain.containers.container_steps import (
    CANDIDATE_IN_THE_CONTAINER,
    OUTPUT_IN_THE_CONTAINER,
    REQUEST_IN_THE_CONTAINER,
)
from mostt_model_builder.domain.onnx.inspection_report import (
    INSPECTION_REPORT_FILE_NAME,
    INSPECTION_REQUEST_FILE_NAME,
    FileInspection,
    InspectionReport,
    LocalFunction,
    Opset,
    parse_inspection_request,
    render_inspection_report,
)
from mostt_model_builder.domain.onnx.model_graphs import (
    count_nodes,
    count_nodes_with_metadata,
    count_sparse_initializers,
    count_training_graphs,
    list_external_tensors,
    list_products,
)
from mostt_model_builder.steps.output_modes import finish_outputs


# onnx is installed in the inspection's image alone, so mypy on the host cannot check that its
# messages match the Protocols of model_graphs, which name only fields of onnx/onnx.proto.
def inspect_file(path: Path) -> FileInspection:
    model = onnx.load(str(path), load_external_data=False)
    return FileInspection(
        name=path.name,
        opsets=tuple(Opset(opset.domain, opset.version) for opset in model.opset_import),
        functions=tuple(
            LocalFunction(function.domain, function.name) for function in model.functions
        ),
        nodes=count_nodes(model),
        metadata_props_node_count=count_nodes_with_metadata(model),
        products=list_products(model),
        external_tensors=list_external_tensors(model),
        training_graph_count=count_training_graphs(model),
        sparse_initializer_count=count_sparse_initializers(model),
    )


def inspect_candidate(
    request_directory: Path, candidate_directory: Path, output_directory: Path
) -> None:
    file_names = parse_inspection_request(
        (request_directory / INSPECTION_REQUEST_FILE_NAME).read_bytes()
    )
    report = InspectionReport(
        tuple(inspect_file(candidate_directory / name) for name in file_names)
    )
    (output_directory / INSPECTION_REPORT_FILE_NAME).write_bytes(render_inspection_report(report))


def main() -> None:
    output_directory = Path(OUTPUT_IN_THE_CONTAINER)
    inspect_candidate(
        Path(REQUEST_IN_THE_CONTAINER), Path(CANDIDATE_IN_THE_CONTAINER), output_directory
    )
    finish_outputs(output_directory)


if __name__ == "__main__":
    main()
