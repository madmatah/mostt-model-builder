from pathlib import PurePosixPath

CONTAINER_ROOT = PurePosixPath("/")
SOURCES_IN_THE_CONTAINER = CONTAINER_ROOT / "sources"
REQUEST_IN_THE_CONTAINER = PurePosixPath("/request")
OUTPUT_IN_THE_CONTAINER = PurePosixPath("/output")
INPUTS_IN_THE_CONTAINER = PurePosixPath("/inputs")
CANDIDATE_IN_THE_CONTAINER = PurePosixPath("/candidate")
TEMPORARY_DIRECTORY_IN_THE_CONTAINER = PurePosixPath("/tmp")
SCRATCH_IN_THE_CONTAINER = PurePosixPath("/scratch")
