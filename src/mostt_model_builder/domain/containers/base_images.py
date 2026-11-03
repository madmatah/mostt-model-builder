from pathlib import PurePosixPath

# The bases, by digest, so that two builds of the same commit start from the same bytes whatever a
# tag points to by then.
PYTHON_BASE_IMAGE = (
    "docker.io/library/python:3.12-slim-bookworm"
    "@sha256:6a0e06f5ba0b373d051aabd3fffb9b873f27f9a2a3a14bcbe37dee6f0e3bbc70"
)
UV_VERSION = "0.13.7"
UV_IMAGE = (
    f"ghcr.io/astral-sh/uv:{UV_VERSION}"
    "@sha256:96f40d3904aad8d0bd010539001d36c8e2188b981e1334471a90c4a1d93bb490"
)
BASE_IMAGES = frozenset({PYTHON_BASE_IMAGE, UV_IMAGE})
BASE_PYTHON = PurePosixPath("/usr/local/bin/python3.12")
ENVIRONMENT_PYTHON = PurePosixPath("/opt/venv/bin/python")
