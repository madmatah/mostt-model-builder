import argparse
import sys
from collections.abc import Sequence

PROGRAM_NAME = "model-build.sh"
DESCRIPTION = (
    "Builds speech recognition models from their original weights, in confined containers, "
    "and verifies them on FLEURS."
)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=PROGRAM_NAME, description=DESCRIPTION, add_help=False, allow_abbrev=False
    )
    parser.add_argument("--help", action="help", help="show this message and exit")
    return parser


def main(arguments: Sequence[str]) -> None:
    parser = build_argument_parser()
    parser.parse_args(arguments)
    parser.error("no command given")


if __name__ == "__main__":
    main(sys.argv[1:])
