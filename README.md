# mostt-model-builder

mostt-model-builder is the tooling of reproducible, confined builds of speech recognition models from their original weights: export, quantization and FLEURS verification.

- **Built:** the command line, `model-build.sh`, which runs on the host's Python and the standard library, isolated from the environment and the site packages, and refuses to run as root.
- **Planned:** its commands, and the build they run: a core, with the pinned sources, the confined containers and their boundary with the host, the formats and their inspection, the FLEURS bench and its acceptance; and one recipe per model family, the first for Parakeet TDT, which builds Parakeet Ultra in ONNX. The checked files land in a candidate directory laid out for Hugging Face, where the maintainer publishes them, and mostt, the local dictation project, pins them by hash.

## Requirements

- Python 3.12 on the host, as `python3`, which runs the command line on its standard library alone;
- git, and the GitHub CLI (`gh`), which the pull requests use;
- [uv](https://docs.astral.sh/uv/) 0.13.7, the version CI runs, which finds or downloads Python 3.12, the version `.python-version` names;
- Docker, for the containers of the build (planned).

## Getting started

```sh
uv sync --frozen
```

This installs the development tools into `.venv/`, from `uv.lock`. The command line needs none of them:

```sh
./model-build.sh --help
```

It prints the build's usage. It has no command yet: they come with the core.

## Commands

| Command | Effect |
|---|---|
| `./model-build.sh --help` | Prints the build's usage. |
| `uv sync --frozen` | Installs the dependencies from the lockfile, without changing it. |
| `uv run --locked pytest` | Runs the tests. |
| `uv run --locked ruff format && uv run --locked ruff check --fix` | Formats the code and applies safe lint fixes. |
| `scripts/check-commit.sh` | Checks the last commit, on what it touches, which for now is the whole gate. Must pass on every commit that reaches GitHub. |
| `scripts/check.sh` | Runs every check: `format` (`ruff format --check`), `lint` (`ruff check`), `types` (`mypy`), `imports` (`lint-imports`) and `tests` (`pytest`). `scripts/check.sh <step>…` runs some of them. Must pass on the last commit of a branch that reaches GitHub. |

## Documentation

[`docs/foundations.md`](docs/foundations.md) explains the design and its reasons: the stack, the architecture, the repository layout and the testing approach. The rules that follow from it are in [`.claude/rules/`](.claude/rules/), indexed in [`CLAUDE.md`](CLAUDE.md).

## License

[MIT](LICENSE).

## Credits

[`CREDITS.md`](CREDITS.md) lists the external work this project reuses.
