# mostt-model-builder

mostt-model-builder is the tooling of reproducible, confined builds of speech recognition models from their original weights: export, quantization and FLEURS verification.

- **Built:** the command line, `model-build.sh`, which runs on the host's Python and the standard library, isolated from the environment and the site packages, and refuses to run as root; `fetch`, which downloads the pinned sources of the FLEURS bench into a work directory, each checked by its size and its SHA-256; the confined containers in which every step runs, with their images, the lock of the images' libraries, and the check of the FLEURS bench; and `clean`.
- **Planned:** the rest of the build: in the core, the formats and their inspection, the measures of the FLEURS bench and their acceptance; and one recipe per model family, the first for Parakeet TDT, which builds Parakeet Ultra in ONNX. The checked files land in a candidate directory laid out for Hugging Face, where the maintainer publishes them, and mostt, the local dictation project, pins them by hash.

## Requirements

- Python 3.12 on the host, as `python3`, which runs the command line on its standard library alone;
- curl, which downloads the sources;
- git, and the GitHub CLI (`gh`), which the pull requests use;
- [uv](https://docs.astral.sh/uv/) 0.13.7, the version CI runs, which finds or downloads Python 3.12, the version `.python-version` names;
- Docker, with BuildKit, its default builder, for the containers of the build and their images;
- the NVIDIA Container Toolkit 1.17.8 or later, with its Container Device Interface (CDI) set up, for the steps on the GPU (planned);
- disk: about 450 MB for the core's images now, and about 27 GB of images and 30 GB of work directory for a whole build (planned).

## Getting started

```sh
uv sync --frozen
```

This installs the development tools into `.venv/`, from `uv.lock`. The command line needs none of them:

```sh
./model-build.sh --help
```

It prints the build's usage and its commands.

## Building the models

A build works in a directory of its own, the work directory, which holds everything it fetches and writes. Create it outside your home directory and outside the clone, in mode 0700, and give it to every command with `--work`:

```sh
mkdir -m 0700 /data/model-build
./model-build.sh --work /data/model-build fetch
```

`fetch` downloads the pinned sources, about 640 MB of FLEURS audio and indexes, into `sources/` in the work directory, over HTTPS from huggingface.co and the storage it redirects to. It checks each file against the size and the SHA-256 that `sources.toml` pins, and keeps only a file that matches. Run it again after an interruption: it resumes a partial download, and checks again the files it already fetched.

Every other step runs in a confined container, from an image that `images` builds, with the network to pull its pinned bases from Docker Hub and the GitHub container registry, from a commit: commit or stash every change first, untracked files included, and build the images again after each commit, since a step refuses an image built from another tree.

```sh
./model-build.sh --work /data/model-build images
./model-build.sh --work /data/model-build step fleurs-check
```

| Image | Size | Holds |
|---|---|---|
| `mostt-model-builder-core-bench` | 186 MB | The Python base and the core's code, for the FLEURS check |
| `mostt-model-builder-core-uv` | 261 MB | The same, with uv, for the lock of the images' libraries |

`step fleurs-check` checks, before any recipe reads them, that every recording of the FLEURS index is in its archive, once, in a format the bench reads, and reports their durations. A step writes its outputs under `runs/` in the work directory, and refuses to run again over them: `clean` removes them, through a container, since some of their files belong to the containers' user.

```sh
./model-build.sh --work /data/model-build clean fleurs-check
```

The libraries of the images are locked in `images/uv.lock`, which `lock` resolves in a container, with the network to reach PyPI, then checks and writes into the clone, printing the packages it adds, removes or changes in version. Review that change in the diff before you commit it. Never run uv in `images/` (`.claude/rules/container-security.md` > The host, "uv never runs on the images' environment on the host"): its manifest allows a package to be built from its source, which would then run on your machine with your rights, and its `required-version` stops every other version of uv, not yours when it is the images' version.

```sh
./model-build.sh --work /data/model-build lock
git diff images/uv.lock
```

The images take some room in Docker's storage. Remove them by name, never with a prune, which would remove other projects' images too:

```sh
docker image rm mostt-model-builder-core-bench mostt-model-builder-core-uv
```

The rest of the build comes with the core and its first recipe (planned).

`docs/model-build.md` describes each step and the reasons behind its method.

## Commands

| Command | Effect |
|---|---|
| `./model-build.sh --help` | Prints the build's usage. |
| `./model-build.sh --work <dir> fetch` | Downloads the pinned sources into the work directory, and checks each by its size and its SHA-256. |
| `./model-build.sh --work <dir> images` | Builds the images of the containers from a clean clone, labeled with the tree of HEAD. |
| `./model-build.sh --work <dir> step fleurs-check` | Checks the FLEURS bench in its container: every indexed recording in its archive, once, in a format the bench reads. |
| `./model-build.sh --work <dir> clean [<step>]` | Removes the outputs, the request and the log of a step at HEAD's commit, or of every step. |
| `./model-build.sh --work <dir> lock` | Resolves the lock of the images' libraries in its container, checks it, prints its change and writes it to `images/uv.lock`. |
| `uv sync --frozen` | Installs the development tools from the lockfile, without changing it. Run uv at the root only, never in `images/`. |
| `uv run --locked pytest` | Runs the tests. |
| `uv run --locked ruff format && uv run --locked ruff check --fix` | Formats the code and applies safe lint fixes. |
| `scripts/check-commit.sh` | Checks the last commit, on what it touches, which for now is the whole gate. Must pass on every commit that reaches GitHub. |
| `scripts/check.sh` | Runs every check: `format` (`ruff format --check`), `lint` (`ruff check`), `types` (`mypy`), `imports` (`lint-imports`) and `tests` (`pytest`). `scripts/check.sh <step>…` runs some of them. Must pass on the last commit of a branch that reaches GitHub. |

## Documentation

[`docs/foundations.md`](docs/foundations.md) explains the design and its reasons: the stack, the architecture, the repository layout and the testing approach. [`docs/model-build.md`](docs/model-build.md) describes the build part by part. The rules that follow from it are in [`.claude/rules/`](.claude/rules/), indexed in [`CLAUDE.md`](CLAUDE.md).

## License

[MIT](LICENSE).

## Credits

[`CREDITS.md`](CREDITS.md) lists the external work this project reuses.
