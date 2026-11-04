# mostt-model-builder

mostt-model-builder builds speech recognition models in ONNX from the weights their authors publish, and checks that each file it produces still transcribes as well as the original model.

Its first model is Parakeet Ultra, Moondream's post-training of NVIDIA's Parakeet TDT 0.6B v3. A build:

1. downloads the pinned sources, each checked by its size and its hash: the model's weights, and the FLEURS recordings it is measured on;
2. exports the model to ONNX, in float32;
3. converts it to float16, bfloat16 and 8-bit blocks;
4. checks every file, then transcribes FLEURS in French and in English with each variant, on the CPU and on the GPU;
5. accepts each variant against its reference, from its word error rate and its parity: the float32 export against the original model, each conversion against the float32 export.

The accepted files land in a directory laid out for Hugging Face, where the maintainer publishes them, and mostt, a local dictation project, installs them, pinned by hash.

The code of the models and of their frameworks is treated as untrusted: every step that runs it does so in a container without network, with a read-only file system and bounded resources. A build is reproducible: two builds of the same commit write the same bytes. The tooling is a core and one recipe per model family, and `parakeet-tdt` is the first.

## Requirements

- Python 3.12 on the host, as `python3`, which runs the command line on its standard library alone;
- curl, which downloads the sources;
- git, with which `images` reads the commit and its tree, and the GitHub CLI (`gh`), which the pull requests use;
- [uv](https://docs.astral.sh/uv/) 0.13.7, the version CI runs, which finds or downloads Python 3.12, the version `.python-version` names;
- Docker, with BuildKit, its default builder, for the containers of the build and their images;
- the NVIDIA Container Toolkit 1.17.8 or later, with its Container Device Interface (CDI) set up, `nvidia.com/gpu=all`, for the steps on the GPU;
- disk: about 27 GB of images, 4.5 GB of sources and 11 GB of work directory for a whole build.

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

A build works in a directory of its own, the work directory, which holds everything it fetches and writes. Create it outside your home directory and outside the clone, in mode 0700, and give it to every command with `--work`. A build follows the recipe of a model family, which every command but `lock` names with `--recipe`: the one recipe is `parakeet-tdt`, which builds Parakeet Ultra.

```sh
mkdir -m 0700 /data/model-build
./model-build.sh --work /data/model-build --recipe parakeet-tdt fetch
```

`fetch` downloads the pinned sources, about 640 MB of FLEURS audio and indexes and 3.8 GB of model files, Moondream's weights of Parakeet Ultra and NVIDIA's archive of Parakeet TDT 0.6B v3, into `sources/` in the work directory, over HTTPS from huggingface.co and the storage it redirects to. It checks each file against the size and the SHA-256 that `sources.toml` and the recipe's `sources.toml` pin, and keeps only a file that matches. Run it again after an interruption: it resumes a partial download, and checks again the files it already fetched.

Every other step runs in a confined container, from an image that `images` builds, with the network to pull its pinned bases from Docker Hub and the GitHub container registry, and the locked packages from PyPI, by their hashes, from a commit, so that the image holds the files of the tree alone, never one that git ignores: commit or stash every change first, untracked files included, and build the images again after each commit, since a step refuses an image built from another tree.

```sh
./model-build.sh --work /data/model-build --recipe parakeet-tdt images
./model-build.sh --work /data/model-build --recipe parakeet-tdt step fleurs-check
./model-build.sh --work /data/model-build --recipe parakeet-tdt step describe
./model-build.sh --work /data/model-build --recipe parakeet-tdt step export
./model-build.sh --work /data/model-build --recipe parakeet-tdt step convert
./model-build.sh --work /data/model-build --recipe parakeet-tdt step inspect
./model-build.sh --work /data/model-build --recipe parakeet-tdt step reference
./model-build.sh --work /data/model-build --recipe parakeet-tdt step transformers-reference
./model-build.sh --work /data/model-build --recipe parakeet-tdt step transcribe
```

`build` runs the same steps in that order, but `describe`:

```sh
./model-build.sh --work /data/model-build --recipe parakeet-tdt build
```

| Image | Size | Holds |
|---|---|---|
| `mostt-model-builder-core-bench` | 186 MB | The Python base and the core's code, for the FLEURS check |
| `mostt-model-builder-core-uv` | 261 MB | The same, with uv, for the lock of the images' libraries |
| `mostt-model-builder-core-onnx-inspect` | 401 MB | onnx, for the inspection of the candidate's graphs |
| `mostt-model-builder-core-onnx-convert` | 605 MB | onnx, onnxconverter-common, ml_dtypes and ONNX Runtime, for the conversions of the exported encoder |
| `mostt-model-builder-parakeet-tdt-export` | 10.7 GB | NeMo, torch and the ONNX exporters, with the core's and the recipe's code, for the steps that read NVIDIA's archive |
| `mostt-model-builder-parakeet-tdt-transformers-reference` | 9.83 GB | Transformers, torch and librosa, with the core's and the recipe's code, for the check of Moondream's weights against NeMo's reference |
| `mostt-model-builder-parakeet-tdt-transcribe-gpu` | 4.39 GB | onnx-asr on ONNX Runtime's CUDA provider, with CUDA and cuDNN, for the transcriptions on the GPU |
| `mostt-model-builder-parakeet-tdt-transcribe-cpu` | 467 MB | onnx-asr on ONNX Runtime, for the transcriptions on the CPU |

`step fleurs-check` checks, before any recipe reads them, that every recording of the FLEURS index is in its archive, once, in a format the bench reads, and reports their durations. `step describe` lists the tensors of Moondream's weights and of the model that NeMo builds from the configuration of NVIDIA's archive, without loading the archive's checkpoint. `step export` writes Parakeet Ultra's encoder, with its external data, its decoder and joint, and its vocabulary, in ONNX, in about a minute and up to 14 GiB of memory, and prints the largest difference between the exported encoder and NeMo's on 20 utterances of FLEURS fr, beside its tolerance. `step convert` converts the exported encoder to float16, bfloat16 and 8-bit blocks, one file each, and prints, for each, its nodes, its products in 8-bit blocks, the operators it kept in float32 and the weights it clamped into float16's range. `step inspect` copies the files of the candidate, laid out as they will be published, into `runs/<recipe>/<commit>/candidate/`, checks each one, and inspects their graphs. `step reference` transcribes FLEURS fr and en with NeMo, on the CPU, in about a quarter of an hour. `step transformers-reference` loads Moondream's weights in Transformers, prints the largest differences of its encoder and its joint with NeMo's on the utterances and the steps that `step reference` kept, beside their tolerances, and transcribes FLEURS fr, in about four minutes. `step transcribe` transcribes FLEURS fr and en with onnx-asr from the candidate, each variant on its devices, on the CPU in four containers at once and on the GPU through CDI, in a few minutes each. Each transcription prints, per language, its count of utterances, of unknown tokens, and its decoding time. A step writes its outputs under `runs/` in the work directory, and refuses to run again over them: `clean` removes them, through a container, since some of their files belong to the containers' user.

```sh
./model-build.sh --work /data/model-build --recipe parakeet-tdt clean describe
```

The libraries of the images are locked in `images/environment.lock`, which `lock` resolves in a container, with the network to reach PyPI, then checks and writes into the clone, printing the packages it adds, removes or changes in version. Review that change in the diff before you commit it. Their manifest, `images/environment.toml`, allows a package to be built from its source, which on your machine would run with your rights, so it never bears the name `pyproject.toml`, nor its lock the name `uv.lock`, outside a container, and no uv on your machine finds them: uv run in `images/` works on the host's project at the root. Never copy the manifest to a `pyproject.toml` or the lock to a `uv.lock`, or give either to uv on your machine (`.claude/rules/container-security.md` > The host, "No uv on the host finds the images' environment").

```sh
./model-build.sh --work /data/model-build lock
git diff images/environment.lock
```

The images take some room in Docker's storage. Remove them by name, never with a prune, which would remove other projects' images too:

```sh
docker image rm mostt-model-builder-core-bench mostt-model-builder-core-uv \
    mostt-model-builder-core-onnx-inspect mostt-model-builder-core-onnx-convert \
    mostt-model-builder-parakeet-tdt-export \
    mostt-model-builder-parakeet-tdt-transformers-reference \
    mostt-model-builder-parakeet-tdt-transcribe-gpu mostt-model-builder-parakeet-tdt-transcribe-cpu
```

`build` ends with `verify`, which measures the build on FLEURS from the outputs of its steps, on the host, once it has checked again that `inspect` accepted the candidate and that no file of the candidate changed since: it rates every run on each of its languages, without and with its unknown tokens, compares the runs that the recipe pairs, and holds each parity, each run and each comparison to the recipe's rules. It prints its figures, writes them to `runs/<recipe>/<commit>/verification.json` in the work directory, and fails, naming each broken rule, when the build does not pass. Run it alone after a step you ran again:

```sh
./model-build.sh --work /data/model-build --recipe parakeet-tdt verify
```

`docs/model-build.md` describes each step and the reasons behind its method.

## Commands

| Command | Effect |
|---|---|
| `./model-build.sh --help` | Prints the build's usage. |
| `./model-build.sh --work <dir> --recipe <name> fetch` | Downloads the pinned sources of the core and the recipe into the work directory, and checks each by its size and its SHA-256. |
| `./model-build.sh --work <dir> --recipe <name> images` | Builds the images of the core's containers and the recipe's from a clean clone, labeled with the tree of HEAD. |
| `./model-build.sh --work <dir> --recipe <name> step fleurs-check` | Checks the FLEURS bench in its container: every indexed recording in its archive, once, in a format the bench reads. |
| `./model-build.sh --work <dir> --recipe parakeet-tdt step describe` | Lists the tensors of Moondream's weights and of the model that NeMo builds from the configuration of NVIDIA's archive. |
| `./model-build.sh --work <dir> --recipe parakeet-tdt step export` | Exports Parakeet Ultra's FP32 encoder, decoder and joint to ONNX, with its vocabulary, and measures the encoder against NeMo's. |
| `./model-build.sh --work <dir> --recipe <name> step convert` | Converts the recipe's exported graph to its variants in a container, and checks the conversion's report. |
| `./model-build.sh --work <dir> --recipe <name> step inspect` | Copies the candidate's files into the candidate, checks their signatures and formats, and inspects their graphs in a container. |
| `./model-build.sh --work <dir> --recipe parakeet-tdt step reference` | Transcribes FLEURS fr and en with NeMo, on the CPU, as the reference of the export. |
| `./model-build.sh --work <dir> --recipe parakeet-tdt step transformers-reference` | Measures Moondream's weights in Transformers against NeMo's reference, and transcribes FLEURS fr with Transformers. |
| `./model-build.sh --work <dir> --recipe parakeet-tdt step transcribe` | Transcribes FLEURS fr and en with onnx-asr from the candidate, on the CPU and on the GPU. |
| `./model-build.sh --work <dir> --recipe <name> verify` | Measures every run on FLEURS, holds the parities, the runs and their comparisons to the recipe's rules, and writes `verification.json`. |
| `./model-build.sh --work <dir> --recipe <name> build` | Runs every step of the recipe's build, in its order, then `verify`. |
| `./model-build.sh --work <dir> --recipe <name> clean [<step>]` | Removes the outputs, the request and the log of a step at HEAD's commit, or of every step of the core and the recipe, and the verification's report. |
| `./model-build.sh --work <dir> lock` | Resolves the lock of the images' libraries in its container, checks it, prints its change and writes it to `images/environment.lock`. |
| `uv sync --frozen` | Installs the development tools from the lockfile, without changing it. |
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
