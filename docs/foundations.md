# mostt-model-builder — Foundations

This document explains the design of mostt-model-builder and the reasons behind it: the stack, the architecture, the repository layout and the testing approach. `docs/model-build.md` describes the build part by part. The rules that follow from it live in `.claude/rules/`, indexed in `CLAUDE.md`, and the procedures in `.claude/skills/`.

## The project

mostt-model-builder builds speech recognition models from their original weights, reproducibly and in confinement: it exports them, quantizes them, and measures them on FLEURS before any file is published. It is made of a core and of one recipe per model family. The core holds what every family shares: the pinned sources, the confined containers and their boundary with the host, the formats and their inspection, the FLEURS bench and its acceptance, the model card and the publication. A recipe holds what one family needs, behind a contract that forces no format, so that a family that is not built on NeMo, or not exported to ONNX, comes as a recipe of its own; the first one, for Parakeet TDT, builds Parakeet Ultra in ONNX. The code of the models and of their frameworks is untrusted, so it runs only in containers without network, which see none of the user's files or secrets. The checked files land in a candidate directory laid out as they will be published on Hugging Face, where the maintainer publishes them, and mostt, the local dictation project, pins the published files by hash.

These constraints drive the choices in this document:

| Constraint | Consequence |
|---|---|
| The code of the models, and their pickles, are untrusted | The libraries of the models and the files that run code when they load are loaded only in containers without network, privilege or any of the host's files but those a step names. The host's side runs on the standard library, and treats what a container writes as hostile input. |
| Nothing is installed on the host | The host's side runs on the host's Python 3.12; the development tools live in the project's own `.venv/`, and every library of a build in a container image. uv builds no package on the host. |
| The outputs are reproducible and verified before publication | Sources are pinned by revision, size and hash; images by digest and packages by hash; an export is compared with its reference framework and measured on FLEURS before it becomes a candidate. Each output is reused only under the key of what determines it, judged and hashed again before it is read, so that a change of the card's text, a language or a variant reruns only what it changes. |
| Publication is the maintainer's | The build ends at a candidate directory laid out for Hugging Face, and its card for review; only `publish` sends them, run by the maintainer at a terminal, with a token typed there, into a repository named on the command line and typed again; no agent publishes. |
| mostt pins the published files by hash, not this code | The contract with mostt is the published files: their layout, their names and their formats. This repository shares no code with mostt and can change freely while that contract holds. The rules on the published names, the structural checks of the `onnx` and `text` formats, and the signatures that mark code, a pickle or an archive are adapted from mostt's model store, so that every published file passes it, and this repository holds them as its own. |

## Technology stack

| Tool | Version | Role |
|---|---|---|
| Python | 3.12 | The language of the host's side and of the steps, pinned by `.python-version` and `requires-python` |
| uv | 0.13.7 in CI; the latest release on the host | Package manager and runner: installs the host's environment from `uv.lock`, and runs each tool of the gate |
| uv, in the images | 0.13.7 | Resolves the images' lock in its container, and installs each image's libraries from it |
| ruff | 0.17.3 | Formatter and linter |
| mypy | 2.4.1 | Type checker, in strict mode |
| pytest | 9.2.0 | Test runner |
| import-linter | 2.15 | Checks the imports between the layers (`lint-imports`) |
| curl | — | Downloads the pinned sources on the host, reads a repository's head before a publication, without a token, and reads and downloads a published revision to check it, over HTTPS alone |
| Docker | — | Runs each step of a build in a confined container, and builds their images with BuildKit |
| GitHub CLI (`gh`) | — | Opens and updates the pull requests |

Notes:

- **CI runs on GitHub Actions** (`.github/workflows/check.yml`):
  - `scripts/check.sh` runs on every pull request and every push to `main`;
  - actions are pinned to full commit SHAs, and the job may only read the repository;
  - `setup-uv` installs uv 0.13.7, the gate's pin, while the host's uv follows the latest release, and uv installs the Python that `.python-version` names;
  - no cache is saved or restored: the install takes seconds, and a restored cache would run in the gate unchecked.
- **`CLAUDE.md` and `.claude/` are the agent guidance** ([Agent guidance](#agent-guidance)).
- **The project has no build backend** (`package = false`): it is a tool run from its clone, never installed as a package, so each tool is given the path of its packages, `mypy_path` for mypy, `pythonpath` for pytest and `PYTHONPATH` for `lint-imports`, and the bootstrap appends them itself, run by path from the launcher, `model-build.sh`, and from every step.
- **No package is built on the host** (`no-build = true`): every development tool publishes wheels, and a package built from its source would run its build code with the user's rights.
- **`ruff format` checks the Python blocks of the Markdown documents too,** such as the example of `.claude/rules/tests.md`, so that an example reads as the code it shows.
- **Python 3.12** is the version of the base image of the build's images, so that the host's side and the steps run the same language.
- **Two environments hold the dependencies, one per place where code runs:** the host's, `pyproject.toml` and `uv.lock`, holds the development tools and builds nothing; the images', `images/environment.toml` and `images/environment.lock`, holds the libraries of every image, one extra per image, so that one lock pins one version of each library across a build. The images' environment allows a build (`no-build = false`), since a package that NeMo needs, antlr4-python3-runtime, is published as an sdist alone, and uv refuses it under `no-build = true` even with its metadata declared; that package is built in a build stage of the export image, without network. A build allowed on the host would run that code with the user's rights, so the manifest never bears the name by which uv finds a project, `pyproject.toml`, outside a container, nor its lock the name `uv.lock`, which uv's `--frozen` uses even without a project beside it: a uv on the host, of any version, finds the host's project instead: a name holds against every version, where a pin of the version lets the host's uv through whenever it is the images'. Inside a container, `./model-build.sh lock` resolves its lock and each image's build installs from it, with `--frozen --no-build`. Its uv, 0.13.7, is pinned by its image's digest and by `required-version`, which keeps the lock's layout tied to the images' uv, so that it changes only through a change that names a new uv.
- **The gate reads the images' lock as text:** it checks both locks, every package from PyPI by hash with a wheel that installs here, and checks that the images' lock records what its manifest declares, which stands in for the `uv lock --check` that no uv on the host can run, since none finds the manifest.
- **Direct dependencies are pinned exactly,** since a new exporter or a new framework can change the graph a build writes: a new version comes only through a change that names it, and the build that follows checks it.

## Architecture

### Principles

The project follows Clean Architecture: the rules live in a pure domain at the center, and each technology stays at the edge, behind an interface that an inner layer owns. `.claude/rules/architecture.md` states the rules, and tooling enforces them wherever it can ([Enforcement](#enforcement)).

- **The domain is pure,** so it runs anywhere the language runs, and its tests need no I/O, no network and no framework.
- **The application reaches the outside world only through its ports,** declared in its own vocabulary, so its tests run it on fakes, and each technology stays one implementation among possible others.
- **Composition roots wire by hand,** so the whole graph sits in one place and the types check it. A DI container would resolve it at run time, by tokens.
- **What is stored or shared is plain data,** so it saves, travels and replays as it is.
- **SOLID applies to modules, functions and types,** so a new case is added beside the others, without rewriting them.
- **A model family is a recipe:** what a family needs, its sources, its export, its variants and its transcriber, lives in its recipe, and the core judges what a recipe declares. A new family then adds a recipe without changing the core, and a recipe cannot widen the confinement that the core imposes.
- **The host trusts no container:** the host's side never imports a library of the models, and reads what a container writes only through its own strict parsers, since that code is the part of a build that no one has reviewed.

### Layers and modules

| Layer | Module | Responsibility | Depends on |
|---|---|---|---|
| Composition roots | `core` | The project's package, `mostt_model_builder`; its command line on the host, `host_main.py`, which reads the terminal's input and the hardware's facts in `/proc` and wires the adapters, and the recipes; and its bootstrap, `bootstrap.py`, which puts the packages on the path and runs an entry module, on the host and in every step | every layer, and each recipe's `recipe.py` and domain |
| Composition roots | `steps` | The entry module of each of the core's container steps, which runs in its container on the libraries of its image: the FLEURS check, the cleaning of a step's outputs and the resolution of the images' lock, on the standard library, the conversions of a recipe's exported graph, on onnx, onnxconverter-common and ONNX Runtime, the inspection of the candidate's graphs, on onnx, and the publication of the candidate and its card, on huggingface_hub, with the reader of FLEURS's audio, the lookup of a bound source, the output modes and the control group's peaks that every step shares, a recipe's included | `domain` |
| Adapters | `adapters` | curl, Docker, git, the terminal and the files of the work directory | `application`, `domain` |
| Application | `application` | Opens the work directory and fetches the sources; builds the images, resolves the build in its order, reusing each entry of the cache whose key is unchanged once judged and hashed against its record and running the others, runs a step of the core or of a recipe, checks the FLEURS bench, cleans the entries of the current build or of a step, keyed from their records, prunes the entries that the current build does not use, and resolves the images' lock; runs the shards of a run and reads their lines through the run's reader; converts a recipe's variants and reads the conversion's report; assembles the candidate and inspects it; verifies the build: reads every run and parity, measures it, hashes the candidate and writes its report; prepares the publication: verifies the build again from a clean clone and writes its model card for review; publishes the candidate and its reviewed card; checks a published revision against the build | `domain` |
| Domain | `domain` | The rules: the pins of the sources, the work directory and its layout, and the file names; the step table and its images, the judgment of the mounts and of the files that containers write, the parsers of what they write and the bound on the nesting of a JSON document they read, the label of an image, the container user's accounts and the locks' check; the names a step shares with the host, in modules that hold no rule of the host's, so that a step's code holds only what it runs: the paths in a container, the devices, the bounds of a step's limits, the mode of what a step writes, the names and bounds of its outputs, and the request of a shard; the languages of the FLEURS bench and their English names, its index and its recordings; the recipe contract, the model it describes, its runs and the rows they derive; the formats, their structural checks, the signatures and the external data; the conversions' request, their decisions of precision and their report, the inspection's report, the walk of an ONNX model's graphs, the graph rules and the placement of a run's nodes; the key of a row's outputs, the digest of what determines them, from the import closure of its entry module in HEAD's tree, the closure of its image's extra in the images' lock, its inputs by content, its parameters and the hardware's facts, read from `/proc`, and the key of the candidate; the units of a build in its order, with what each draws on, and the record of an entry of the cache; the hypotheses of a run and its coverage of the bench; the parity reports; the normalization, the word error rate, the paired comparisons and the rules of acceptance; the verification and its report; the model card, the summary of each variant and the libraries that measured it, the listing of a published revision, the maintainer's token and its redaction, the request and the report of a publication and the files a repository may hold | — |
| Domain, composition roots | `parakeet-tdt` | The recipe of the Parakeet TDT family, which builds Parakeet Ultra in ONNX: its contract, `recipe.py`; its sources, its export image, the description of its weights and its export to ONNX with NeMo, with the preparation of NeMo's configuration, the names of its weights and of its pointwise convolutions, and the reader of the name lists in its domain; NeMo's reference transcription and onnx-asr's, on the CPU and on the GPU, with the vocabulary, the rendering of tokens, the readers of its runs and onnx-asr's layout of the candidate in its domain; the dump of NeMo's reference tensors in a step of its own, and its check of Moondream's weights in Transformers against them, with the layout of that dump and the ids it keeps from Transformers' generated sequences in its domain; its variants, onnx-asr's suffix of each and their conversions, the description of the model it publishes, and the names and counts of CPUs that its steps share with its contract, in its domain | `domain`, `steps` |

## Repository layout

### Repository tree

```
mostt-model-builder/
├── .claude/                     # Claude Code: settings, rules, skills and the expert reviewers
├── .github/
│   ├── pull_request_template.md # The body of every pull request, with its checklist
│   └── workflows/               # CI: the gate on every pull request and every push to main
├── docs/
│   ├── foundations.md           # This document
│   └── model-build.md           # The build, part by part, and the published layout
├── images/
│   ├── environment.toml         # The images' environment: their libraries, one extra per image
│   ├── environment.lock         # Its lockfile, which only ./model-build.sh lock writes
│   └── <image>/                 # The Containerfile of each of the core's images
├── recipes/
│   └── <family>/                # A recipe, one per model family: parakeet-tdt
├── scripts/
│   ├── check.sh                 # The gate
│   └── check-commit.sh          # The check of the last commit
├── src/
│   └── mostt_model_builder/     # The core
├── tests/                       # The core's tests
├── .python-version              # The Python that uv runs
├── CLAUDE.md                    # The session manual for AI agents
├── CREDITS.md                   # External work reused, with its terms
├── LICENSE                      # MIT
├── README.md
├── model-build.sh               # The launcher: runs the bootstrap on the host's Python, isolated
├── pyproject.toml               # The host's environment: the development tools, and their configuration
├── sources.toml                 # The core's pinned sources: the FLEURS test sets
└── uv.lock                      # The host's lockfile
```

### Module anatomy

The project is a single module, and its packages stand for the modules of [Layers and modules](#layers-and-modules):

```
src/mostt_model_builder/
├── __init__.py                  # The package
├── bootstrap.py                 # Run by path: puts the packages on the path, runs an entry module
├── domain/                      # The pure rules, a directory per concern, one concept per file
├── application/                 # The use cases and their ports
├── adapters/                    # One file or directory per technology
├── steps/                       # The entry modules of the core's container steps
└── host_main.py                 # The command line on the host
recipes/<family>/                # a recipe
├── images/                      # its Containerfiles
├── sources.toml                 # its pinned sources
├── src/mostt_recipe_<family>/
│   ├── domain/                  # its own rules
│   ├── recipe.py                # the plain data it gives the core
│   └── steps/                   # the entry modules of its container steps
└── tests/
```

The domain groups its files in a directory per concern, such as `domain/sources/` for the pins and their URLs or `domain/containers/` for the work directory, since it holds the rules of every part of the build. These directories are not modules: the domain keeps its one row, and its files still import one another without a cycle.

A recipe keeps its own package, apart from the core's, so that the import contracts can tell the two apart and keep the core from importing a recipe, and one recipe from importing another. The core's code and the recipes' share the host's environment for their tools, and the libraries of every image share the images' environment, one manifest and one lock, so that a build pins one version of each library.

### Enforcement

Each row names a rule, the rules file that states it, and what enforces it.

| Rule | Enforced by |
|---|---|
| Dependencies point inward (`.claude/rules/architecture.md` > Layers) | `lint-imports`, from the layers contract of `pyproject.toml` and the contract of each layer |
| The domain holds the rules and is pure (`.claude/rules/architecture.md` > Responsibilities) | `lint-imports`, which keeps out of the domain the modules of the standard library that reach the outside world; review, the rest |
| Formatted by `ruff format` and linted by `ruff check` (`.claude/rules/conventions.md` > Python) | `ruff format --check` and `ruff check`, in the gate |
| Typed strictly (`.claude/rules/conventions.md` > Python) | `mypy`, strict with `warn_unreachable`, in the gate |
| Python 3.12 syntax (`.claude/rules/conventions.md` > Python) | `ruff`'s `UP` rules, with `target-version = "py312"` |
| The host runs on its own Python and the standard library, isolated (`.claude/rules/container-security.md` > The host) | `tests/test_mostt_model_builder.py`, which imports the command line under `-I -S -B`; `tests/test_model_build_launcher.py`, which checks that the launcher runs the bootstrap under `-I -S -B`; `tests/test_bootstrap.py`, which checks the paths the bootstrap appends; `lint-imports`, which keeps the bootstrap's imports out of the packages |
| Root is refused (`.claude/rules/container-security.md` > The host) | `tests/test_bootstrap.py`, on each of the six ids |
| A source is fetched over HTTPS alone, and kept only once it matches its pin (`.claude/rules/container-security.md` > The host) | `tests/domain/sources/test_source_pins.py`, the refusals of the pins; `tests/application/test_fetch_sources.py`, which keeps a download only once it matches; `tests/adapters/test_curl_downloader.py`, curl's options |
| A source's name never marks code, a pickle or an archive, unless its pin says it runs code (`.claude/rules/container-security.md` > The host) | `tests/domain/sources/test_source_pins.py`, on each refused suffix and on the mark; `tests/test_recipe_listing.py`, which holds every `sources.toml` to its one marked file |
| The work directory is the user's own, outside the home directory and the clone (`.claude/rules/container-security.md` > The host) | `tests/domain/containers/test_work_directory.py` and `tests/application/test_open_work_directory.py`, on each refusal |
| uv never builds a package on the host (`.claude/rules/container-security.md` > The host) | `no-build = true` in the host's `[tool.uv]`, which `tests/test_project_manifests.py` holds, with the host's empty dependencies and its exact pins, and the tracked files named `pyproject.toml`, `uv.toml` or `uv.lock` to the root's; the locks' check of `tests/test_lockfiles.py`, which refuses a package without a wheel that installs here |
| No uv on the host finds the images' environment (`.claude/rules/container-security.md` > The host) | `tests/test_project_manifests.py`, on the names of the tracked files; `tests/test_recipe_listing.py`, which holds each image's copies of the manifest, as `pyproject.toml`, and of the lock, as `uv.lock`, to their accepted forms; `tests/steps/test_lock_resolution.py`, on the copies that the `lock` step gives uv and the name of the lock it writes; review, the commands |
| The images' lock is resolved only by `./model-build.sh lock`, and installed only by an image's build (`.claude/rules/configuration.md` > Dependencies) | `required-version` in `images/environment.toml`, held by `tests/test_project_manifests.py` to the uv of the `uv` image, with the manifest's fixed shape, which keeps the lock's layout tied to the images' uv; `tests/test_recipe_listing.py`, which holds each image's install to `--frozen --no-build` |
| Both locks pass the locks' check (`.claude/rules/configuration.md` > Dependencies) | `tests/test_lockfiles.py`, on `uv.lock` and `images/environment.lock`, and the freshness of `images/environment.lock` against its manifest; `lock` runs both checks on the lock it resolves |
| A container step's Python runs under `-I -B` (`.claude/rules/container-security.md` > The host) | `tests/domain/containers/test_container_steps.py` and `tests/adapters/test_docker_engine.py`, on every row of the step table |
| A container that loads that code has no network, no privilege and none of the host's secrets (`.claude/rules/container-security.md` > Confinement) | `tests/domain/containers/test_container_steps.py`, which judges every row against its bounds, and `tests/adapters/test_docker_engine.py`, every option of every row, the core's and each recipe's |
| A file that runs code when it loads reaches only the one image declared to read it (`.claude/rules/container-security.md` > Confinement) | `tests/domain/contract/test_recipe_contract.py`, which refuses it bound in any other image; `tests/domain/containers/test_container_steps.py`, on every row; the recipe's own tests, on the members of the `.nemo` it reads |
| A recipe cannot widen the confinement (`.claude/rules/container-security.md` > Confinement) | `tests/domain/contract/test_recipe_contract.py`, the refusal table of the contract, which the core judges when it loads a recipe; `tests/test_host_main.py`, the host's check that the card template is a regular file, never a link; `tests/test_recipe_listing.py`, each recipe's import contracts and Containerfiles |
| What a container prints or writes is data, never an instruction (`.claude/rules/container-security.md` > The host) | `tests/domain/containers/test_container_files.py`, the refusals of the files written by containers and of the mounts; the refusal tests of each parser; `tests/domain/containers/test_printable_output.py`, the filter over every byte |
| An image is built from a commit, and runs only at that commit (`.claude/rules/container-security.md` > Images and outputs) | `tests/domain/containers/test_provenance.py` and `tests/application/test_build_images.py`; `tests/adapters/test_git_tree.py`, which archives the files of the tree alone, and `tests/adapters/test_docker_engine.py`, which gives that archive to `docker build` as its context; `tests/test_recipe_listing.py`, which holds every Containerfile to its table of accepted forms |
| A file leaves the work directory only once it is verified (`.claude/rules/container-security.md` > Images and outputs) | `tests/application/test_verify_build.py`, which reads every output through its judgment, the inspection's report and every file of the candidate again, and the index of the bench through its pin; `tests/adapters/test_work_files.py`, the hash of each file of the candidate that `verify` accepts; `tests/domain/containers/test_container_files.py`, the judgment of a copy of the candidate; `tests/domain/acceptance/test_acceptance.py`, each rule on both sides of its bound |
| A step's code imports only what it runs (`.claude/rules/container-security.md` > Images and outputs) | `lint-imports`, from the contract of `pyproject.toml` that forbids to the core's steps and each recipe's, indirect imports included, the modules of the host's side, of the build's plan and of the card |
| An output is reused only under the key of what determines it, and judged again before it is read (`.claude/rules/container-security.md` > Images and outputs) | `tests/domain/cache/test_output_keys.py`, a change of each component against the key, and the host's paths, the image's ID and the ids against none; `tests/domain/cache/test_import_closure.py`, against `modulefinder` on every entry module; `tests/domain/cache/test_output_records.py`, the strict reading of a record; `tests/application/test_resolve_build.py`, the rows that a change of each component reruns, and the refusal of an output changed after its record, of a reused report that today's parser refuses and of a reused candidate that the checks of its copy refuse; `recipes/parakeet-tdt/tests/test_recipe_keys.py`, on the real recipe |
| The host never imports a container step, and the libraries of the models load only in the steps that need them (`.claude/rules/container-security.md` > The host) | `lint-imports`, from the contracts of `steps` and of the libraries of the models; `tests/test_mostt_model_builder.py`, which imports each step of the standard library under `-I -S -B` |
| Markers are strict and warnings are errors (`.claude/rules/tests.md` > pytest) | `--strict-markers`, `--strict-config` and `filterwarnings = ["error"]` in `pyproject.toml` |
| Only the maintainer publishes (`.claude/rules/container-security.md` > Images and outputs) | `publish`'s check of its terminal, before the work directory opens, `tests/test_host_main.py`; the typed confirmation of the repository and the typed token, `tests/application/test_publish_candidate.py` and `tests/adapters/test_terminal_prompt.py` |
| No agent publishes (`CLAUDE.md` > Machine boundaries) | `permissions.deny` in `.claude/settings.json`, which refuses an agent any Bash command whose text holds `model-build.sh`, or the module of the command line, then later ` publish` as a word: it also refuses a search whose text holds that pattern, which an agent runs through its Grep tool instead, and it is no boundary around the program itself: a defense in depth, which matches only a literal ` publish`, behind `publish`'s check of its terminal and the token typed there; `tests/test_claude_settings.py` holds the rules and what they match |
| A commit footer holds only what tools read (`CLAUDE.md` > Workflow) | `attribution` in `.claude/settings.json`, which keeps Claude Code's own commit trailers and pull request line out |
| Every rule no tool can check: architecture, tests as specification… | `/expert-review` ([Code review](#code-review)), before every pull request |

## Conventions

### Code conventions

The conventions serve one aim: the code explains itself, so that it needs no comments, and a reviewer can check it against the domain. `.claude/rules/conventions.md` states them.

- **Names carry the meaning** that comments may not carry, in the vocabulary of the domain. A unit in a name, spelled out or in its standard symbol, makes a unit error visible where it is written.
- **Comments only say why.** The names say what the code does, and the tests state the business rules ([Tests document the rules](#tests-document-the-rules)).
- **The project's own files are not checked as outside input is,** since such a check costs code and tests and guards against nothing but its own bugs, which a loud failure shows sooner. A file that a download or a container wrote is never one of them, even once stored, since the code of a model runs in the containers. The pins of the sources and the fields of a recipe's contract that reach Docker or the file system are the exception: the build's confinement is what it guarantees whatever a recipe holds, so the core judges what a recipe asks of it.
- **The units are written as their symbols,** `ms`, `s`, `Hz`, `KiB`, `MiB` and `GiB`, the units in which Docker, the kernel's cgroups and the measurements of a build give their figures, while bytes are spelled out, since a byte has no symbol that a name could not misread.
- **Number formats are spelled out in names,** `float16` or `eight_bit_blocks`, while the published names keep the spelling their readers expect, such as onnx-asr's `fp16` and `int8` suffixes: the code reads the same everywhere, and the files keep the contract with their readers.
- **Typing is strict,** and a value read from outside, such as a container's report, has no type until it is checked, since the host must trust nothing that a container wrote.

### Git

`CLAUDE.md` > Workflow says how a session works, and the `develop`, `pull-request` and `writing-commits` skills hold the procedures. The reasons behind them:

- **A worktree per piece of work** keeps parallel sessions from changing each other's files, and keeps the main checkout on `main`, where the user's running work never sees work in progress.
- **A stacked branch** costs only a rebase of its own commits when its base is rewritten, so no work waits for a merge.
- **A history rewritten before it is pushed,** into atomic commits whose messages say why, keeps the reasons that a diff cannot show. Each commit passing the checks of what it touches keeps `main` a safe start for any work, and the whole gate, which the CI runs on the pull request, keeps every part working with the others: every pull request must pass it there anyway, so it runs once per push rather than on each machine at each rewrite of the branch. The slower tests would add their run to every commit for what the next commit may still complete, unless they guard what a commit could harm once checked out, such as the safety checks of a deployment, since a commit that a `git bisect` checks out runs as it is. While the whole gate runs in seconds, the check of a commit is the whole gate.
- **A header that says what the change achieves, under the scope it belongs to,** lets a reader find a change in `git log --oneline` without opening it: the diff already lists the files.
- **A footer that holds only what tools read** lets release notes and the issue tracker trust it. A trailer naming the tool that typed a change would add a line to every message and say nothing of its reason.
- **"Rebase and merge"** lands each commit as written, after the review ([Code review](#code-review)) and CI ([Technology stack](#technology-stack)).

### Plans

`CLAUDE.md` > Workflow, "Plan before writing code" says when a plan is due, and the `develop` skill how it is made. The reasons:

- **A mistake in a plan costs every step after it:** the code built on it, its review and its fixes. A plan is short, so thorough reasoning costs less there than anywhere else: the planner and the critic of large work run on Opus at `xhigh`, the critic of normal work at `high`, since a normal plan adds no module, no public interface and no decision that changes the project, and the implementation that follows a checked plan at `high`.
- **A plan chooses the method before the code.** A test that covers the whole input space, a simpler design or a library costs nothing to choose in a plan; chosen after the code, it costs a revised plan, the code written again and another full review.
- **The work is sized before it is planned:** a sync made of mechanical commits and a new module do not need the same planning, and a planner with three rounds of a critic at `xhigh` can cost a third of a session that a brief would serve. The plan states its size and the reason, so that the critic and the user can question it.
- **The decisions are settled with the user before the plan is written:** a decision that the user meets only in a written plan sends the plan back to planning and to its critic, while a question asked first, with its recommended answer, costs a line. The agents find the facts, since a question that the code answers spends the user's attention on nothing. When the user does not follow the work, the pull request lists the answers taken alone, so that the user checks them once.
- **The session that held the interview writes the plan of normal work:** it holds the decisions first-hand, and a planner would read the code again to rebuild them. The planner stays for large work, whose design needs a context of its own.
- **A plan holds decisions, not code:** the goal, the design choices and the alternatives they reject, where the tests go, the commits, the documents and the risks. File paths and code go stale as soon as the work starts, the implementation finds them anyway, and a plan that describes the code line by line runs to hundreds of lines that the planner, the critic and the implementation all pay for.
- **A mock of an output a person reads comes before its plan:** a critic checks a plan against the code and the rules, but cannot tell whether a reader will like the output. A mock costs a few lines to write and to change, while an output that the user rejects once it is built costs a revised plan, a new implementation and another review.
- **A critic that did not write the plan checks it,** against the code and the rules, as an expert reviews code it did not write: an assumption about the code that no one checked is the cheapest mistake to catch in a plan.
- **The critic names a simpler alternative, with its size beside the plan's,** since nothing else in the loop asks whether a much smaller design would do: the critic checks the design it is given, and reviews then harden it. A plan sized next to its alternative shows a design out of proportion with its goal before any code is written.
- **The code follows the plan, or the plan changes first,** since a design changed while it is built escapes the critic.
- **The planner writes its plan file and reports in a few lines:** a plan returned in a report is written twice, by the planner and again by the session, and then weighs on every later turn of the session, which needs its summary and its open questions, and reads the rest of the plan only where it needs it, such as its decisions.
- **The critic checks the whole plan, and its mock, at each round,** since a plan of decisions is short: a check of a revision from its diff alone would miss the parts of the plan that a changed mock settles and the diff does not touch, such as a test that still asserts the old order of the output.
- **Within one planning, the planner answers the critic itself,** since it already holds the code the plan rests on, in a context far smaller than an implementer's: a new planner would read it all again.
- **The smallest changes go without a plan,** such as a bug fix inside one function: even a brief would cost more than the mistakes it could catch there.
- **The plan lives outside the repository,** since the repository holds only the current state, and a plan describes work to come. Each plan has one writer, the session that owns its branch, itself or through its planner, so plans written in parallel never collide.

### Agent guidance

Agents work from four kinds of documents, each with one job, so that a statement has one home and never drifts from a copy. `.claude/rules/documents.md` > Where a statement lives says what goes where. The reasons:

- **`CLAUDE.md` is the session manual** because Claude Code loads it whole into every session: each line costs every session, so it holds only what every session needs.
- **The rules live in `.claude/rules/`** because a file with `paths` loads only with the files it governs, and a rule stated there in full, with its reason, needs no second document. A load by `paths` can be missed, by a subagent in its own worktree or after a compaction, which is why each file has its line in `CLAUDE.md`.
- **The rules files follow concerns, not directories,** because a concern often spans several directories, where a `CLAUDE.md` per directory would scatter its rules.
- **Skills hold procedures** because a procedure matters only while its task runs, and a skill loads only then.
- **This document holds the design and its reasons,** and states no rule, so that a rule changes in one place.
- **An expert reviewer reads at least the rules files of its domain,** one hop away, so that it never chases a rule across documents. It starts without the session's `CLAUDE.md` files, which would cost it tokens on every turn and could steer it: the rules that apply are the checkout's.
- **Each agent runs on the model and at the effort its task needs,** because every turn of every run pays for both, and a review multiplies the runs: Opus where a miss costs most or where code is written, at `xhigh` only for large plans ([Plans](#plans)), Sonnet where written rules guide the check, Haiku for search. A low effort saves Haiku and Sonnet little and costs them much quality, so they never run at it.
- **The history of a decision lives outside the repository,** with the project-management material, so the documents here describe only the present, with its reasons.

## Testing

### Tests document the rules

The tests are the specification of the business rules. To learn how a part of the system behaves, read its tests: each rule reads as one sentence, from the component through the operation and the situations to the test's name, in the vocabulary of the domain. That is why the code carries no comment that states a rule. Every test file follows one structure, so that a reader, and an agent, finds the arrangement, the act and the assertions in the same place in every file. `.claude/rules/tests.md` says how tests are written.

### Test levels

- **The unit tests run on every commit,** since they need no network, no Docker and no model, and the whole gate runs in seconds. They hold the rules of the build, the confinement included: what a container may mount, how the host judges what it wrote, and what a recipe may declare, all checked on plain data and fakes.
- **A build on the real containers and models runs by hand,** `./model-build.sh build`, since it takes tens of minutes, tens of gigabytes and a GPU. Its results, the parity with the reference framework and the error rates on FLEURS, go in the pull request that changes what it builds.

### Budgets

No budget is set yet. The first one comes with the benchmark that measures it.

### Code review

Every branch goes through `/expert-review` before its pull request. It is a project skill in `.claude/skills/expert-review/`, named so that it never shadows the built-in `/code-review`.

- **It runs in its own subagent**, which never saw the code being written, so the review stays independent of the session that wrote the code, and that session keeps its context: the experts report to the review, which waits for all of them.
- **The review runs in the background.** In the foreground, an interactive review can hand back when its first turn ends, while its triage and its experts still run, and their answers then reach the session instead. In the background, Claude Code keeps a fork running until it hands back its report, and delivers its agents' answers to it.
- **One expert per domain, in parallel.** Correctness reviews every diff, and conventions every diff that holds code or configuration. Architecture, tests, performance, security, container security and documentation join when the diff touches their domain. Each expert is an agent of `.claude/agents/`, on Opus for bugs, security, the confinement of the containers, architecture, performance and tests, and on Sonnet for conventions and documentation, which written rules guide. No expert can run a command, so that no reviewed text can make it act.
- **The confinement has an expert of its own,** since a build runs code that no one reviewed, and a mistake at the boundary between the host and the containers would hand that code the user's machine.
- **A triage on Sonnet weighs the experts that the paths pick,** by what the diff holds, before any of them runs: an expert pays a full reading of its rules, the diff and the code even when it has nothing to say, and the triage costs a fraction of that. It keeps an expert when unsure, and never drops correctness, nor security on its paths, nor documentation on an added file, the one expert that reads a file's content for what the repository must not hold, nor narrows what they read, since a missed expert costs more than a needless one, and a triage that reads the diff can be steered by it.
- **The fixes of blocking findings are checked against the findings they answer,** by their experts, by correctness, by security on its paths, since a fix can weaken a guard as well as add one, and by the experts whose concerns the fixes newly touch, such as tests for a new test, rather than by every expert whose paths they touch: a fix of three files can touch the paths of every domain, and only the experts it concerns have something to check.
- **The orchestrator confirms every blocking and should-fix finding** before reporting it, and lists the findings it dropped with their counter-evidence, so the report makes no claim it has not checked. It also holds every suggested fix to the project's rules, since an expert reads only those of its domain.
- **A target other than the current branch is checked out outside the repository**, so a reviewed `CLAUDE.md` never loads as an instruction.
- **Reviews can run at the same time, from the same checkout:** each one keeps its scope in a temporary directory of its own.
- **A branch is reviewed once.** A fix brings code of its own to review, and two runs of the same review judge differently, so each new review of the fixes finds something new, and reviews repeated until none finds anything never end. Only the fixes of blocking findings are checked again, since a defect there costs the most, and that check starts no further round. A finding that calls for a change of design goes back to the plan, where the method is cheapest to change.
- **A new agent fixes the findings,** from a brief: the agent that wrote the branch, resumed, re-reads its whole, grown context on every turn, which costs more than the review it answers.
- **The report says what the review cost,** agent by agent, so that it shows which agents a review spends on, such as an expert that finds nothing on a large diff, and the triage and the experts can be tuned.
- **The report ends the review.** It suggests a fix for each finding but applies none: the session that owns the branch answers each one or has a `fixer` agent fix it, and the user decides at the merge.

## References

- uv: https://docs.astral.sh/uv/
- ruff: https://docs.astral.sh/ruff/
- mypy: https://mypy.readthedocs.io/
- pytest: https://docs.pytest.org/
- import-linter: https://import-linter.readthedocs.io/
- FLEURS, the multilingual speech benchmark that measures a build: https://huggingface.co/datasets/google/fleurs
- Parakeet TDT 0.6B v3, NVIDIA's model of the first recipe: https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3
- Parakeet Ultra, Moondream's post-training of it, the first model built: https://huggingface.co/moondream/parakeet-ultra
- NeMo, NVIDIA's toolkit, which builds Parakeet TDT's model and exports it: https://github.com/NVIDIA-NeMo/NeMo
- NVIDIA-NeMo/Speech #16052, which runs the Conformer's convolutions of kernel size 1 as matrix products, the form the export ports: https://github.com/NVIDIA-NeMo/Speech/pull/16052
- onnx-asr, the runtime whose layout the candidate follows and which transcribes the bench: https://github.com/istupakov/onnx-asr
- parakeet-rs, whose export of Parakeet TDT and Parakeet Ultra to ONNX the export ports: https://github.com/altunenes/parakeet-rs
- Creative Commons Attribution 4.0 International, the license of the published files and of the works they adapt, whose section 3(a)(1) the card's attribution follows: https://creativecommons.org/licenses/by/4.0/legalcode.en
- The Hub's model cards, whose metadata the card's front matter writes: https://huggingface.co/docs/hub/model-cards
- M. Bisani and H. Ney, "Bootstrap estimates for confidence intervals in ASR performance evaluation", ICASSP 2004, the paired bootstrap of the comparisons: https://doi.org/10.1109/ICASSP.2004.1326009
- mattpocock/skills, whose plans of decisions rather than code and whose interview of the user before planning [Plans](#plans) adapts: https://github.com/mattpocock/skills
