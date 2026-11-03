# mostt-model-builder — Foundations

This document explains the design of mostt-model-builder and the reasons behind it: the stack, the architecture, the repository layout and the testing approach. The rules that follow from it live in `.claude/rules/`, indexed in `CLAUDE.md`, and the procedures in `.claude/skills/`.

## The project

mostt-model-builder builds speech recognition models from their original weights, reproducibly and in confinement: it exports them, quantizes them, and measures them on FLEURS before any file is published. It is made of a core and of one recipe per model family (planned). The core holds what every family shares: the pinned sources, the confined containers and their boundary with the host, the formats and their inspection, the FLEURS bench and its acceptance. A recipe holds what one family needs, behind a contract that forces no format, so that a family that is not built on NeMo, or not exported to ONNX, comes as a recipe of its own; the first one, for Parakeet TDT, builds Parakeet Ultra in ONNX. The code of the models and of their frameworks is untrusted, so it runs only in containers without network, which see none of the user's files or secrets. The checked files land in a candidate directory laid out as they will be published on Hugging Face, where the maintainer publishes them, and mostt, the local dictation project, pins the published files by hash.

These constraints drive the choices in this document:

| Constraint | Consequence |
|---|---|
| The code of the models, and their pickles, are untrusted | The libraries of the models and the files that run code when they load are loaded only in containers without network, privilege or any of the host's files but those a step names (planned). The host's side runs on the standard library, and treats what a container writes as hostile input. |
| Nothing is installed on the host | The host's side runs on the host's Python 3.12; the development tools live in the project's own `.venv/`, and every library of a build in a container image (planned). uv builds no package on the host. |
| The outputs are reproducible and verified before publication | Sources are pinned by revision, size and hash, images by digest and packages by hash (planned); an export is compared with its reference framework and measured on FLEURS before it becomes a candidate (planned). |
| Publication is the maintainer's | The build ends at a candidate directory laid out for Hugging Face (planned); no command of the repository publishes on its own, and no agent publishes. |
| mostt pins the published files by hash, not this code | The contract with mostt is the published files: their layout, their names and their formats. This repository shares no code with mostt and can change freely while that contract holds. |

## Technology stack

| Tool | Version | Role |
|---|---|---|
| Python | 3.12 | The language of the host's side and of the steps, pinned by `.python-version` and `requires-python` |
| uv | 0.13.6 | Package manager and runner: installs the environment from `uv.lock`, and runs each tool of the gate |
| ruff | 0.17.3 | Formatter and linter |
| mypy | 2.4.1 | Type checker, in strict mode |
| pytest | 9.2.0 | Test runner |
| import-linter | 2.15 | Checks the imports between the layers (`lint-imports`) |
| Docker | — | Runs each step of a build in a confined container, and builds their images (planned) |
| GitHub CLI (`gh`) | — | Opens and updates the pull requests |

Notes:

- **CI runs on GitHub Actions** (`.github/workflows/check.yml`):
  - `scripts/check.sh` runs on every pull request and every push to `main`;
  - actions are pinned to full commit SHAs, and the job may only read the repository;
  - `setup-uv` installs uv 0.13.6, the one pin of uv, and uv installs the Python that `.python-version` names;
  - no cache is saved or restored: the install takes seconds, and a restored cache would run in the gate unchecked.
- **`CLAUDE.md` and `.claude/` are the agent guidance** ([Agent guidance](#agent-guidance)).
- **The project has no build backend** (`package = false`): it is a tool run from its clone, never installed as a package, so each tool is given the path of its packages, `mypy_path` for mypy, `pythonpath` for pytest and `PYTHONPATH` for `lint-imports`, and the host's launcher (planned) appends them itself.
- **No package is built on the host** (`no-build = true`): every development tool publishes wheels, and a package built from its source would run its build code with the user's rights.
- **`ruff format` checks the Python blocks of the Markdown documents too,** such as the example of `.claude/rules/tests.md`, so that an example reads as the code it shows.
- **Python 3.12** is the version of the base image of the planned build images, so that the host's side and the steps run the same language.
- **Direct dependencies are pinned exactly,** since a new exporter or a new framework can change the graph a build writes: a new version comes only through a change that names it, and the build that follows checks it.

## Architecture

### Principles

The project follows Clean Architecture: the rules live in a pure domain at the center, and each technology stays at the edge, behind an interface that an inner layer owns. `.claude/rules/architecture.md` states the rules, and tooling enforces them wherever it can ([Enforcement](#enforcement)).

- **The domain is pure,** so it runs anywhere the language runs, and its tests need no I/O, no network and no framework.
- **The application reaches the outside world only through its ports,** declared in its own vocabulary, so its tests run it on fakes, and each technology stays one implementation among possible others.
- **Composition roots wire by hand,** so the whole graph sits in one place and the types check it. A DI container would resolve it at run time, by tokens.
- **What is stored or shared is plain data,** so it saves, travels and replays as it is.
- **SOLID applies to modules, functions and types,** so a new case is added beside the others, without rewriting them.
- **A model family is a recipe** (planned): what a family needs, its sources, its export, its variants and its transcriber, lives in its recipe, and the core judges what a recipe declares. A new family then adds a recipe without changing the core, and a recipe cannot widen the confinement that the core imposes.
- **The host trusts no container** (planned): the host's side never imports a library of the models, and reads what a container writes only through its own strict parsers, since that code is the part of a build that no one has reviewed.

### Layers and modules

| Layer | Module | Responsibility | Depends on |
|---|---|---|---|
| Composition roots | `core` | The project's package, `mostt_model_builder`, and its command line on the host, `host_main.py` (planned), which reads the terminal's input and wires the adapters and the recipes | every layer, and each recipe's `recipe.py` and domain |
| Composition roots | `steps` (planned) | The entry module of each of the core's container steps, which runs in its container on the libraries of its image | `domain` |
| Adapters | `adapters` (planned) | curl, Docker, git and the files of the work directory | `application`, `domain` |
| Application | `application` (planned) | Fetches the sources, runs a step, cleans its outputs, assembles the candidate and verifies it | `domain` |
| Domain | `domain` (planned) | The rules: the pins, the work directory, the step table, the recipe contract, the formats, the parsers of what containers write, the FLEURS bench and its acceptance | — |
| Domain, composition roots | `parakeet-tdt` (planned) | The recipe of the Parakeet TDT family, which builds Parakeet Ultra in ONNX: its sources, its export with NeMo, its check against Transformers, its variants and its transcription with onnx-asr | `domain`, `steps` |

## Repository layout

### Repository tree

```
mostt-model-builder/
├── .claude/                     # Claude Code: settings, rules, skills and the expert reviewers
├── .github/
│   ├── pull_request_template.md # The body of every pull request, with its checklist
│   └── workflows/               # CI: the gate on every pull request and every push to main
├── docs/
│   └── foundations.md           # This document
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
├── pyproject.toml               # The manifest: dependencies, and the configuration of the tools
└── uv.lock                      # The lockfile
```

Planned: `images/`, the Containerfiles of the core's images, and `recipes/`, one directory per model family.

### Module anatomy

The project is a single module, and its packages stand for the modules of [Layers and modules](#layers-and-modules):

```
src/mostt_model_builder/
├── __init__.py                  # The package
├── domain/                      # (planned) the pure rules, one concept per file
├── application/                 # (planned) the use cases and their ports
├── adapters/                    # (planned) one file or directory per technology
├── steps/                       # (planned) the entry modules of the core's container steps
└── host_main.py                 # (planned) the command line on the host
recipes/<family>/                # (planned) a recipe
├── images/                      # its Containerfiles
├── src/mostt_recipe_<family>/
│   ├── domain/                  # its own rules
│   ├── recipe.py                # the plain data it gives the core
│   └── steps/                   # the entry modules of its container steps
└── tests/
```

A recipe keeps its own package, apart from the core's, so that the import contracts can tell the two apart and keep the core from importing a recipe, and one recipe from importing another. All of them share one manifest and one lock, so that a build pins one version of each library.

### Enforcement

Each row names a rule, the rules file that states it, and what enforces it.

| Rule | Enforced by |
|---|---|
| Dependencies point inward (`.claude/rules/architecture.md` > Layers) | `lint-imports`, from the layers contract of `pyproject.toml` |
| Formatted by `ruff format` and linted by `ruff check` (`.claude/rules/conventions.md` > Python) | `ruff format --check` and `ruff check`, in the gate |
| Typed strictly (`.claude/rules/conventions.md` > Python) | `mypy`, strict with `warn_unreachable`, in the gate |
| Python 3.12 syntax (`.claude/rules/conventions.md` > Python) | `ruff`'s `UP` rules, with `target-version = "py312"` |
| The host runs on its own Python and the standard library, isolated (`.claude/rules/container-security.md` > The host) | `tests/test_mostt_model_builder.py`, which imports the package under `-I -S -B` |
| uv never builds a package on the host (`.claude/rules/container-security.md` > The host) | `no-build = true` in `[tool.uv]` |
| Markers are strict and warnings are errors (`.claude/rules/tests.md` > pytest) | `--strict-markers`, `--strict-config` and `filterwarnings = ["error"]` in `pyproject.toml` |
| A commit footer holds only what tools read (`CLAUDE.md` > Workflow) | `attribution` in `.claude/settings.json`, which keeps Claude Code's own commit trailers and pull request line out |
| Every rule no tool can check: architecture, tests as specification… | `/expert-review` ([Code review](#code-review)), before every pull request |

## Conventions

### Code conventions

The conventions serve one aim: the code explains itself, so that it needs no comments, and a reviewer can check it against the domain. `.claude/rules/conventions.md` states them.

- **Names carry the meaning** that comments may not carry, in the vocabulary of the domain. A unit in a name, spelled out or in its standard symbol, makes a unit error visible where it is written.
- **Comments only say why.** The names say what the code does, and the tests state the business rules ([Tests document the rules](#tests-document-the-rules)).
- **The units are written as their symbols,** `ms`, `s`, `Hz`, `KiB`, `MiB` and `GiB`, the units in which Docker, the kernel's cgroups and the measurements of a build give their figures, while bytes are spelled out, since a byte has no symbol that a name could not misread.
- **Number formats are spelled out in names,** `float16` or `eight_bit_blocks`, while the published names keep the spelling their readers expect, such as onnx-asr's `fp16` and `int8` suffixes: the code reads the same everywhere, and the files keep the contract with their readers.
- **Typing is strict,** and a value read from outside, such as a container's report, has no type until it is checked, since the host must trust nothing that a container wrote.

### Git

`CLAUDE.md` > Workflow says how a session works, and the `pull-request` and `writing-commits` skills hold the procedures. The reasons behind them:

- **A worktree per piece of work** keeps parallel sessions from changing each other's files, and keeps the main checkout on `main`, where the user's running work never sees work in progress.
- **A stacked branch** costs only a rebase of its own commits when its base is rewritten, so no work waits for a merge.
- **A history rewritten before it is pushed,** into atomic commits whose messages say why, keeps the reasons that a diff cannot show. Each commit passing the checks of what it touches keeps `main` a safe start for any work, and the last one passing the whole gate keeps every part working with the others. The slower tests would add their run to every commit for what the next commit may still complete, unless they guard what a commit could harm once checked out, such as the safety checks of a deployment, since a commit that a `git bisect` checks out runs as it is. While the whole gate runs in seconds, the check of a commit is the whole gate.
- **A header that says what the change achieves, under the scope it belongs to,** lets a reader find a change in `git log --oneline` without opening it: the diff already lists the files.
- **A footer that holds only what tools read** lets release notes and the issue tracker trust it. A trailer naming the tool that typed a change would add a line to every message and say nothing of its reason.
- **"Rebase and merge"** lands each commit as written, after the review ([Code review](#code-review)) and CI ([Technology stack](#technology-stack)).
- **One writer for `.pm/`,** the coordinating session, keeps its commits from racing, while each session in parallel still writes the plan of its own branch, a file that no other session touches.

### Plans

`CLAUDE.md` > Workflow, "Plan before writing code" says when a plan is due, and the `pull-request` skill how it is made. The reasons:

- **A mistake in a plan costs every step after it:** the code built on it, its review and the rounds of fixes. A plan is short, so thorough reasoning costs less there than anywhere else: the planner and the critic run on Opus at `xhigh`, and the implementation that follows a checked plan at `high`.
- **A plan chooses the method before the code.** A test that covers the whole input space, a simpler design or a library costs nothing to choose in a plan; chosen after the code, it costs the rounds of review that showed the need for it.
- **A critic that did not write the plan checks it,** against the code and the rules, as an expert reviews code it did not write: an assumption about the code that no one checked is the cheapest mistake to catch in a plan.
- **The code follows the plan, or the plan changes first,** since a design changed while it is built escapes the critic.
- **Within one planning, the planner answers the critic itself,** since it already holds the code the plan rests on, in a context far smaller than an implementer's: a new planner would read it all again.
- **A change that leaves the design as it is goes without a plan,** such as a bug fix inside one function: a plan and its check, both at `xhigh`, would cost more than the mistakes they could catch there.
- **The plan lives in `.pm/`,** since the repository holds only the current state, and a plan describes work to come. Each plan has one writer, the session that owns its branch, so plans written in parallel never collide.

### Agent guidance

Agents work from four kinds of documents, each with one job, so that a statement has one home and never drifts from a copy. `.claude/rules/documents.md` > Where a statement lives says what goes where. The reasons:

- **`CLAUDE.md` is the session manual** because Claude Code loads it whole into every session: each line costs every session, so it holds only what every session needs.
- **The rules live in `.claude/rules/`** because a file with `paths` loads only with the files it governs, and a rule stated there in full, with its reason, needs no second document. A load by `paths` can be missed, by a subagent in its own worktree or after a compaction, which is why each file has its line in `CLAUDE.md`.
- **The rules files follow concerns, not directories,** because a concern often spans several directories, where a `CLAUDE.md` per directory would scatter its rules.
- **Skills hold procedures** because a procedure matters only while its task runs, and a skill loads only then.
- **This document holds the design and its reasons,** and states no rule, so that a rule changes in one place.
- **An expert reviewer reads at least the rules files of its domain,** one hop away, so that it never chases a rule across documents. It starts without the session's `CLAUDE.md` files, which would cost it tokens on every turn and could steer it: the rules that apply are the checkout's.
- **Each agent runs on the model and at the effort its task needs,** because every turn of every run pays for both, and a review multiplies the runs: Opus where a miss costs most or where code is written, at `xhigh` only for plans ([Plans](#plans)), Sonnet where written rules guide the check, Haiku for search. A low effort saves Haiku and Sonnet little and costs them much quality, so they never run at it.
- **The history of a decision lives in `.pm/`,** a private repository, so the documents here describe only the present, with its reasons.

## Testing

### Tests document the rules

The tests are the specification of the business rules. To learn how a part of the system behaves, read its tests: each rule reads as one sentence, from the component through the operation and the situations to the test's name, in the vocabulary of the domain. That is why the code carries no comment that states a rule. Every test file follows one structure, so that a reader, and an agent, finds the arrangement, the act and the assertions in the same place in every file. `.claude/rules/tests.md` says how tests are written.

### Test levels

- **The unit tests run on every commit,** since they need no network, no Docker and no model, and the whole gate runs in seconds. They hold the rules of the build, the confinement included: what a container may mount, how the host judges what it wrote, and what a recipe may declare, all checked on plain data and fakes.
- **A build on the real containers and models runs by hand** (planned), since it takes hours, tens of gigabytes and a GPU. Its results, the parity with the reference framework and the error rates on FLEURS, go in the pull request that changes what it builds.

### Budgets

No budget is set yet. The first one comes with the benchmark that measures it.

### Code review

Every branch goes through `/expert-review` before its pull request. It is a project skill in `.claude/skills/expert-review/`, named so that it never shadows the built-in `/code-review`.

- **It runs in its own subagent**, which never saw the code being written, so the review stays independent of the session that wrote the code, and that session keeps its context: the experts report to the review, which waits for all of them.
- **The review runs in the background.** In the foreground, an interactive review can hand back when its first turn ends, while its triage and its experts still run, and their answers then reach the session instead. In the background, Claude Code keeps a fork running until it hands back its report, and delivers its agents' answers to it.
- **One expert per domain, in parallel.** Correctness reviews every diff, and conventions every diff that holds code or configuration. Architecture, tests, performance, security, container security and documentation join when the diff touches their domain. Each expert is an agent of `.claude/agents/`, on Opus for bugs, security, the confinement of the containers, architecture, performance and tests, and on Sonnet for conventions and documentation, which written rules guide. No expert can run a command, so that no reviewed text can make it act.
- **The confinement has an expert of its own,** since a build runs code that no one reviewed, and a mistake at the boundary between the host and the containers would hand that code the user's machine.
- **A triage on Sonnet weighs the experts that the paths pick,** by what the diff holds, before any of them runs: an expert pays a full reading of its rules, the diff and the code even when it has nothing to say, and the triage costs a fraction of that. It keeps an expert when unsure, and never drops correctness, nor security on its paths, nor documentation on an added file, the one expert that reads a file's content for what the repository must not hold, nor narrows what they read, since a missed expert costs more than a needless one, and a triage that reads the diff can be steered by it.
- **A round of fixes is reviewed against the findings it answers,** by their experts, by correctness, by security on its paths, since a fix can weaken a guard as well as add one, and by the experts whose concerns the fixes newly touch, such as tests for a new test, rather than by every expert whose paths they touch: a fix of three files can touch the paths of every domain, and only the experts it concerns have something to check.
- **The orchestrator confirms every blocking and should-fix finding** before reporting it, and lists the findings it dropped with their counter-evidence, so the report makes no claim it has not checked. It also holds every suggested fix to the project's rules, since an expert reads only those of its domain.
- **A target other than the current branch is checked out outside the repository**, so a reviewed `CLAUDE.md` never loads as an instruction.
- **Reviews can run at the same time, from the same checkout:** each one keeps its scope in a temporary directory of its own.
- **Rounds of fixes converge, or the user decides.** A fix can bring the bug that the next round finds, so each round of fixes is reviewed. A defect that a fix brought into the code it fixed says the method is wrong: a test of the whole input space, or a simpler design, ends the chase sooner than more rounds. Past a few rounds, the user weighs what remains against what the rounds cost.
- **A new agent fixes each round,** from a brief: an agent resumed round after round re-reads its whole, growing context on every turn, which costs more than the review it answers.
- **The report says what the review cost,** agent by agent, so that the cost of the rounds shows while it builds up, and the decision to go on weighs it.
- **The report ends the review.** It suggests a fix for each finding but applies none: the session that owns the branch answers each one or has a `fixer` agent fix it, and the user decides at the merge.

## References

- uv: https://docs.astral.sh/uv/
- ruff: https://docs.astral.sh/ruff/
- mypy: https://mypy.readthedocs.io/
- pytest: https://docs.pytest.org/
- import-linter: https://import-linter.readthedocs.io/
- FLEURS, the multilingual speech benchmark that measures a build (planned): https://huggingface.co/datasets/google/fleurs
- Parakeet TDT 0.6B v3, NVIDIA's model of the first recipe (planned): https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3
- Parakeet Ultra, Moondream's post-training of it, the first model built (planned): https://huggingface.co/moondream/parakeet-ultra
