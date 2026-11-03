# Code conventions

These rules apply to code, tests and configuration, in every part of the repository: the Python of the host and of the containers, and the shell scripts. The examples are in `snake_case`, the casing of Python.

## Naming

Names carry the meaning that comments may not carry.

- **Explicit, in the vocabulary of the domain.** A name says what a thing is or does: `encoder_size_bytes`, `fetch_source`, `is_pinned`. A concept keeps one name everywhere: in the documents, the code and the tests.
- **No generic names:** `data`, `info`, `value`, `item`, `result`, `temp`, `obj`, `manager`, `handler`, `helper`, `util`.
- **No abbreviations and no single letters.** Write `elapsed_ms`, not `dt`; `row_index`, not `i`; `request`, not `req`; `float16`, not `fp16`; `word_error_rate`, not `wer`. Type parameters are named too: `Entry`, not `T`. A unit symbol that [Units](#units) allows is neither an abbreviation nor an acronym, nor a single letter when it ends a name.
- **Acronyms are written as words:** `source_url`, `onnx_runtime`, `JsonReport`. Only the acronyms established in the domain or on the platform are allowed: `id`, `url`, `json`, `toml`, `utf8`, `ci`, `pid`, `uid`, `gid`, `cpu`, `gpu`, `asr`, `onnx`, `pcm`, `wav` and `sha256`. Anything else is spelled out.
- **Booleans are predicates:** `is_pinned`, `has_external_data`, `can_retry`.
- **Functions start with a verb:** `fetch_source`, `judge_output`. Types and values are nouns.
- **Casing:**
  - Python: PEP 8, `CapWords` for classes, `snake_case` for functions, variables, modules, packages and files, `UPPER_SNAKE_CASE` for module constants; a package of the project is `mostt_model_builder`, or `mostt_recipe_<family>` for a recipe;
  - shell: `snake_case` for functions and local variables, `UPPER_SNAKE_CASE` for `readonly` constants and environment variables, `kebab-case.sh` for files;
  - a directory that holds no package, such as a recipe's, is in `kebab-case`: `recipes/parakeet-tdt/`.

  A file is named after the concept it holds, and so is its test.
- **A name that the outside of the program sees keeps its spelling:** a command-line option, an environment variable, a field of a file that another program reads, the name of a published file, such as onnx-asr's `fp16`, `bf16` and `int8` suffixes, or a name a library imposes, such as a weight's name in a checkpoint. Other programs rely on it, so it changes only when changing that interface is worth breaking them, and the change says so.
- **Only the language idioms listed in [Python](#python) and [Shell](#shell) override these rules.**

## Units

- **A quantity that has a unit carries it in its name,** spelled out or in its standard symbol, with one form per unit in the whole project: `decoding_time_ms`, `encoder_size_bytes`, `memory_limit_mib`. Discrete units count too: `window_samples`, `shift_frames`. Dimensionless quantities take no suffix: `overlap_ratio`, `retry_count`. A standard symbol is one that the SI, the IEC or the platform defines, such as `ms`, `Hz`, `KiB` or `px`, and the symbols the project writes are: `ms`, `s`, `Hz`, `KiB`, `MiB` and `GiB`. Every other unit is spelled out, bytes included (`size_bytes`), and so is a unit that has no standard symbol, such as samples or frames. A symbol is a suffix in the casing of the name it ends, never a name on its own: `timeout_s`, `SAMPLE_RATE_HZ`.
- **A type that carries the unit replaces the suffix.** A duration type, or a quantity type of the project's own, already says the unit, and a name that repeats it could contradict it: `timeout: timedelta`, not `timeout_s: timedelta`.
- **A vector or a structure carries its unit, and the axis or the field is a property:** `position_meters.x`, never `position_x_meters`.

## Comments

The names and the structure say what the code does, and the tests state the business rules, so a comment can only add a why.

- **No comments, except a why.** A comment never says what the code does. Rather than comment complex logic, extract a function whose name says it.
- **A business rule is never a comment.** It is a test whose name states it.
- **A why is a last resort,** written only when neither the code nor a test can say it. Typical cases: a workaround for a platform or library bug, with its link; a constraint taken from a reference, with its source.
- **A documentation comment follows the same rule.** A doc block or a docstring that restates the signature, carries tags such as `@param` or `@returns`, or says what the code does is forbidden: the names and the types already say it. One that gives a why is a comment like any other.
- **Allowed directives,** as comments: `# noqa: <code>` and `# type: ignore[<code>]` in Python. A directive that suppresses a check states its reason, next to it.
- **A version label next to a pinned commit SHA is allowed,** such as `# v7.0.1` after an action in a GitHub Actions workflow: it names the release, which the SHA cannot say.
- **Forbidden:** commented-out code, and banners.
- **Scope:** every file but prose documents, configuration and scripts included. No tool checks comments: review does.

## Python

- **Formatted by `ruff format`, at 100 columns, and linted by `ruff check`** with the rule sets `E`, `W`, `F`, `I`, `B`, `UP`, `SIM`, `PT`, `ARG` and `RUF`, so that style and the common bugs never reach a review. Check: the gate.
- **Typed strictly:** `mypy --strict` with `warn_unreachable`, on `src` and `tests`; every function is annotated; no `Any` without a reason, so that a value read from outside is checked before it gets a type; a `# type: ignore[<code>]` names its code and gives its why. Check: the gate (`mypy`) for the types, review for the reasons.
- **Python 3.12 syntax,** the version of `.python-version` and of the images' base. Check: the gate (`ruff`'s `UP` rules, `target-version = "py312"`).
- **A docstring says a why, never what the code does,** as any comment ([Comments](#comments)). Check: review.
- **Python idioms that override the naming rules:** a name the platform imposes, such as `__init__`, a pytest fixture's parameter or `cls`.

## Shell

- **Every script starts with `#!/usr/bin/env bash` and `set -euo pipefail`,** so that a failing command stops it. Check: review.
- **Constants are `readonly` and in `UPPER_SNAKE_CASE`, at the top of the script; functions are in `snake_case`, and their variables `local`.** Check: review.
- **Expansions are quoted, tests use `[[ … ]]`, and `--` comes before a path** that could start with a dash. Check: review.
- **A script finds the repository from its own location,** `REPO_ROOT` computed from `BASH_SOURCE`, so that it runs from any directory. Check: review.

## Input from outside the program

- **Outside input is checked before use.** A file the user loads, a request, a message, a command-line argument, an environment variable, a file that a download or a container wrote, or what a container prints, is checked for its shape and its bounds. Malformed input fails cleanly, without crashing the program or corrupting its state.
- **Outside text never reaches a parser of markup, script, queries, URLs or command lines.** A command's arguments are passed as a list, never through a shell: `subprocess.run` with a list and without `shell=True` in Python, and in shell, a quoted array. An option of `docker` or `curl` built from outside text is built only from a value that its grammar check accepted.
