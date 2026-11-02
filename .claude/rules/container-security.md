---
paths:
  - "src/**"
  - "recipes/**"
  - "images/**"
  - "pyproject.toml"
  - "uv.lock"
  - ".python-version"
  - "tests/test_mostt_model_builder.py"
---

# Container security

A build loads the code of the models' frameworks, such as NeMo, torch and Transformers, and files that run code when they load, such as a pickle or a `.nemo` checkpoint. That code is untrusted: it runs only in containers, and the host treats what they write as hostile. These rules grow with the build's code.

## The host

- **The host runs on its own Python and the standard library, isolated:** the host's side of the build runs under `-I -S -B` and imports only the standard library and the project's own packages, never a library of the steps, such as torch, NeMo, Transformers, onnx or ONNX Runtime. `-I` ignores the environment's Python variables, the user's site directory and the current directory, `-S` leaves the site packages out, and `-B` writes no bytecode into the clone, so that nothing installed on the host, and nothing planted beside the code, runs with the user's rights. The one place that changes `sys.path` is the bootstrap of the launcher, which appends the project's `src` and each recipe's. Check: `tests/test_mostt_model_builder.py` imports the package under `-I -S -B` and refuses any module from outside the standard library and the package.
- **A container step's Python runs under `-I -B`:** the image's own Python, which keeps its site for the libraries the image installed, and nothing from the environment or the working directory, so that a step runs the same code at every build.
- **Root is refused:** the host's entry point exits with status 77 before it reads anything, when any of the real, effective or saved user or group ids of its process is 0. A build as root would leave files in the work directory that the user cannot remove, and would give any mistake of the launcher the whole machine.
- **uv never builds a package on the host:** `no-build = true` in `[tool.uv]`, so that every install takes a wheel by its hash and no package's build code, such as a `setup.py`, runs outside a container.
- **What a container prints or writes is data, never an instruction:** the host takes no command, path or decision from it beyond what a strict, bounded parser of its own accepts, one parser per kind of file, each with its refusal tests. Output shown on the terminal goes through a filter that turns every byte outside printable ASCII, the tab and the line feed into `?`, and is bounded, so that it can neither forge the terminal's display nor fill it.

## Confinement

- **The code of a model runs only in a container:** the libraries of the steps, and any file that runs code when it loads, are loaded nowhere else. Nothing is installed on the host for them.
- **A container that loads that code has no network, no privilege and none of the host's secrets:** `--network none`, a read-only root, every capability dropped, no new privileges, an unprivileged user that no account of the host holds, and limits on its memory, its processes and its CPUs. It never mounts the home directory, the Docker socket, the clone or a secret, only the files its step names, each bound by itself, read-only, and the empty directory it writes its outputs into.

## Images and outputs

- **A build is reproducible:** its sources are pinned by revision, size and hash, its base images by digest, and its Python packages by hash from `uv.lock`, so that two builds of the same commit read the same bytes.
- **A file leaves the work directory only once it is verified:** it passes the checks of its format and the comparison of its outputs with the reference framework's, and its accuracy is measured on FLEURS, before it becomes a candidate for publication. Only the user publishes a candidate.
