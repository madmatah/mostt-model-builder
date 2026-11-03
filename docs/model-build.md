# mostt-model-builder — The build

This document describes the build part by part: what each command does, and the reasons behind its method. `docs/foundations.md` explains the design as a whole, and `.claude/rules/container-security.md` states the rules that hold the build's code and data. The results of a build, which depend on its run, go in the pull request that changes what it builds, never here. The trial build is a run of these steps by hand, outside this tooling, on the same sources and library versions, whose measurements the tolerances, thresholds and limits below cite.

## The core

The core holds what every model family shares. Built: the work directory, the sources, the confined containers and their boundary with the host, the images and their lock, and the check of the FLEURS bench, which the sections below describe. Planned: the formats and their inspection, the measures of the FLEURS bench and its acceptance, and the recipe contract.

Every command runs as `./model-build.sh --work <dir> <command>`, on the host's Python and its standard library, isolated from the environment and the site packages.

### The work directory

The work directory holds everything a build fetches and writes: the sources, the outputs of each step, its request and its log, the runs of `lock`, the logs of the image builds, and the candidate (planned). It lives outside the clone, so that a build never writes into the repository, and nothing it fetches can be committed by accident.

`--work` names it, and no default does, since a default would choose for the user where tens of gigabytes go. The user creates it, in mode 0700, outside the home directory: containers write into it and the host binds its files into them, so it must hold none of the user's own files, and no other user may change a file between its check and its use. `.claude/rules/container-security.md` > The host gives each check it must pass.

Its layout:

| Path | Holds |
|---|---|
| `sources/<namespace>/<name>/<revision>/<path>` | Each pinned source, read-only, at the place its pin gives it, shared by every recipe |
| `runs/<owner>/<commit>/<step>/` | The outputs of a step at a commit, in mode 2770, which its container writes; the owner is `core` for the core's steps |
| `runs/<owner>/<commit>/requests/<step>/` | The request the host writes for that step, in mode 0750 |
| `runs/<owner>/<commit>/logs/<step>.log` | What its container printed, through the printable filter, and the log of its cleaning, `clean-<step>.log` |
| `lock/<n>/` | A run of `lock`: its `request/`, the copies of the images' manifest and lock, its `output/`, the new lock and its peaks, and its `logs/`, a new number each time |
| `image-builds/<image>.log` | The output of the last build of each image, through the printable filter |

### The sources

A build reads only files that it pins. The core pins its own in `sources.toml`, at the root of the clone: the FLEURS test sets of French (`fr_fr`) and English (`en_us`), each as its index, `test.tsv`, and its audio, `audio/test.tar.gz`, from the dataset `google/fleurs` at its commit `70bb2e84b976b7e960aa89f1c648e09c59f894dd`. A recipe pins its own the same way (planned).

Each pin gives:

| Key | Holds |
|---|---|
| `repository_type` | `model` or `dataset`, the kind of Hugging Face repository the file comes from |
| `repository` | The repository, as `<namespace>/<name>` |
| `revision` | The commit of the repository, in its 40 hexadecimal digits |
| `path` | The file's path in the repository |
| `size_bytes` | The file's size, in bytes |
| `sha256` | The file's SHA-256, in lowercase hexadecimal |

A revision is a commit rather than a branch or a tag, which can move, so that a pin names one content forever, and the size and the hash check that content whatever served it.

`./model-build.sh --work <dir> fetch` downloads each pinned file into the work directory, at `sources/<namespace>/<name>/<revision>/<path>`, with curl, from the file's `resolve` URL on huggingface.co, which redirects to the hub's storage. A download goes into the file's name followed by `.partial`, and a fetch that stops, or fails, resumes it where it ended. Once complete, its size and its SHA-256 are compared with its pin's: a file that matches takes its name, read-only, and one that does not is removed, and the fetch fails, naming both sizes and both hashes. A file fetched earlier is checked again at every fetch, so a file changed since fails the fetch rather than reach a container, and is left for the user to look at and remove.

The fetch reads no token and sends none, since every source is public.

## The containers

Every step of a build runs in a container, started by the host, which takes from the step table, `domain/containers/container_steps.py`, everything Docker is given. The step table holds the core's rows, and each recipe's (planned); `.claude/rules/container-security.md` > Confinement states what each container may and may not have.

A row of the step table gives:

| Field | Holds |
|---|---|
| name | The step's name, such as `fleurs-check` |
| image | The image it runs in, and the image's Python: `/opt/venv/bin/python` in an image with an environment, the base's `/usr/local/bin/python3.12` in one without |
| entry module | The module the image's copy of the bootstrap runs, under `-I -B` |
| sources | The pinned files it reads, each bound by itself, read-only, at its pin's layout under `/sources` |
| request | Whether the host writes a request for it, bound read-only at `/request` |
| outputs | The files it leaves at the top of `/output`, each with its size bound; `peaks.json` among them, except for `clean` |
| limits | Its memory in MiB, with the swap at the same value, its processes, its CPUs, and the size of its `/tmp`, and of its `/scratch` where it asks for one |
| GPU, network | Whether it reaches the GPU, which only a row of a GPU image may, and the network, which only `lock` has |

A row's limits leave about a quarter above the peaks that a full build reads from its `peaks.json`, its memory rounded up to a whole GiB and its processes to a power of two, so that a step that grows beyond them fails with its log rather than take the machine's memory. A row that no full build runs keeps the limits of the run that measured it.

The core's rows:

| Row | Image | Memory MiB | Processes | CPUs | `/tmp` MiB | Reason of the limits |
|---|---|---|---|---|---|---|
| `fleurs-check` | `bench` | 1 024 | 8 | 2 | 64 | it reads the archives member by member, and peaked at 24 MiB and 6 processes |
| `clean` | the image of the step it cleans | 256 | 32 | 1 | 16 | the core's own value: it only removes files |
| `lock` | `uv` | 4 096 | 256 | 8 | 3 072 | limits that a resolution of every image's libraries stays well within, uv's cache in `/tmp` included |

A run of a step:

1. refuses an image that is missing, or whose label is not the tree of HEAD, and refuses the run when an account or a subordinate range holds the container user, 65531;
2. makes the step's output directory, and refuses a step whose outputs exist, naming `clean`;
3. writes its request, if it has one;
4. judges each mount, from the work directory down, entry by entry;
5. runs the container, its output going to its log, through the printable filter, capped at 16 MiB;
6. on a failure, shows the last 8 KiB of the log;
7. otherwise judges the output directory, then each declared output through its descriptor, by its facts, and reads only the step's peaks and the reports that the host parses, through their own strict parsers, and never reads any other output.

Every step ends through the core's shared step modules: it writes its control group's `memory.peak` and `pids.peak` into `peaks.json`, read inside the container at its exit, since the host's polling misses a short container, then sets each of its outputs to mode 0640, since some libraries write theirs readable by their owner alone, which the host is not.

`./model-build.sh --work <dir> clean [<step>]` removes the outputs of a step at HEAD's commit, or of every step: a container of the step's image empties the output directory, mounting it alone, since a directory that a step made there belongs to the container user; the host then removes the empty directory, the request and the log.

## The images

Each image has a Containerfile in `images/<image>/`, built with the repository's root as its context and an allow-list dockerignore beside it, `Containerfile.dockerignore`, which excludes everything but what the image copies. The core's images:

| Image | Holds |
|---|---|
| `bench` | The Python base and the core's sources: the FLEURS check runs on the standard library |
| `uv` | The Python base, the uv binary of the pinned uv image, and the core's sources: the lock's resolution |

`./model-build.sh --work <dir> images` builds them, from a clone with no change and no untracked file, under the tags `mostt-model-builder-<owner>-<image>`, such as `mostt-model-builder-core-bench`, so that they never clash with other images. Each carries the git tree of HEAD as its `mostt-model-builder.tree` label, set by the last lines of its Containerfile, after the heavy layers, which a change of code then reuses: a step runs an image by its ID only while its label is the tree of HEAD, so that a step never runs code that no commit names. The bases are pinned by digest in `domain/containers/base_images.py`.

## The images' environment and its lock

The libraries of every image live in one uv project, `images/pyproject.toml`, the project `mostt-model-builder-images`, with one extra per image that installs any, named `<owner>-<image>`, each pinned exactly. Its `[tool.uv]` holds exactly `package = false`, `no-build = false`, the one environment of Linux on x86-64, an `exclude-newer` that holds the transitive dependencies to what the build measured, and `required-version`, the uv of the `uv` image.

`./model-build.sh --work <dir> lock` resolves its lock:

1. the host copies the manifest and the committed lock, if any, into a new request directory of `lock/<n>/`;
2. the `lock` step copies them into its `/tmp` and runs `uv lock` there, with Docker's default bridge as its network;
3. it writes the new lock into its output directory, in mode 0640, with its peaks;
4. the host judges the new lock as any output, then runs the locks' check and the freshness check on it;
5. it prints the packages added, removed or changed in version against the committed lock, through the printable filter, every package as added for the first lock;
6. only then does it write the new lock to `images/uv.lock`, where the user reviews that change in the diff before committing it.

Both checks read a lock as text, with `tomllib`, since no command on the host may run uv in `images/`, and the gate runs them on both locks, `tests/test_lockfiles.py`. The locks' check reads each package's source and each distribution's URL and hash, and reads the tags of each wheel's name, its Python, its ABI and its platform: `cp312`, `py3`, `py3x` with the ABI `none` and x at most 12, the pure Python wheels that pip and uv install on CPython 3.12, or `cp3x` with the stable ABI and x at most 12, on `any`, `linux_x86_64` or a `manylinux` platform for x86-64, whose glibc PEP 600 gives, the legacy tags included. A new package without such a wheel is a refusal that `lock` reports for the user to decide. The freshness check compares the lock's record of its manifest, its root's metadata and its options, with the manifest itself. `.claude/rules/configuration.md` > Dependencies, "Both locks pass the locks' check", and `.claude/rules/container-security.md` > Images and outputs, "Only `lock` resolves the images' lock, and the user reviews its change", give what each check requires.

A resolution may run the build code of an sdist, which the images' environment allows: `.claude/rules/container-security.md` > Images and outputs states the risks that remain and what stands against them.

## The FLEURS check

`./model-build.sh --work <dir> step fleurs-check` checks the bench before any recipe reads it: in the `bench` image, on the standard library, it reads the index of each language, then its archive member by member, and requires every recording of the index exactly once, under `test/<recording>`, as a WAV file of one channel at 16 kHz, in 16-bit integers or 32-bit floats, with the count of samples its index gives. It reads no member outside the index, and the archives are read in a container only, never on the host. Its report, `fleurs-check.json`, gives each recording's duration in samples, in the order of the index, for the reports of the measures (planned); the host reads it through its strict parser.
