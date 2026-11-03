# mostt-model-builder — The build

This document describes the build part by part: what each command does, and the reasons behind its method. `docs/foundations.md` explains the design as a whole, and `.claude/rules/container-security.md` states the rules that hold the build's code and data. The results of a build, which depend on its run, go in the pull request that changes what it builds, never here.

## The core

The core holds what every model family shares. Built: the work directory and the sources, which [The work directory](#the-work-directory) and [The sources](#the-sources) describe. Planned: the confined containers and their boundary with the host, the images and their lock, the formats and their inspection, the FLEURS bench and its acceptance, and the recipe contract.

Every command runs as `./model-build.sh --work <dir> <command>`, on the host's Python and its standard library, isolated from the environment and the site packages.

### The work directory

The work directory holds everything a build fetches and writes: the sources now, and the outputs of each step, their logs and the candidate (planned). It lives outside the clone, so that a build never writes into the repository, and nothing it fetches can be committed by accident.

`--work` names it, and no default does, since a default would choose for the user where tens of gigabytes go. The user creates it, in mode 0700, outside the home directory: containers will write into it and the host will bind its files into them, so it must hold none of the user's own files, and no other user may change a file between its check and its use. `.claude/rules/container-security.md` > The host gives each check it must pass.

Its layout:

| Path | Holds |
|---|---|
| `sources/<namespace>/<name>/<revision>/<path>` | Each pinned source, read-only, at the place its pin gives it, shared by every recipe |

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
