# mostt-model-builder — The build

This document describes the build part by part: what each command does, and the reasons behind its method. `docs/foundations.md` explains the design as a whole, and `.claude/rules/container-security.md` states the rules that hold the build's code and data. The results of a build, which depend on its run, go in the pull request that changes what it builds, never here. The trial build is a run of these steps by hand, outside this tooling, on the same sources and library versions, whose measurements the tolerances, thresholds and limits below cite.

## The core

The core holds what every model family shares. Built: the work directory, the sources, the confined containers and their boundary with the host, the images and their lock, the check of the FLEURS bench, the recipe contract and its runs, the formats with their checks, the conversions of an ONNX graph, the candidate and its inspection, the parity reports, the verification of a build against FLEURS, with the bytes it accepted, the model card with its writing for review, the publication, and the check of a published revision, which the sections below describe.

Every command runs as `./model-build.sh --work <dir> --recipe <name> <command>`, on the host's Python and its standard library, isolated from the environment and the site packages. `--recipe` names the model family to build, [Recipes](#recipes) lists them, and every command needs it but `lock`, which resolves the libraries of every recipe's images at once.

### The work directory

The work directory holds everything a build fetches and writes: the sources, the outputs of each step, its request and its log, the runs of `lock`, the logs of the image builds, and the candidate. It lives outside the clone, so that a build never writes into the repository, and nothing it fetches can be committed by accident.

`--work` names it, and no default does, since a default would choose for the user where tens of gigabytes go. The user creates it, in mode 0700, outside the home directory: containers write into it and the host binds its files into them, so it must hold none of the user's own files, and no other user may change a file between its check and its use. `.claude/rules/container-security.md` > The host gives each check it must pass.

Its layout:

| Path | Holds |
|---|---|
| `sources/<namespace>/<name>/<revision>/<path>` | Each pinned source, and each published file that `check-published` downloads, read-only, at its revision, shared by every recipe |
| `runs/<recipe>/<commit>/<step>[-<run>-<shard>]/` | The outputs of a step of the recipe's build at a commit, or of one shard of a run, the core's steps included, in mode 2770, which its container writes |
| `runs/<recipe>/<commit>/requests/<step>/` | The request the host writes for that step, in mode 0750 |
| `runs/<recipe>/<commit>/logs/<step>.log` | What its container printed, through the printable filter, and the log of its cleaning, `clean-<step>.log` |
| `runs/<recipe>/<commit>/candidate/` | The candidate, the host's own copies of the files its recipe names, in mode 0444 |
| `runs/<recipe>/<commit>/README.md` | The model card that `card` writes for review, in mode 0444 |
| `runs/<recipe>/<commit>/publications/<n>/` | A run of `publish`: its `request/`, the request and the card it sends, its `output/`, the report of the commit and its peaks, and its `logs/`, a new number each time, never cleaned |
| `lock/<n>/` | A run of `lock`: its `request/`, the copies of the images' manifest and lock, its `output/`, the new lock and its peaks, and its `logs/`, a new number each time |
| `image-builds/<image>.log` | The output of the last build of each image, through the printable filter |

### The sources

A build reads only files that it pins. The core pins its own in `sources.toml`, at the root of the clone: the FLEURS test sets of French (`fr_fr`) and English (`en_us`), each as its index, `test.tsv`, and its audio, `audio/test.tar.gz`, from the dataset `google/fleurs` at its commit `70bb2e84b976b7e960aa89f1c648e09c59f894dd`. A recipe pins its own the same way, in `recipes/<family>/sources.toml`, and a build reads the core's pins and its recipe's together, which may not pin the same file twice.

Each pin gives:

| Key | Holds |
|---|---|
| `repository_type` | `model` or `dataset`, the kind of Hugging Face repository the file comes from |
| `repository` | The repository, as `<namespace>/<name>` |
| `revision` | The commit of the repository, in its 40 hexadecimal digits |
| `path` | The file's path in the repository |
| `size_bytes` | The file's size, in bytes |
| `sha256` | The file's SHA-256, in lowercase hexadecimal |
| `runs_code_when_loaded` | Optional, `false` by default: `true` for a file that runs code when it loads, such as a `.nemo` archive, which holds a pickle |

A revision is a commit rather than a branch or a tag, which can move, so that a pin names one content forever, and the size and the hash check that content whatever served it. A pin whose file name has a suffix that marks code, a pickle or an archive, such as `.nemo`, is refused unless it says `runs_code_when_loaded = true`, and the core then binds that file only in the one image its recipe declares to read it, so that such a file never reaches a container by mistake.

`./model-build.sh --work <dir> --recipe <name> fetch` downloads the core's pinned files and the recipe's, each into the work directory, at `sources/<namespace>/<name>/<revision>/<path>`, with curl, from the file's `resolve` URL on huggingface.co, which redirects to the hub's storage. A download goes into the file's name followed by `.partial`, and a fetch that stops, or fails, resumes it where it ended. Once complete, its size and its SHA-256 are compared with its pin's: a file that matches takes its name, read-only, and one that does not is removed, and the fetch fails, naming both sizes and both hashes. A file fetched earlier is checked again at every fetch, so a file changed since fails the fetch rather than reach a container, and is left for the user to look at and remove.

The fetch reads no token and sends none, since every source is public.

## The containers

Every step of a build runs in a container, started by the host, which takes from the step table, `domain/containers/container_steps.py`, everything Docker is given. The step table holds the core's rows, and the rows of the recipe that a command names, which the core judges when it loads the recipe ([The recipe contract](#the-recipe-contract)); `.claude/rules/container-security.md` > Confinement states what each container may and may not have.

A row of the step table gives:

| Field | Holds |
|---|---|
| name | The step's name, such as `fleurs-check` |
| image | The image it runs in, and the image's Python: `/opt/venv/bin/python` in an image with an environment, the base's `/usr/local/bin/python3.12` in one without |
| entry module | The module the image's copy of the bootstrap runs, under `-I -B` |
| sources | The pinned files it reads, each bound by itself, read-only, at its pin's layout under `/sources` |
| earlier outputs | The declared outputs of earlier steps it reads, each bound by itself, read-only, at `/inputs/<step>/<name>` |
| candidate | Whether it reads the candidate, which the host assembled, bound read-only at `/candidate` |
| request | Whether the host writes a request for it, bound read-only at `/request` |
| outputs | The files it leaves at the top of `/output`, each with its size bound; `peaks.json` among them, except for `clean` |
| limits | Its memory in MiB, with the swap at the same value, its processes, its CPUs, and the size of its `/tmp`, and of its `/scratch` where it asks for one |
| GPU, network | Whether it reaches the GPU, which only a row of a GPU image may, and the network, which only `lock` and `publish` have |
| token | Whether it reads the maintainer's token on its standard input, which only `publish` does |

A row's limits leave about a quarter above the peaks that a full build reads from its `peaks.json`, its memory rounded up to a whole GiB and its processes to a power of two, so that a step that grows beyond them fails with its log rather than take the machine's memory. A row that no full build runs keeps the limits of the run that measured it.

The core's rows:

| Row | Image | Memory MiB | Processes | CPUs | `/tmp` MiB | Reason of the limits |
|---|---|---|---|---|---|---|
| `fleurs-check` | `bench` | 1 024 | 8 | 2 | 64 | it reads the archives member by member, and peaked at 24 MiB and 6 processes |
| `clean` | the image of the step it cleans | 256 | 32 | 1 | 16 | the core's own value: it only removes files |
| `lock` | `uv` | 4 096 | 256 | 8 | 3 072 | limits that a resolution of every image's libraries stays well within, uv's cache in `/tmp` included |
| `inspect` | `onnx-inspect` | 3 072 | 32 | 2 | 64 | the inspection of the graphs, without their weights, peaked at 2.4 GiB and 20 processes |
| `convert` | `onnx-convert` | 13 312 | 64 | 4 | 256 | the conversions of the encoder, its weights loaded whole for each variant, peaked at 9.7 GiB and 41 processes |
| `publish` | `hub` | 8 192 | 512 | 4 | 2 048 | (hypothesis: these hold the upload of the 2.48 GB external data with hf-xet's chunks in `/tmp`; the trial publication's peaks settle it) |

A run of a step:

1. refuses an image that is missing, or whose label is not the tree of HEAD, and refuses the run when an account or a subordinate range holds the container user, 65531;
2. makes the step's output directory, and refuses a step whose outputs exist, naming `clean`;
3. writes its request, if it has one;
4. judges each mount, from the work directory down, entry by entry, and each output of an earlier step that it binds as it judges any output: the directory of that step holds exactly the names its row declares, and the file is a regular file of the container user's, in mode 0640, within its bound and with a single link;
5. runs the container, its output going to its log, through the printable filter, capped at 16 MiB;
6. on a failure, shows the last 8 KiB of the log;
7. otherwise judges the output directory, then each declared output through its descriptor, by its facts, and reads only the step's peaks and the reports that the host parses, through their own strict parsers, and never reads any other output.

Every step ends through the core's shared step modules: it writes its control group's `memory.peak` and `pids.peak` into `peaks.json`, read inside the container at its exit, since the host's polling misses a short container, then sets each of its outputs to mode 0640, since some libraries write theirs readable by their owner alone, which the host is not.

`./model-build.sh --work <dir> --recipe <name> clean [<step>]` removes the outputs of a step at HEAD's commit, or of every step of the core and the recipe: a container of the step's image, with the core's entry module even in a recipe's image, empties the output directory, mounting it alone, since a directory that a step made there belongs to the container user; the host then removes the empty directory, the request and the log.

## The images

Each image has a Containerfile, the core's in `images/<image>/` and a recipe's in `recipes/<family>/images/<image>/`, built from the archive of HEAD's tree, which `git archive` writes and `docker build` reads on its standard input, so that a build sees the files of the tree alone, at the repository's layout, never a file that git ignores, such as a `.env`. The core's images:

| Image | Holds |
|---|---|
| `bench` | The Python base and the core's sources: the FLEURS check runs on the standard library |
| `uv` | The Python base, the uv binary of the pinned uv image, and the core's sources: the lock's resolution |
| `onnx-inspect` | The extra `core-onnx-inspect`, onnx alone, and the core's sources: the inspection of the candidate's graphs, in an image that produced none of them |
| `onnx-convert` | The extra `core-onnx-convert`, onnx, onnx-ir, onnxconverter-common, ml_dtypes, numpy and ONNX Runtime, and the core's sources: the conversions of a recipe's exported graph |
| `hub` | The extra `core-hub`, huggingface_hub with hf-xet, and the core's sources: the client of Hugging Face's upload protocol, for the publication of a candidate and its card; it runs no code of a model |

An image of a recipe also copies the recipe's sources, at the repository's layout, and installs its libraries from its extra of the images' environment ([Recipes](#recipes) lists them).

`./model-build.sh --work <dir> --recipe <name> images` builds the core's images and the recipe's, from a clone with no change and no untracked file, since the host's side runs from the clone and must run the code its images hold, under the tags `mostt-model-builder-<owner>-<image>`, such as `mostt-model-builder-core-bench`, so that they never clash with other images. Each carries the git tree of HEAD as its `mostt-model-builder.tree` label, set by the last lines of its Containerfile, after the heavy layers, which a change of code then reuses: a step runs an image by its ID only while its label is the tree of HEAD, so that a step never runs code that no commit names. The bases are pinned by digest in `domain/containers/base_images.py`.

## The images' environment and its lock

The libraries of every image live in one uv project, `images/environment.toml`, the project `mostt-model-builder-images`, with one extra per image that installs any, named `<owner>-<image>`, each pinned exactly. Its `[tool.uv]` holds exactly `package = false`, `no-build = false`, the one environment of Linux on x86-64, an `exclude-newer` that holds the transitive dependencies to what the build measured, and `required-version`, the uv of the `uv` image. Its lock is `images/environment.lock`. The manifest takes the name `pyproject.toml`, by which uv finds a project, and the lock the name `uv.lock`, which uv's `--frozen` uses even without a project beside it, only inside a container.

`./model-build.sh --work <dir> lock` resolves its lock:

1. the host copies the manifest and the committed lock, if any, into a new request directory of `lock/<n>/`;
2. the `lock` step copies them into its `/tmp`, the manifest as `pyproject.toml` and the lock as `uv.lock`, and runs `uv lock` there, with Docker's default bridge as its network;
3. it writes the new lock into its output directory, in mode 0640, with its peaks;
4. the host judges the new lock as any output, then runs the locks' check and the freshness check on it;
5. it prints the packages added, removed or changed in version against the committed lock, through the printable filter, every package as added for the first lock;
6. only then does it write the new lock to `images/environment.lock`, where the user reviews that change in the diff before committing it.

Both checks read a lock as text, with `tomllib`, since no uv on the host finds the images' environment, and the gate runs them on both locks, `tests/test_lockfiles.py`. The locks' check reads each package's source and each distribution's URL and hash, and reads the tags of each wheel's name, its Python, its ABI and its platform: `cp312`, `py3`, `py3x` with the ABI `none` and x at most 12, the pure Python wheels that pip and uv install on CPython 3.12, or `cp3x` with the stable ABI and x at most 12, on `any`, `linux_x86_64` or a `manylinux` platform for x86-64, whose glibc PEP 600 gives, the legacy tags included. A new package without such a wheel is a refusal that `lock` reports for the user to decide. The freshness check compares the lock's record of its manifest, its root's metadata and its options, with the manifest itself. `.claude/rules/configuration.md` > Dependencies, "Both locks pass the locks' check", and `.claude/rules/container-security.md` > Images and outputs, "Only `lock` resolves the images' lock, and the user reviews its change", give what each check requires.

A resolution may run the build code of an sdist, which the images' environment allows: `.claude/rules/container-security.md` > Images and outputs states the risks that remain and what stands against them.

A package published as an sdist alone is named in the manifest's `[[tool.uv.dependency-metadata]]`, with its version and no dependency, so that a resolution never builds it to learn them, and an image that needs it builds it in a stage of its own, from the hashes the repository holds and without the network, rather than through `pip download`, which would run its `setup.py` with the network to read its metadata. `.claude/rules/container-security.md` > Images and outputs, "A build is reproducible", gives what such a stage holds, and `tests/test_recipe_listing.py` holds each build stage to it.

The one such package is antlr4-python3-runtime 4.9.3, which hydra-core and omegaconf require for NeMo and which publishes no wheel, built with setuptools 84.2.1 in the export image of Parakeet TDT.

## The FLEURS check

`./model-build.sh --work <dir> --recipe <name> step fleurs-check` checks the bench before any recipe reads it: in the `bench` image, on the standard library, it reads the index of each language, then its archive member by member, and requires every recording of the index exactly once, under `test/<recording>`, as a WAV file of one channel at 16 kHz, in 16-bit integers or 32-bit floats, with the count of samples its index gives. It reads no member outside the index, and the archives are read in a container only, never on the host. Its report, `fleurs-check.json`, gives each recording's duration in samples, in the order of the index; the host reads it through its strict parser.

## The recipe contract

A recipe gives the core plain data, `RECIPE` in its `recipe.py`, of the type `RecipeContract` of `domain/contract/recipe_contract.py`, and the core judges every field of it when it loads the recipe, before any command runs. The host's command line imports a recipe's `recipe.py` and domain, never its steps, which run only in their containers. A recipe adds what its family needs without changing the core, and cannot widen the confinement the core imposes: `.claude/rules/container-security.md` > Confinement gives each check.

| Field | Holds |
|---|---|
| `name` | The recipe's name, `[a-z0-9][a-z0-9-]{0,31}`, equal to its directory under `recipes/`; its package is `mostt_recipe_<name>`, with `-` turned into `_` |
| `images` | Its images, each with its name, never one of the core's, its device, `cpu` or `gpu`, and its extra of the images' environment, `<recipe>-<image>`; at most one of them reads the sources that run code when they load |
| `steps` | Its rows of the step table, each with its name, never one of the core's nor `requests`, `logs` or `candidate`, its entry module inside the recipe's `steps` package, or the core's for a core image, its mounts, and its outputs with their size bounds; and either its image, the recipe's or the core's, its limits, and whether it uses the GPU, or, for a step that produces runs, its template: for each device, `cpu` or `gpu`, its image on that device, its limits, and the outputs of one shard, `lines.jsonl` among them, and `placement.json` too on the GPU and only there |
| `candidate` | The files of the candidate, each with its name, never `README.md` in any case, the name of a Hugging Face repository's model card, its format, the variants it serves, where it comes from, a declared output of a step or a content that the host writes, such as a small configuration file a runtime reads, and, for `onnx-external-data`, `external_data_of`, the `onnx` file whose tensors it holds, which serves the same variants |
| `runs` | Its runs, each with its name, `[a-z0-9][a-z0-9-]{0,31}`, the step with a template that produces it, its device, which the template gives, the variant it reads, one the candidate serves, or `reference` for a run of a reference framework, its languages among the bench's, its count of shards, from 1 to 16, and its reader |
| `conversion_input` | The step and the output that the conversions take as their input, an `onnx` file of the candidate, whose constant weights must then all be plain ([The candidate and its inspection](#the-candidate-and-its-inspection)) |
| `conversions` | The conversions of the input, each with the variant it makes, `[a-z0-9][a-z0-9-]{0,31}`, its conversion with its output file and its parameters ([The conversions](#the-conversions)), and the size bound of its output; each output is an `onnx` file of the candidate that serves its variant alone |
| `build_order` | The steps that `build` runs, the core's `fleurs-check`, `convert` when the recipe converts, and `inspect`, and the recipe's, in their order |
| `parities` | The parities of its exports against their reference framework, each with its name, `[a-z0-9][a-z0-9-]{0,63}`, the step and the declared output that holds its report, never `peaks.json`, and its tolerance on the maximum absolute difference, `absolute_difference_tolerance`, a finite number above 0 |
| `comparisons` | The comparisons of its runs, each with the run it judges, the reference run it is compared with, never itself, both runs of the recipe that share at least one language, and its rule ([The verification](#the-verification)): an equivalence, with its least ratio of utterances whose token ids are identical, above 0 and at most 1, and its bound in points on either side, above 0 and at most 10; a non-inferiority, with its bound in points on the upper side; or a figure reported only. A pair of runs is compared once |
| `model` | Optional: the model it publishes, as its card needs it: its title, which the card and the commit of a publication give; the path of the card's template, relative to the recipe's directory; the libraries whose versions the card gives as those that measured its variants, by their normalized names; the name under which the card shows each variant of the candidate, with an optional note; the title of each run of a reference framework that the card names, such as "the original model in NeMo"; and the placeholders of the template that each take the pinned revision of a repository. The core judges only the template's path, which reaches the file system: a file name under the file-name grammar, in the recipe's directory, where it must be a regular file, never a link. The rest is the recipe's own data, which reaches neither Docker nor the file system, and the maintainer reads the card it gives before each publication |

A mount names either a pinned file of the core's or the recipe's sources, by its repository and its path, a declared output of an earlier step of the recipe, by the step and the file's name, or the candidate. No other directory is mounted, so that a step sees only the files its row names.

A parity report is a JSON object of exactly three keys: `name`, the parity's own; `maximum_absolute_difference`, a finite decimal number of at least 0; and `compared_count`, the count of outputs compared, from 1 to 1 000 000. Once its step has run, the host reads each parity report of that step through its strict parser, `domain/acceptance/parity_reports.py`, which refuses a report that names another parity, and prints its difference beside its tolerance; `verify` refuses the build when the difference exceeds it ([The verification](#the-verification)).


## The runs

A run is a transcription of the bench by one engine, on one device, of one variant of the candidate or of the reference framework. The core derives its rows from the template of the step that produces it, `domain/contract/derived_rows.py`, one per shard: each takes the image and the limits of the run's device, the outputs of a shard followed by the step's own, the GPU on a run of the GPU alone, and a request, `shard.json`, that names the run, its variant, its device, its languages, the shard and the count of shards. A step whose runs make a single row keeps its name, so that a later step binds its outputs by it; the rows of the others are named `<step>-<run>-<shard>`, and the contract refuses a row whose name another row already holds, since its outputs would land in the other's directory. The utterances of the run's languages, in the order of their indexes, are dealt to the shards in turn. `./model-build.sh --work <dir> --recipe <name> step <step>` runs every row of the step, the shards of a run at the same time, each in its own container with its own CPUs. Before a row that binds the candidate starts, the host reads the inspection's report again and judges its rules and each copy of the candidate, `application/assemble_candidate.py`, so that a run never reads a candidate that its inspection refused, never inspected, or that changed since.

Each shard writes `lines.jsonl`, one JSON object per utterance, in a form its recipe chooses. The host reads it through the run's reader, a pure function of the recipe that turns the lines and the candidate's text files into the core's hypotheses, `domain/bench/hypotheses.py`: the utterance, its language, its text, its count of unknown tokens and its decoding time, so that the core never sees a token. The core parses the lines strictly, bounded in size, count and length, and refuses a hypothesis in a language other than its run's, an utterance that is not a recording of the bench or that has two hypotheses, a text that holds a character a reader cannot see, a count that is not one, or a decoding time outside 0 to an hour.

A shard on the GPU also writes `placement.json`, from a profile of ONNX Runtime over 5 utterances after the warm-ups: for each provider and operator, the element types of its nodes and their count. The host refuses a node on the CPU that handles a tensor other than `int64` or `bool`, `domain/onnx/node_placement.py`, since ONNX Runtime places there only the arithmetic of shapes, lengths and masks, and a node of floating-point tensors there would be a part of the model fallen back to the CPU, silently. A shard on the CPU never writes one, since the contract refuses a template of the CPU that declares it, so every placement the host reads is judged.

## The verification

`./model-build.sh --work <dir> --recipe <name> verify` measures the build on FLEURS and holds it to its recipe's rules, on the host, from the outputs of its steps, without a container: `application/verify_build.py`. `build` runs it after its last step. It first judges the candidate again: it reads the report of `inspect` through the judgment of its outputs, judges every file of the candidate as the host's own copy, a regular file of the user's, in mode 0444, with a single link, within the bound of the output it was copied from or the length of the content the host wrote, and runs the inspection's graph rules and the check of the external data again on that report and those files, so that no candidate that its inspection refused, that it never inspected, or whose files may have changed since, is measured. A copy that failed stops `inspect` before its container runs, so a partial candidate has no report either. It then hashes every file of the candidate, each opened relative to the judged candidate directory without following a link, and judges the facts of the descriptor it hashes as the host's own copy again, so that the SHA-256 it records is that of the bytes the runs read. It reads every output through the judgment of [The containers](#the-containers), as a later step would bind it: the lines of every shard of every run through the run's reader, the placement of every shard on the GPU, and the report of every parity, with the candidate's text files that the readers need. It reads the index of each language of the bench from the work directory, a pinned source, and checks it again against its size and its SHA-256 before it reads a byte of it, since the transcriptions of the index are the measure of every run. Then:

1. **Every run covers its languages.** A run holds every utterance of the index of each of its languages exactly once across its shards, and no utterance outside the index, `domain/bench/run_coverage.py`, so that no run is measured on a part of the bench, nor an utterance counted twice.
2. **Every run is measured.** Each hypothesis and each transcription of the index, FLEURS's `transcription` column, are normalized alike, `domain/bench/normalization.py`: NFKC, lowercase, the typographic apostrophes `’` and `‘` read as `'`, every run of characters that are neither a word's nor an apostrophe turned into a space, which keeps `_`, then every apostrophe at the edge of a word turned into a space. It is the normalization of the trial build, with `‘` added, which FLEURS en writes where an apostrophe stands. The errors of each utterance are the fewest substitutions, deletions and insertions of words that turn its transcription into its hypothesis, `domain/bench/word_error_rate.py`, and the word error rate of a run on a language is the sum of its errors over the sum of the transcriptions' words. A transcription that normalizes to no word is refused, since an utterance without words cannot be rated. Each run is rated twice, without its unknown tokens and with them, since the vocabulary of Parakeet TDT lacks characters that FLEURS writes, such as `«` or `’`, which the model writes as `<unk>`: the rate without them is the one the rules judge, the one the variants were chosen on, and the rate with them shows what a runtime that keeps them would print.
3. **Every comparison is measured,** on each language both of its runs hold: the share of utterances whose token ids are identical in both, and the difference of their word error rates in points, the run's minus its reference's, with its 95 % interval from a paired bootstrap, `domain/acceptance/paired_comparison.py`. The utterances, in the order of their ids, are drawn with replacement 2000 times, each draw the size of the language, from a generator of its own seeded with 0, so that an interval replays at every verification; the interval runs from the 51st to the 1950th of the sorted resamples. The pairing keeps each utterance's difficulty out of the difference, which a comparison of two independent rates would add to its noise.
4. **Every rule is judged,** `domain/acceptance/acceptance.py`: a parity passes when its difference is at most its tolerance; an equivalence when the share of identical token ids reaches its ratio and both bounds of the interval lie within its bound on either side; a non-inferiority when the upper bound of the interval is at most its bound; a figure reported only always passes.
5. **The report is written,** `verification.json` beside the outputs of the run, `runs/<recipe>/<commit>/verification.json`: whether the build is accepted, each parity with its difference, its tolerance and its failure, each run on each language with its counts of utterances, words, errors without and with its unknown tokens, its rates, its unknown tokens and its decoding time, each comparison on each language with its rule, its identical token ids, its difference, its interval and its failures, and each file of the candidate, in the contract's order, with its format, its variants, its size and its SHA-256. Then `verify` prints the same figures, and fails, naming each broken rule, when any rule fails. `clean` removes the report, whatever step it cleans, since a report of outputs that are gone would vouch for what no longer exists.

Token ids are compared rather than texts, since two tokenizations can render the same words: the core never reads a token, it only compares a run's ids with another's, which the reader gives with each hypothesis's two texts, with and without its unknown tokens.

**Speed** takes no part in a verification. Each transcription records its decoding time, which the report sums per run and language, but no rule reads it: a decision of speed, such as the choice of a quantization, rests on a comparison of its own. Such a comparison runs its two configurations in ABBA blocks, A then B then B then A, each block over the whole bench, and reads the ratio of their times from the blocks: alternating utterance by utterance slows every run, by 2.6 times in the trial build, and plain AB blocks leave an effect of position of 1 to 5 % in favor of one side.

## The formats

Every file of a candidate has one of three formats, which the core knows by name, `domain/formats/file_formats.py`; the contract refuses any other, and a format comes with the first recipe that needs it:

| Format | Holds |
|---|---|
| `onnx` | An ONNX model |
| `text` | A text file in UTF-8, such as a vocabulary or a configuration |
| `onnx-external-data` | The external data of one `onnx` file, its tensors at the offsets that file gives |

Whatever its format, a candidate's name follows the grammar of a published file name, `[A-Za-z0-9][A-Za-z0-9._-]{0,127}`, and is refused with a suffix that marks code, a pickle or an archive, in any position and whatever its case, so that no published file can carry one.

Every candidate file, whatever its format, is refused when its first 512 bytes carry the signature of code, a pickle or an archive, `domain/formats/file_signatures.py`: a binary pickle, a zip, a NumPy array, an executable, a script, a tar, gzip, bzip2, xz, Zstandard or 7-Zip archive. Each format then has its own check, as the file streams through the host:

| Format | Check |
|---|---|
| `onnx` | The top level of a `ModelProto`: each field of the table of `onnx.proto` in its own wire type, a singular field once, keys in their shortest encoding, varints within 64 bits, an `ir_version` from 1 to 64 and a graph; then the inspection of its graph |
| `text` | Strict UTF-8, without NUL |
| `onnx-external-data` | Against the inspection of its `onnx` file: every tensor that names it lies within it, none overlaps another, they cover it from its first byte to its last without a gap, as the exporter lays them, and no other `onnx` file names it |

The checks of `onnx` and `text` and the signatures are those of mostt's model store, adapted, so that every published file passes the store that installs it.

## The conversions

`./model-build.sh --work <dir> --recipe <name> step convert` converts the recipe's conversions' input, `domain/onnx/`, in the `convert` row, which the core derives from the contract: in the `onnx-convert` image, it binds the input and its external data, each by itself, side by side at `/inputs/<step>/`, reads the request that the host writes, `conversion-request.json`, and writes one file per conversion, each a single ONNX file without external data, and its report, `conversion-report.json`. It converts one variant at a time, the input loaded whole each time. Three conversions are built:

| Conversion | Method | Parameters a recipe gives |
|---|---|---|
| `float16` | onnxconverter-common's `convert_float_to_float16`, after a shape inference on the graph without its weights, with its own list of operators kept in float32, then the realignment of the `Cast` nodes | the operators it keeps in float32 beyond the core's; `keep_io_types`, true unless the recipe says otherwise |
| `bfloat16` | the conversion to float16 with `Pad`, `Tile` and `Floor` kept in float32, then every float16 tensor, value and `Cast` retyped bfloat16 | the same |
| `eight-bit-blocks` | ONNX Runtime's `MatMulNBitsQuantizer`, 8 bits, weight-only, on every matrix product of a constant weight | the size of its blocks, a power of two from 16 to 256; whether they are symmetric; ONNX Runtime's accuracy level, from 0 to 4; and the count of products it quantizes |

The decisions hold for any ONNX graph on ONNX Runtime 1.31 and onnxconverter-common 1.16, so they are the core's, `domain/onnx/precision_decisions.py`:

- **`Pad`, `Tile` and `Floor` stay in float32 in bfloat16,** since ONNX Runtime 1.31's CUDA provider has no bfloat16 kernel for them and refuses the whole graph when one is left in bfloat16; torch 2.15's export brings the `Floor`. A recipe adds an operator of its own family to the list, never removes one.
- **The inputs and outputs stay in float32** with `keep_io_types`, as a runtime feeds them, the casts inside the graph.
- **A `Cast` whose output the conversion declares float16 targets float16,** where onnxconverter-common leaves its target at float32, so that the graph's types agree.
- **A float32 weight beyond float16's range is clamped,** under 1e-7 in magnitude to 1e-7 and above 1e4 to 1e4, onnxconverter-common's defaults, and each one is counted.

The host reads the report through its strict parser, `domain/onnx/conversion_report.py`: for each conversion, in the order of the request, its kind and its output, its count of nodes, its count of products in 8-bit blocks, the operators it kept in float32, the counts of clamped initializers and values, and of `Cast` nodes realigned. It refuses a report that describes another conversion than the request asked, operators kept in float32 other than the core's and the recipe's, or a count of products in 8-bit blocks other than the recipe declares, 0 for a conversion to float16 or bfloat16. The inspection then judges each converted file again, from its graph ([The candidate and its inspection](#the-candidate-and-its-inspection)).

## The candidate and its inspection

`./model-build.sh --work <dir> --recipe <name> step inspect` assembles the candidate, `runs/<recipe>/<commit>/candidate/`, and inspects it, `application/assemble_candidate.py`. The host copies each file the contract names from the declared output of its step, through the descriptor of the step's judged directory, into a new file of its own, in mode 0444, never a link, hashing it and checking its signatures and its format as it streams by; it writes the contents the recipe gives itself, such as `config.json`. Then the `inspect` row reads every `onnx` file of the candidate, bound read-only, without its weights, and writes its report, from a walk of every graph of the model, `domain/onnx/model_graphs.py`: its main graph, each subgraph at any depth, each function and each subgraph inside a function. For each file, the report gives its opsets, its local functions, the count of its nodes by scope, domain and operator, the count of nodes that hold `metadata_props`, the count of its training graphs, the count of its sparse initializers in every graph, each matrix product, `MatMul` or `Gemm`, with the trace of its weight back through the producers that hold only constants, `Constant`, `Identity`, `Transpose`, `Squeeze`, `Unsqueeze`, `Reshape` and `Cast`, to an initializer with its rank, a `Constant` node or a computed tensor, and the location, offset and length of every external tensor, wherever it lies: an initializer, a sparse initializer, a tensor attribute of a node, or a function's default attribute. A subgraph sees its own initializers and its ancestors', never a sibling's, and a function sees none, since ONNX gives it none of the graph that calls it. The host reads the report through its strict parser, `domain/onnx/inspection_report.py`, and judges each graph, `domain/onnx/graph_rules.py`:

- every node is of ONNX's domain, of ONNX Runtime's `com.microsoft`, or calls a function the model defines;
- no node holds `metadata_props`, where an exporter records paths of its container;
- the model holds no training graph, which no runtime of the candidate runs;
- the model holds no sparse initializer, whose weights no rule of the products judges, since the quantization converts dense initializers alone and the exports hold none;
- a tensor lies outside the graph's file only in the `onnx-external-data` file the contract gives it;
- in the conversions' input, every product whose weight is a constant reads it as a plain 2-D initializer, read directly, the form that the quantization to 8-bit blocks converts, so that no product is left unquantized;
- in the output of the conversion to 8-bit blocks, no `MatMul` or `Gemm` of a constant weight is left, and the count of `MatMulNBits` nodes of ONNX Runtime's domain is the count the recipe declares, so that a partial quantization, such as one that leaves the pointwise convolutions in float32, fails;
- any other file, such as the decoder and joint, keeps its products as they are: the rules follow where a file comes from, never its name or its variant, since a recipe may publish one file under several names;
- the opsets are recorded, not judged.

`clean inspect` removes the inspection's outputs and the candidate. The assembly refuses to start while the outputs of an earlier inspection remain, since a candidate assembled beside them would be read against a report of other copies.

## The published layout

The published files are the contract with mostt, which pins them by hash: their names, their formats and their structure. The two repositories share no code, and the names and the suffixes that this one refuses are those of mostt's model store, so that every published file passes it. The candidate of Parakeet TDT, laid out for onnx-asr:

| File | Format |
|---|---|
| `encoder-model.onnx` | `onnx`, the FP32 encoder |
| `encoder-model.onnx.data` | `onnx-external-data`, the external data of `encoder-model.onnx` |
| `encoder-model.fp16.onnx`, `encoder-model.bf16.onnx`, `encoder-model.int8.onnx` | `onnx`, the converted encoders, each a single file |
| `decoder_joint-model.onnx`, `decoder_joint-model.fp16.onnx`, `decoder_joint-model.bf16.onnx`, `decoder_joint-model.int8.onnx` | `onnx`, the FP32 decoder and joint, the same file under each name, since onnx-asr picks one file per quantization |
| `vocab.txt` | `text`, the vocabulary of Parakeet TDT 0.6B v3, at the SHA-256 `d58544679ea4bc6ac563d1f545eb7d474bd6cfa467f0a6e2c1dc1c7d37e3c35d` |
| `config.json` | `text`: `{"model_type": "nemo-conformer-tdt", "features_size": 128, "subsampling_factor": 8, "max_tokens_per_step": 10}`, which onnx-asr reads, since its defaults would choose the wrong preprocessor |
| `README.md` | The model card ([The model card](#the-model-card)), which `publish` adds to the candidate |
| `.gitattributes` | The Hub's own, which it writes into every repository it creates |

## The publication

### The model card

The card is the `README.md` of the Hugging Face repository that the files are published in, `domain/publication/model_card.py`: the recipe's template, a `string.Template` of the standard library, with each placeholder filled from the build's records, the same bytes for the same records, since it holds no date and writes every number in a fixed format. The template holds the prose: the front matter that the Hub reads, the summary, the languages and limitations, how the files were built, the usage and the license and attribution; the core fills what the build computes. A placeholder that the core does not fill fails the render, which is the only check of the template's text; its path and its file are judged when the recipe loads ([The recipe contract](#the-recipe-contract)). The core takes the model's description from the recipe ([The recipe contract](#the-recipe-contract)), the pins, a verification with the size of every file of the candidate, the runs and the conversions of the contract, and the libraries that measured the variants, each with its version on each device, `domain/publication/measuring_libraries.py`: the libraries that the description names, as the images of the variants' runs pin them, the images of a reference framework's runs left out. It reads by variant, from the summary of each, `domain/publication/variant_summaries.py`. Its placeholders:

| Placeholder | Holds |
|---|---|
| `title` | The description's title |
| `variant_table` | A table for choosing a variant: each variant under the description's name for it, the devices its runs used, its size, the sum of the files it loads, the files of every variant included, in GB, 10⁹ bytes, as the Hub shows sizes, and one word error rate per language that the rated runs measured, in percent, without the unknown tokens, which the vocabulary cannot write. A variant's rated run is its run on the CPU, or on the GPU when it has none, since the runtime and the hardware, not the files, make the devices differ |
| `variant_sections` | A section per variant: its own files; the description's note on it, if any; one sentence on its differences from the runs its rated run is compared with, from each comparison that judges the rated run against a run of another variant or of a reference framework, and each that judges a reference framework's run against the rated run, its difference turned around, each with its differences in points per language and its verdict, a reference run named by the description's title for it; and its conversion, or none for the exported graph |
| `shared_files` | The files that serve every variant, when the candidate has more than one |
| `measured_languages` | The languages that FLEURS measured |
| `sources` | Each pinned repository, once, linked on the Hub at its revision |
| `measured_with` | The versions of the libraries that measured the rates, on each device |
| `build_commit` | The commit of this repository that built the files, so that the card links the code that built them |
| One per pinned revision | The revision at which the pins name the repository that the description gives for it, such as an upstream work that the attribution links |

The card gives no file's size or SHA-256, nor the libraries and base images of the tooling: the Hub shows each file's size and hash, a hash beside its file in the same repository proves nothing, and the integrity of the published files rests on mostt's pins of their hashes and on the check of the published revision against the build. A variant without a name in the description, a reference run that a sentence names without a title, a variant without a run, a pinned revision that no pin names or that takes a placeholder the core fills, and a placeholder of the template that nothing fills fail the render, with `CardError` or `VariantSummaryError`, since the card would have no word for them.

CC BY 4.0 asks a work that adapts licensed material to credit its creator, keep the notices its card gives, link to it, and say that it was modified, its section 3(a)(1). The template therefore links each upstream work on the Hub at its pinned revision, and links the code of this repository at the commit that built the files, which must therefore be on GitHub's `main` before the card is written, since a rebased or unpushed commit leaves the link without a target and a published card is held to its reviewed bytes. The recipe's test of its card renders it from fixed measures and compares it with a committed copy, so that a change of the card shows in its diff.

### The card for review

`./model-build.sh --work <dir> --recipe <name> card` prepares the publication, `application/prepare_publication.py`:

1. it refuses a clone with any change, untracked files included, since the card states the versions that HEAD's `images/environment.toml` pins;
2. it refuses a recipe that describes no model, naming the missing description;
3. it runs the verification again, which writes the same `verification.json` and hashes every file of the candidate, and refuses a build that it does not accept;
4. it renders the recipe's template from that verification, with the versions of the libraries the recipe names, as the images of its variants' runs pin them, so that the card's "accepted" and "passed its comparisons" hold.

It then writes `runs/<recipe>/<commit>/README.md`, in mode 0444, for the user and the reviewers to read before anything leaves the machine. A card already there that equals the new render is kept, and `card` says it is unchanged; one that differs is refused, naming `clean`, so that the card on disk is always the render of the records beside it. `clean` removes the card, whatever step it cleans, as it removes `verification.json`. `card` publishes nothing.

### Publishing

`./model-build.sh --work <dir> --recipe <name> publish --repository <namespace>/<name>` publishes the candidate and its card into a Hugging Face repository that the maintainer created, `application/publish_candidate.py`. The maintainer runs it, at a terminal, and its check of the terminal and the token typed there are its guard; `.claude/settings.json` denies it to agents as a defense in depth, which matches only the literal command.

1. It refuses to go on unless its standard input is a terminal, right after its arguments are read and before it opens the work directory.
2. It refuses a repository outside the grammar of `<namespace>/<name>`.
3. It prepares the publication as `card` does ([The card for review](#the-card-for-review)), and refuses a card that differs from `runs/<recipe>/<commit>/README.md`, or a missing one, naming `card`: the card it sends is the one that was reviewed, and with it every figure it gives.
4. It shows the repository and each file, the card included, with its size and its SHA-256, and asks the maintainer to type the repository again; a difference stops it.
5. It reads the token at the terminal with echo off, as `getpass` does, bounded to 512 bytes, the terminal's settings restored on the way out, an interruption included, and holds it to the grammar that `.claude/rules/container-security.md` > The host gives a token, `domain/publication/hub_token.py`.
6. It writes its request into `runs/<recipe>/<commit>/publications/<n>/request/`, numbered as `lock` numbers its runs: `publication-request.json`, the repository, a commit message from the card's title, and each file with its size and its SHA-256, and `README.md`, the card.
7. It judges the inspected copies again, as every row that binds the candidate does, then runs the `publish` row in the `hub` image, with the token on the container's standard input alone, and prints where its log is while the upload runs.
8. The step, `steps/hub_publication.py`, reads the token, takes the repository's head as the parent of its commit, refuses a repository that holds any file but the Hub's `.gitattributes` and the build's own, reads each file as huggingface_hub's `CommitOperationAdd` reads it, refuses before it commits when one reads as another size or SHA-256 than the request gives, `domain/publication/requested_files.py`, then commits every file in one commit on that parent, which the Hub applies whole or not at all; it never creates the repository.
9. The host parses `publication.json`, the repository, whether it is private, the parent revision and the new one, and each file with the size and the SHA-256 the container read, `domain/publication/publication_report.py`, and refuses another repository, a missing or an extra file, or a size or a SHA-256 that differs from the verification's or the card's.
10. It prints the revision, whether the repository is private, and the `check-published` command, which shows what the Hub stored ([Checking the published files](#checking-the-published-files)).

A commit carries the identity of the token's account: huggingface_hub's `create_commit` takes no author.

### Checking the published files

`./model-build.sh --work <dir> --recipe <name> check-published --repository <namespace>/<name> --revision <commit>` checks that a published revision holds exactly the build's files and its card, `application/check_published.py`, without a token:

1. it holds the repository and the revision, 40 hexadecimal digits, to their grammars before any request;
2. it refuses a recipe that describes no model, and a missing card, naming `card`;
3. it collects the files to compare against: each file of the candidate, judged again with the inspection's report and hashed as `verify` hashes it, and the card, `runs/<recipe>/<commit>/README.md`, judged as the host's own read-only file and hashed;
4. it reads the listing of the revision, `https://huggingface.co/api/models/<repository>/tree/<revision>`, with curl and `fetch`'s options but its retries, in a single attempt, within the time and the size that `.claude/rules/container-security.md` > The host bounds it to;
5. it parses the listing strictly, `domain/publication/published_tree.py`, reading only the `type`, the `path` and the `size` of each entry, since the Hub adds fields of its own, such as `oid`, `lfs` and `xetHash`;
6. it refuses a directory, a missing file, a file of another size, and any file but the Hub's `.gitattributes` and the build's own;
7. it downloads each file as `fetch` downloads a source, through a pin of the published repository at the revision, with the build's size and SHA-256, into `sources/<namespace>/<name>/<revision>/`, where a file that does not match is removed and fails the check;
8. it prints each file with its SHA-256, then the repository and the revision, which mostt pins.

The downloads take as much room as the candidate, 5.9 GB for Parakeet TDT.

## Recipes

| Recipe | Builds |
|---|---|
| `parakeet-tdt` | Parakeet Ultra, Moondream's post-training of NVIDIA's Parakeet TDT 0.6B v3, in ONNX |

### Parakeet TDT

**Sources.** `recipes/parakeet-tdt/sources.toml` pins Moondream's `config.json`, `tokenizer.json` and `model.safetensors` from `moondream/parakeet-ultra`, and the `.nemo` archive of `nvidia/parakeet-tdt-0.6b-v3`, the one source of the build that runs code when it loads. Moondream publishes weights in the layout of Transformers, without a `.nemo`, so NeMo builds the model from the configuration of v3's archive, whose architecture and vocabulary Ultra keeps, and loads Moondream's weights into it. The same file pins v3's `processor_config.json`, `tokenizer_config.json` and `generation_config.json`, at the revision of the archive, which Transformers reads with Moondream's files, since Moondream publishes no processor and Ultra keeps v3's.

**The export image,** `mostt-model-builder-parakeet-tdt-export`, installs the extra `parakeet-tdt-export`: NeMo 3.0.0 with its speech recognition, torch 2.15.0, safetensors, onnx, onnxscript, onnx-ir, ONNX Runtime and numpy, each pinned exactly, and antlr4-python3-runtime from its build stage. It is the one image that reads the `.nemo`.

**`describe`** lists the tensors on both sides of the weights' mapping, so that the mapping is tested on the host against real names without loading a model there: Moondream's, read from the header of its safetensors file without loading a tensor, and those of NeMo's model, built from the `.nemo`'s configuration alone. NeMo reads the configuration with `restore_from(return_config=True)`, which reads no checkpoint; the step extracts only the tokenizer's files that the configuration names, `nemo:<file name>`, each found by its exact name at the archive's root, into the container's `/tmp`, points the configuration at them, and sets to null the sections of the training, validation and test data, which would otherwise build data loaders for datasets that are not there. The checkpoint, a pickle, is never loaded: the weights come from Moondream's safetensors file. `recipes/parakeet-tdt/src/mostt_recipe_parakeet_tdt/domain/export/nemo_configuration.py` holds that preparation, tested on the archive's real names.

| Row | Image | Memory MiB | Processes | CPUs | `/tmp` MiB | Reason of the limits |
|---|---|---|---|---|---|---|
| `describe` | `export` | 5 120 | 128 | 8 | 512 | NeMo's build of the model, with Moondream's weights loaded, peaked at 3.4 GiB and 60 processes |

Its outputs, `moondream-tensors` and `nemo-state-dict`, give one tensor per line, its name, its type and its shape, separated by tabs, in printable ASCII. Copies of them are committed in `recipes/parakeet-tdt/tests/fixtures/`, where the recipe's tests read them through their strict parser, `domain/export/tensor_fixtures.py`, to check the mapping of the weights' names and the names of the pointwise convolutions against the real names; a change of the sources runs `describe` again and replaces them.

**`export`** builds the FP32 ONNX files of Parakeet Ultra from NeMo's model, as parakeet-rs's export of Parakeet TDT does, at its commit `cadebd4469921ff426bf179e1cfd78fbe6fd9e05`: NeMo describes the model, and parakeet-rs's export is the one whose graph, speed and accuracy the files were measured against.

1. NeMo builds the model from the `.nemo`'s configuration alone, as for `describe`, and copies Moondream's weights into its own tensors, one by one, so that no second float32 copy of the weights is held beside the model: each name of Moondream's file but those of its voice activity head, `vad_head.*`, which serves Moondream's own runtime, is mapped onto NeMo's by the rules of parakeet-rs, `recipes/parakeet-tdt/src/mostt_recipe_parakeet_tdt/domain/export/weight_names.py`, and the mapping refuses a name that NeMo's model does not hold, a shape that differs, two names onto one, and any name of NeMo's model left to its construction but the filter bank and the window of its preprocessor, which NeMo computes. Moondream stores most weights in float16, widened to the float32 of NeMo's model. Moondream's tokenizer must begin with the pieces of NeMo's, since `vocab.txt` comes from NeMo's. torch runs on as many threads as the row has CPUs.
2. The unpatched encoder's outputs on the first 20 utterances of FLEURS fr's index are kept as the reference of the parity.
3. The 48 convolutions of kernel size 1 of the encoder's Conformer layers, two per layer, which `domain/export/pointwise_convolutions.py` names, run as matrix products, as NVIDIA-NeMo/Speech #16052 runs them, on features kept channel-last, with no added transpose, and with their weights made 2-D before the export: each product then reads its weight as a plain 2-D initializer, the form a quantization of 8-bit blocks converts, where parakeet-rs's graph reads `Transpose(Squeeze(<3-D initializer>))`, which a quantizer leaves in FP32. NeMo 3.0.0 predates that pull request, so the step patches the forward of each `ConformerConvolution` rather than build NeMo from git, which the images' lock and `--no-build` refuse.
4. The attention runs through torch's `scaled_dot_product_attention`, as parakeet-rs's export of Parakeet TDT does, so that each layer exports as one ONNX `Attention` node. The positional encoding takes its rows with `narrow()`, the `exact_position_slices` of parakeet-rs's `onnx_export.py`, which its export of Parakeet TDT does not call: it is the configuration whose graph, parity and speed were measured, and it spares the graph one `Add`.
5. torch's dynamo exporter writes the encoder at opset 23, which writes attention as `Attention` and keeps the LSTMs as `LSTM`, folded by its default optimization, into `/scratch`, and the step refuses an export that fixed an axis it was given as dynamic, the batch and the time of `audio_signal` and the batch of `length`. Once the decoder and joint and `vocab.txt` are written, the step frees NeMo's model, and only then loads the raw encoder, so that the two never hold the weights at the same time, and saves it as `encoder-model.onnx`, with every tensor of 1 KiB or more in `encoder-model.onnx.data`, since ONNX Runtime cannot read the exporter's small shape constants from external data, and a single file would pass the 2 GB limit of a protobuf.
6. The decoder and joint of one prediction step are exported the same way, from NeMo's own example inputs, names and dynamic axes once its exportable modules are prepared, then folded with onnxscript's optimizer, which turns the weights of the LSTMs, which the exporter builds at run time, into plain initializers, into `decoder_joint-model.onnx`.
7. Before each file is saved, the step strips the `metadata_props` of every node, in the graph, its subgraphs and its functions: the exporter records there the stack trace and the module of each node, with paths of the container, which a published file must not carry.
8. `vocab.txt` lists NeMo's pieces and `<blk>` last, one `<piece> <id>` per line.
9. ONNX Runtime's CPU provider runs the exported encoder on the 20 utterances, and the step writes the maximum absolute difference with the reference as the parity `encoder-onnx-against-nemo`, in `encoder-parity.json`, through the core's `domain/acceptance/parity_reports.py`, which refuses a difference that is not finite: Python's `max()` would drop a NaN that is not first. Its tolerance is 5e-5: the trial build measured 4.18e-6, and the tolerance leaves ten times that, while a wrong weight or a wrong graph differs by orders of magnitude more.

The export is the same to the byte from one run to the next under `-I`, without a fixed `PYTHONHASHSEED`.

| Row | Image | Memory MiB | Processes | CPUs | `/tmp` MiB | `/scratch` MiB | Reason of the limits |
|---|---|---|---|---|---|---|---|
| `export` | `export` | 14 336 | 128 | 8 | 512 | 3 072 | the trial build's export peaked at 11.1 GiB, its `/scratch` included, 66 processes and 2.37 GiB of `/scratch` |

It binds Moondream's tokenizer and weights, the `.nemo`, and FLEURS fr's index and audio, the files it reads. Its outputs:

| Output | Holds |
|---|---|
| `encoder-model.onnx` | The encoder's graph |
| `encoder-model.onnx.data` | The encoder's external data |
| `decoder_joint-model.onnx` | The decoder and joint of one prediction step |
| `vocab.txt` | The vocabulary of Parakeet TDT 0.6B v3 |
| `encoder-parity.json` | The parity of the encoder against NeMo's |

**The variants.** One FP32 graph serves every variant: the exported encoder is the conversions' input, whose 265 products of a constant weight the inspection holds to a plain 2-D initializer, and the decoder and joint stays in float32 for every variant. `domain/variants/variants.py` declares them:

| Variant | Encoder | Conversion | Devices |
|---|---|---|---|
| `float32` | `encoder-model.onnx`, with its external data `encoder-model.onnx.data` | none: it is the conversions' input | CPU, GPU |
| `float16` | `encoder-model.fp16.onnx` | `float16`, nothing kept in float32 beyond onnxconverter-common's own list | GPU |
| `bfloat16` | `encoder-model.bf16.onnx` | `bfloat16`, with the core's `Pad`, `Tile` and `Floor` kept in float32 | GPU |
| `eight-bit-blocks` | `encoder-model.int8.onnx` | `eight-bit-blocks`, in blocks of 64 symmetric weights, at the accuracy level 4, every one of the 265 products quantized, the 48 pointwise convolutions included | GPU, CPU |

float16 and bfloat16 serve the GPU, whose kernels compute in them, where ONNX Runtime on the CPU would convert them back to float32 around most nodes. Each converted encoder is a single file, of 1.24 GB in float16 and bfloat16 and 692 MB in 8-bit blocks, under the 2 GiB a protobuf may hold. The candidate holds, for each variant, its encoder, the decoder and joint of `export` copied under the variant's name, and `vocab.txt` from `export` and `config.json`, which the host writes, shared by every variant ([The published layout](#the-published-layout)).

**`reference`** is NeMo's own transcription of FLEURS fr and en, the reference that the export's transcripts are compared with, so that they check the export, the patch of the pointwise convolutions and the mapping of the weights together: the model built as for `export`, unpatched, on the CPU in float32, so that no TF32 enters, called directly, without `transcribe()` and its data loader: the preprocessor, the encoder, then NeMo's greedy decoding of TDT with at most 10 symbols per frame, as onnx-asr's `max_tokens_per_step`. Its one shard writes each utterance's token ids in `lines.jsonl`, and, for the check against Transformers, the features, the encoder's outputs and the joint's logits on the first 5 greedy steps of the first 5 utterances of FLEURS fr, in `reference.safetensors`, with the frames and the tokens emitted before each step in `reference.json`.

**`transformers-reference`** checks Moondream's weights in Transformers 5.22.1, which reads them as Moondream publishes them, with its own names and its own `ParakeetForTDT`, apart from the mapping onto NeMo's names that the export applies: a weight that the mapping misplaced, between two tensors of the same shape, would pass the export's parity, which compares ONNX with NeMo, but not this one. Its image, `mostt-model-builder-parakeet-tdt-transformers-reference`, installs the extra `parakeet-tdt-transformers-reference`: Transformers, torch, safetensors, numpy and librosa, which its feature extractor needs, each pinned exactly. It reads Moondream's three files and v3's three processor files, each bound by itself and linked under its own name into one directory of the container's `/tmp`, from which Transformers loads the model in float32 and its processor, offline; the `.nemo` never reaches it. It binds `reference.safetensors` and `reference.json` of `reference`, each by itself, and replays them, so that both frameworks start from the same numbers and NeMo runs once:

1. Transformers' encoder reads NeMo's features of each of the 5 utterances, and the step writes the maximum absolute difference with NeMo's encoder outputs, over the encoded frames, as the parity `encoder-transformers-against-nemo`, in `transformers-encoder-parity.json`. Its tolerance is 1.5e-5: the trial build measured 1.34e-6.
2. Transformers' decoder reads the blank then the tokens NeMo emitted before each of the 5 steps of each utterance, and its joint reads that output with NeMo's encoder outputs at the step's frame, projected by Transformers. NeMo's joint applies one `log_softmax` over its whole output, the 8 193 tokens and the 5 durations together, which Transformers' joint does not, so their raw logits differ by a constant per head: each head is compared once it is log-softmaxed on its own. The step writes the maximum absolute difference over the 25 steps as the parity `joint-transformers-against-nemo`, in `transformers-joint-parity.json`. Its tolerance is 5.3e-4: the trial build measured 5.3e-5. Both parities go through the core's `domain/acceptance/parity_reports.py`, which refuses a difference that is not finite.
3. Transformers transcribes FLEURS fr with `generate()`, from its own feature extractor, and its one shard writes each utterance's token ids in `lines.jsonl`. `generate()` gives the blank tokens besides the others, led by the decoder's start token, the blank, so `domain/transcripts/generated_sequences.py` keeps every id below the blank, in order, `<unk>` included, which `generate()` keeps as id 0 where only a decoding that skips the special tokens would drop it, drops every blank, and refuses an id above the blank, which no token has. Its transcripts are compared with NeMo's on fr, reported only, since the two feature extractors differ.

The step refuses a dump whose tensors are not exactly those its steps name. `recipes/parakeet-tdt/src/mostt_recipe_parakeet_tdt/domain/export/reference_dump.py` holds the layout of the dump, which `reference` writes and this step reads.

**`transcribe`** transcribes FLEURS fr and en with onnx-asr 0.12, from the candidate bound read-only, under onnx-asr's names: on the GPU, in the image `transcribe-gpu`, with ONNX Runtime's CUDA provider, reached through CDI's `nvidia.com/gpu=all`; on the CPU, in `transcribe-cpu`. Greedy, one utterance at a time, after 3 warm-ups that are not timed; each shard writes, for each utterance, the pieces of onnx-asr's `with_timestamps()` and the decoding time. `domain/variants/candidate_layout.py` alone maps a variant to onnx-asr's suffix: none for `float32`, `fp16`, `bf16` and `int8` for `float16`, `bfloat16` and `eight-bit-blocks`.

| Run | Step | Device | Variant | Shards | Languages |
|---|---|---|---|---|---|
| `nemo` | `reference` | CPU, 8 threads | `reference` | 1 | fr, en |
| `transformers` | `transformers-reference` | CPU, 8 threads | `reference` | 1 | fr |
| `float32-cpu` | `transcribe` | CPU, 2 threads each | `float32` | 4 | fr, en |
| `float32-gpu` | `transcribe` | GPU, 8 threads | `float32` | 1 | fr, en |
| `float16-gpu` | `transcribe` | GPU, 8 threads | `float16` | 1 | fr, en |
| `bfloat16-gpu` | `transcribe` | GPU, 8 threads | `bfloat16` | 1 | fr, en |
| `eight-bit-blocks-gpu` | `transcribe` | GPU, 8 threads | `eight-bit-blocks` | 1 | fr, en |
| `eight-bit-blocks-cpu` | `transcribe` | CPU, 2 threads each | `eight-bit-blocks` | 4 | fr, en |

| Row | Image | Memory MiB | Processes | CPUs | `/tmp` MiB | `/scratch` MiB | Reason of the limits |
|---|---|---|---|---|---|---|---|
| `reference` | `export` | 8 192 | 128 | 8 | 512 | 1 024 | NeMo's reference peaked at 6.2 GiB, its `/scratch` included, and 58 processes |
| `transformers-reference` | `transformers-reference` | 7 168 | 128 | 8 | 512 | — | the check in Transformers peaked at 5.4 GiB and 79 processes |
| a shard of the GPU | `transcribe-gpu` | 5 120 | 128 | 8 | 256 | — | a transcription on the GPU peaked at 3.4 GiB and 99 processes |
| a shard of the CPU | `transcribe-cpu` | 5 120 | 64 | 2 | 256 | — | a transcription on the CPU, one process of 2 threads per container, peaked at 3.4 GiB and 33 processes |

**The readers.** Every reader renders the text from the token ids, so that NeMo's ` ⁇ `, onnx-asr's `<unk>` and Transformers' id 0 never differ: `domain/transcripts/run_readers.py` reads the ids of NeMo and of Transformers, and maps onnx-asr's pieces back to ids through the candidate's `vocab.txt`, with `▁` written as a space; `domain/transcripts/token_rendering.py` joins the pieces, writes `▁` as a space and `<unk>` as itself, and counts `<unk>` from the ids, refusing the blank and an id beyond the vocabulary. `domain/transcripts/vocabulary.py` accepts `vocab.txt` only at the SHA-256 of v3's vocabulary, with `<unk>` first, `<blk>` last, its indexes in order, no piece twice and no raw space in a piece, so that the mapping of pieces back to ids is one to one.

**Comparisons.** `domain/variants/comparisons.py` holds the recipe's comparisons, each on fr and en, but Transformers', on fr alone:

| Run | Against | Rule | Reason |
|---|---|---|---|
| `nemo` | `float32-cpu` | equivalence: identical token ids on every utterance, and the interval within ±0.10 point | NeMo and onnx-asr, both in float32 on the CPU, gave the same ids on every utterance of fr and en in the trial build: any difference is a fault of the export, of its patch or of the mapping of the weights |
| `float32-gpu` | `float32-cpu` | equivalence: identical token ids on at least 99 % of the utterances, and the interval within ±0.10 point | CUDA's kernels sum in another order than the CPU's, which moved the ids of 0.6 % of the utterances in fr and 0.3 % in en, without changing the rate |
| `float16-gpu`, `bfloat16-gpu`, `eight-bit-blocks-gpu` | `float32-gpu` | non-inferiority: the upper bound of the interval at most +0.20 point | a variant may lose a fifth of a point at most against float32 on its device; each one was measured within 0.1 point, and the interval of the bench is a few tenths wide |
| `eight-bit-blocks-cpu` | `float32-cpu` | the same | the same |
| `transformers` | `nemo` | reported only | Transformers reads its features with its own extractor, apart from NeMo's |

A tenth of a point is a few words of the bench, 17 in fr's 17 075 and 14 in en's 14 467, so an equivalence admits no change that a transcript would show more than by chance.

**The model description.** `domain/publication/model_description.py` describes the published model, Parakeet Ultra in ONNX, for the card, whose prose lives in the recipe's template, `model-card.md`: a quantization of `moondream/parakeet-ultra` in the Hub's terms, under CC BY 4.0, with Moondream's Parakeet Ultra and NVIDIA's Parakeet TDT 0.6B v3 as its upstream works, each at the revision its pins give, and the 25 languages that both their cards declare, of which FLEURS measures French and English alone. Neither card, at its pinned revision, gives a copyright notice, a requested form of attribution or a disclaimer, so the template keeps no notice of theirs. The template also says what the build changed, how it built the files, their limitations and how onnx-asr loads them. The description names onnx-asr and ONNX Runtime's two packages, `onnxruntime` and `onnxruntime-gpu`, as the libraries that measured its variants; it shows the 8-bit blocks as `int8`, onnx-asr's name for them, and float16 and bfloat16 with a note that they serve the GPU, since ONNX Runtime computes most of them in float32 again on the CPU; it names the run `nemo` the original model in NeMo; and it gives the template the revisions of FLEURS and of both upstream works.
