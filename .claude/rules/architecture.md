---
paths:
  - "src/**"
  - "tests/**"
  - "recipes/**"
  - "pyproject.toml"
---

# Architecture

The project follows Clean Architecture, in four layers, and applies SOLID to modules, functions and types.

## Layers

- **Four layers, from the inside out:** the domain, the application, the adapters and the composition roots. A source file's layer is the directory it lives in, within the package `mostt_model_builder` in `src/`: `domain/`, `application/` and `adapters/`, with the composition roots in their own places: `bootstrap.py`, which puts the packages on the path and runs an entry module, on the host and in every step; `host_main.py`, the command line on the host; and `steps/`, the entry modules of the core's container steps. A recipe, in `recipes/<family>/src/mostt_recipe_<family>/`, holds a domain of its own in `domain/`, its `recipe.py`, the plain data it gives the core, and the entry modules of its container steps in `steps/`.
- **Dependencies point inward.** An inner layer never refers to an outer one: no import, no type, no name. The domain depends only on the domain, the application on the domain, the adapters on the application and the domain, and the composition roots on every layer. A library of plain values that does no I/O and reads no clock, such as a decimal or a date library, may serve any layer. Check: `lint-imports`, from the layers contract of `pyproject.toml` and the contract of each layer, which names the outer layers it may not import.
- **No dependency cycle,** between modules, or between the directories that stand for them (`.claude/rules/configuration.md` > Modules), those of the same layer included. Inside a module, no cycle either between its files, since Python imports file by file. A path that only passes through a package to reach another is no dependency; using a name defined in a file is one.

## Ports and wiring

- **Inner layers own their ports.** When the application needs the outside world, it declares an interface in its own vocabulary, a `Protocol`, and an adapter implements it. The application reaches a technology only through that port.
- **Only composition roots wire.** `host_main.py`, the entry module of each container step, each of which reads its request as an adapter does and wires what it runs, and the tests choose and connect the concrete adapters, by hand: written by hand, the graph stays in one place and checked by the types, where a DI container would resolve it at run time, by tokens. No DI container, no service locator, no singleton, and no global or module-level mutable state shared between callers.
- **Plain data crosses boundaries:** values whose types belong to the inner side. No framework or platform type crosses a port.
- **A module's public API is its single entry point.** Everything else is internal, and nothing reaches inside another module. Inside a module, a file is used only through the names it makes public, never through a name that starts with an underscore. A directory adds no entry point of its own: no `__init__.py` re-exports the names of a directory's files, apart from the package's own, `mostt_model_builder/__init__.py`.

## Responsibilities

- **The domain holds the rules and is pure:** no I/O, no clock, no global mutable state, no framework, no platform API. Check: `lint-imports` keeps out of it the modules of the standard library that reach the outside world, `subprocess`, `socket`, `urllib`, `http`, `shutil`, `tempfile` and `asyncio`; review, the rest.
- **The application orchestrates.** It passes the inputs to the domain, calls its ports and assembles the result. Any rule it needs belongs in the domain.
- **Adapters translate, nothing more.** An adapter converts between a technology and an inner layer, through a port or through the plain data the inner layer defines, and is named after that technology. It holds no business rule.
- **Composition roots decide no rule either.** They wire the adapters, and turn outside input into the requests of the inner layers.

## Stored and shared data

- **What is stored, sent, or crosses a thread or a process is plain data:** values in the types of the inner side that survive a round trip through their serialized form unchanged, with no handle, no closure, and nothing the serialization drops or alters. It then saves, travels and replays as it is. Elsewhere, a type may hold behavior. The project's data of this kind: the pins of the sources; the request the host writes for a container step, such as the files of the FLEURS check or the shard of a run; the reports a step writes back, such as the FLEURS check's durations, the inspection's report of the candidate's graphs or the placement of a run's nodes; the parity reports, such as the export's against NeMo; the lines of each run and the hypotheses its reader gives; the peaks of each step; and the candidate files with their hashes.

## SOLID

- **Single responsibility.** A function or a type has one reason to change, a file holds one concept, a directory within a layer, such as `domain/acceptance/`, holds one concern, which for a directory of the adapters' directory is its technology. A function whose name needs an "and" does two things. When a change to one rule forces edits across unrelated concepts, the responsibilities are mixed.
- **Open for extension.** A new case is added as data, or beside the existing cases, without rewriting the code that handles them: a new model family is a new recipe, and changes the core only where the contract between them needs it.
- **Substitutable implementations.** Every implementation of a port, the fakes of the tests included, honors its whole contract: units, ordering, errors, missing values. The application behaves the same on a fake and on the real adapter.
- **Narrow ports.** A port offers only what its client uses, in the client's vocabulary, and nothing depends on methods it does not call.
- **Dependency inversion.** Policy depends on the abstractions it owns, the ports of the inner layers, never on a concrete technology.
- **No speculative generality:** no abstraction with a single caller and no rule behind it.

## Organization

- **Folders, files and modules are named after domain concepts,** never after technical kinds: `utils`, `helpers`, `services`, `models` or `common`, nor their singulars.
- **Where the code lives:** the core is the package `mostt_model_builder`, in `src/`; each recipe lives in `recipes/<family>/`, named after the model family it builds, with its package `mostt_recipe_<family>` in its `src/` and its tests in its `tests/`. Their casing belongs to `conventions.md`.
