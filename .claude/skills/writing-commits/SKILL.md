---
name: writing-commits
description: Writes the commit messages of this repository, so that each one says what its change achieves and why it was made, in Conventional Commits. Use when committing, amending or rewording a commit, splitting work into commits, shaping a branch's history before pushing it, or checking the commit messages of a branch.
---

# Writing commits

`CLAUDE.md` > Workflow states the rules of a commit: "Commits", "A commit scope names where the change belongs", "A commit message says why" and "A commit footer holds only what tools read". This skill applies them to each commit that will be pushed, whether you write it at once or while you rewrite the branch before pushing. A work-in-progress commit or a `fixup!` commit needs none of it, since the rewrite replaces it, but the message that an `amend!` commit carries below its first line follows it all, since it replaces the message of its commit. Where a rule leaves a choice, the sections below say how to make it. Write from the rules, not from `git log`, whose older messages may predate them.

## Draw the commits

Each commit holds one reason to change ("Commits"), so that a reader can review it, revert it or find it with `git bisect` on its own.

- **A subject that needs an "and"** usually holds two commits. Two unrelated fixes made the same afternoon are two commits.
- **The test that pins a fix** goes in the commit of the fix, since every commit passes its own checks ("Commits").
- **A correction to an earlier commit of the branch** is folded into it, as `.claude/skills/develop/SKILL.md` > Shape the history shows. The history describes the change that ships, not the path that found it.

## Choose the type

Take the type of the commit's dominant intention. When two seem to fit:

| The change | Type | Not |
|---|---|---|
| Restores the behavior that was intended | `fix` | `feat`, even when the fix adds code |
| Adds behavior that a user or a caller can observe | `feat` | `fix` |
| Restructures, renames or moves code without changing its behavior | `refactor` | `perf`, unless a measurement motivates it |
| Makes the code faster or leaner, as a measurement shows | `perf` | `refactor` |
| Draws a module boundary or a layer, or adds a module | `refactor`, or `feat` when it brings new behavior | `chore` |
| Adds or extends the tests of behavior that already works | `test` | `fix`, whose commit carries the test that pins it |
| Documents, guides and agent guidance | `docs` | `chore` |
| The CI workflows | `ci` | `chore` |
| Undoes an earlier commit | the type of what the undoing does, such as `fix` when the undone commit broke something | `chore` |
| The build, the dependencies, the tools' configuration, and the repository's own files that are not documents, such as `.gitignore` | `chore` | `ci`, `docs` |

A breaking change keeps the type of what it does, and "Commits" says how its header shows it.

## Choose the scope

Take the scope as `CLAUDE.md` > Workflow, "A commit scope names where the change belongs" says. Where it leaves a doubt:

- **A document that lives in a module,** such as its guide, takes the module's scope, not `docs`.
- **A test** takes the scope of the code it pins.
- **A dependency** takes the scope of the module whose manifest declares it, or `build` for the root's.
- **A new module** takes its own name, the one its new row of `docs/foundations.md` > Layers and modules gives it.
- **A rule changed together with the skills, agents and documents that follow it** takes `rules`: the rule owns the decision.

## Write the subject

Picture a reader scanning `git log --oneline` for the commit that changed a behavior. The subject is what they read, and it decides whether they open the commit. Tell them what the change achieves, which only you know: the diff already shows the mechanism and the files. What does not fit goes in the body. A revert gets a header of this form too, not the `Revert "…"` that `git revert` writes.

💩 The mechanism, or nothing at all:

```
fix(adapters): fix download bug
fix(adapters): add a retry around fetch_file
feat(domain): update source_pins.py and work_directory.py
fix(adapters): Fixed the timeout error.
```

❤ The intent:

```
fix(adapters): never keep a truncated download as a fetched source
fix(domain): say which pin broke instead of rejecting the whole file
feat(parakeet-tdt): export the encoder with fused attention
refactor(domain): compute a file's hash in one place instead of three
```

## Write the body

See the body as a brain dump of your state of mind at the moment you commit. Picture someone asking you, six months from now: "Why did you do this?" You would not answer with the list of the files you edited. You would give the context that led you there, the context you will have forgotten by then. Without it, the change gets misread, criticized, or reverted by someone who could not see what it protected against. That someone is most likely you.

Answer the questions that reader would ask:

- What broke, or what constraint appeared? What did the previous code assume, and when did that stop holding?
- Which alternative was tempting, and why did you turn it down? Nothing else in the repository records a rejected approach.
- Which facts from outside the code does the change rest on: the line of a datasheet or a standard, a measured value, how an external service behaves? No reader can recover them from the code.

Then reread the body: a sentence that the diff could have told is one to cut.

Under `fix(adapters): never keep a truncated download as a fetched source`:

💩 The diff, told again:

```
Download to a temporary name and check the size and the hash before
renaming. Update the fetch tests accordingly.
```

❤ The context that led there:

```
A server can close the connection in the middle of a large file without
an error. The short file kept its final name, so the next fetch took it
for done, and the build failed much later, in the export, on a
truncated checkpoint.

A file now takes its final name only once its size and its hash match
its pin. A retry on a transfer error would not have caught it: the
transfer reported no error.
```

## Add the footer

Write it as `CLAUDE.md` > Workflow, "A commit footer holds only what tools read" says, and never on a guess: a breaking change only when its users will meet it, an issue only when this commit closes it.

```
feat(domain)!: report word error rates as fractions

The verification report gave percentages rounded to two decimals, so
two variants within 0.005 point of each other read as equal, and the
acceptance compared rounded figures. A fraction keeps every digit that
the bench measured.

BREAKING CHANGE: `word_error_rate` in the verification report is a
fraction between 0 and 1, no longer a percentage. Multiply it by 100
to show it.
Closes #42
```

## Check the message

Before committing, read the message against the rules:

1. **The header** ("Commits"): one line, in the form of the project's style, with the type of [Choose the type](#choose-the-type) and the scope of [Choose the scope](#choose-the-scope).
2. **The subject** ("A commit message says why"): what the change achieves, not the files it touches, in the mood, the case, the punctuation and the length that the rule gives.
3. **The body** ("A commit message says why"): laid out as the rule says, it says why and lists nothing that the diff shows. Without one, the change really speaks for itself.
4. **The footer** ("A commit footer holds only what tools read"): what the rule requires, nothing that it excludes, and nothing untrue.
5. **The checks:** once committed, the commit passes what "Commits" asks of it, `scripts/check-commit.sh`, which checks the last commit. When you rewrite a branch, `.claude/skills/develop/SKILL.md` > Shape the history checks every commit at once, and the last one on the whole gate.
