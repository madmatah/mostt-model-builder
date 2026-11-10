## Purpose

<!-- Why this pull request exists: the need or the problem, in a few sentences. -->

## Proposal

<!-- The solution in broad strokes: what it is made of, not how each line works. If the branch is stacked on another, name its pull request. -->

## How to test

<!-- Steps for the reviewer: what to run, what to try, what to look for. -->

<details>
<summary>How it was tested</summary>

<!-- What the author ran and observed: the tests added, the manual checks, captures of anything visible. -->

</details>

## Checklist

<!-- Tick an item only once it holds. -->

### Quality

- [ ] `scripts/check-commit.sh` passes on every commit, and the whole gate ran locally too when the change touches the gate itself or what the CI cannot run, as `CLAUDE.md` > Workflow, "Run each check once, at its level" says.
- [ ] `/expert-review` ran once on the completed branch, and again on the fixes of its blocking findings, when the first fixer fixed any, and this pull request names the fixes no expert checked; every finding is fixed or answered here; when the branch changes the skills or agents that review or fix it, this pull request names the definitions that reviewed it, and says when they are not the branch's.

### History

- [ ] Each commit is atomic, and its message follows `.claude/skills/writing-commits/SKILL.md` > The rules of a commit message: a header as "A commit header follows the commit style" says, a body that says why unless the change speaks for itself, and a footer as "A commit footer holds only what tools read" says; no `fixup!`, `amend!` or `squash!` commit is left.

### Documentation

- [ ] The documents describe the new state: `docs/foundations.md`, `docs/model-build.md`, `README.md`, `CLAUDE.md`, `.claude/`.
- [ ] Every external work brought in, including a library shipped in a build, is credited in `CREDITS.md`.

### Hygiene

- [ ] No secret, credential, `.env` file or personal data is committed, apart from what credits a work.
- [ ] No project-management material is committed: status, roadmap, decisions, handoffs, notes, `TODO`, nor where that material lives.
- [ ] No file is committed by accident: build output, logs, temporary captures, model weights, datasets or exported models.
