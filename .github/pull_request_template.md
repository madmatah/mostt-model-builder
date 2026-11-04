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

- [ ] `scripts/check-commit.sh` passes on every commit, and `scripts/check.sh` on the last one.
- [ ] `/expert-review` ran on the completed branch and on each round of its fixes, or this pull request names the fixes no expert reviewed; every finding is fixed or answered here; when the branch changes the skills or agents that review or fix it, this pull request names the definitions that reviewed it, and says when they are not the branch's.

### History

- [ ] Each commit is atomic, and its message follows `CLAUDE.md` > Workflow: the header of "Commits", a body that says why unless the change speaks for itself, and a footer as "A commit footer holds only what tools read" says; no `fixup!`, `amend!` or `squash!` commit is left.

### Documentation

- [ ] The documents describe the new state: `docs/foundations.md`, `docs/model-build.md`, `README.md`, `CLAUDE.md`, `.claude/`.
- [ ] Every external work brought in, including a library shipped in a build, is credited in `CREDITS.md`.

### Hygiene

- [ ] No secret, credential, `.env` file or personal data is committed, apart from what credits a work.
- [ ] No project-management material is committed: status, roadmap, decisions, handoffs, notes, `TODO`, nor where that material lives.
- [ ] No file is committed by accident: build output, logs, temporary captures, model weights, datasets or exported models.
