---
name: pull-request
description: Opens and updates the pull request of a branch in this repository, and reviews someone else's pull request. Use when opening or updating a pull request, retargeting it to another base, or reviewing someone else's pull request.
---

# Pull request

`CLAUDE.md` > Workflow holds the rules this procedure applies, and, for every section but the review of someone else's pull request, `.claude/skills/deliver-branch/SKILL.md` > Deliver a branch says what `<base>` is, how to write a file whose text you write, and how a session that did not start in the branch's worktree runs each command below.

## Open the pull request

The branch is shaped, reviewed and pushed first, and its CI watched once the pull request is open, as `.claude/skills/deliver-branch/SKILL.md` > Shape the history, > Review and > Push the branch and watch its CI say.

1. Write the body from `.github/pull_request_template.md`, which `gh pr create --body-file` does not apply: fill each section in place of its comment, and tick a checklist item only once it holds.
2. Open the pull request against `main`, or against the base it is stacked on: write its title with the Write tool to a file in a directory that `mktemp -d` creates outside the worktree, then run `gh pr create --base <branch> --title "$(cat <title file>)" --body-file <file>`, since `gh` reads no title from a file and a title typed between double quotes runs its backquoted text. Without a terminal, `gh` needs both the title and the body. Remove the title file's directory once the pull request is open.

## Update the body

The body and the base of an open pull request change through GitHub's API: update the body from its file with `gh api -X PATCH 'repos/{owner}/{repo}/pulls/<number>' -F body=@<file>`, and retarget the pull request to another base with `gh api -X PATCH 'repos/{owner}/{repo}/pulls/<number>' -f base=<branch>`, where `gh` fills in `{owner}` and `{repo}` from the checkout: `gh pr edit`, with `--body-file` or `--base`, fails on a repository that still carries GitHub's retired Projects (classic).

## Review someone else's pull request

From the main checkout, run `/expert-review <number>`. The skill checks the pull request out in a temporary directory outside the repository, so it never touches a session's worktree.
