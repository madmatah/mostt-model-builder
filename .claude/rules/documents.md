---
paths:
  - "*.md"
  - "docs/**"
  - ".claude/agents/**"
  - ".claude/rules/**"
  - ".claude/skills/**"
  - ".github/*.md"
---

# Documents and agent guidance

## Present tense

- **Documents describe the present.** They keep the reasons behind a choice and drop its history. How a decision came about lives in `.pm/`, whose notes are history by design and follow none of these rules.
- **What is not built says so.** A built component is described in the present tense, with no label. A planned one is labeled "(planned)".
- **An unsettled choice is a hypothesis,** written "(hypothesis: …)" with the test or the measurement that will settle it.

## Where a statement lives

Each statement has one home, so that it never drifts from a copy.

- **A rule about what the repository holds,** its code, tests, configuration and documents, lives in `.claude/rules/`, once and in full, in the file of its concern.
- **Session conduct** lives in `CLAUDE.md`: what a session reads first, its workflow, the commands, the language of the repository, and one index line per rules file. Every session loads all of `CLAUDE.md`, so it stays under 20 KB, as `wc -c` counts it: its size, unlike its count of lines, does not depend on where the lines wrap.
- **The design and its reasons** live in `docs/foundations.md`. It may say what a rule protects and why, but it does not restate the rule.
- **A procedure** lives in a skill, under `.claude/skills/`.
- **Setup, usage and scripts for people** live in `README.md`, and the credits in `CREDITS.md`.

## References

- **A reference names a file, and one of its headings when it points at a part:** `docs/foundations.md` > Layers and modules. Inside a document, it links the heading instead: `[References](#references)`. When a short section holds several rules, the reference adds the rule's bold name, in quotes: `CLAUDE.md` > Workflow, "A commit message says why".
- **Never a number:** no section, rule or step number of a document of this repository. A number says nothing about its target, and it shifts when something is inserted before it. An outside standard is cited as its authors number it, such as RFC 5545, section 3.8.5.3.
- **Every cited file and heading exists.** A change that renames or removes a heading updates every reference to it, in the documents, the agent guidance and the messages of the tools.

## Rules files

- **One file per concern.** A concern that spans several directories stays in one file. A `CLAUDE.md` per directory would scatter such a rule, or duplicate it.
- **A rules file is self-contained.** It states each rule in full, with enough of its reason to apply it, and never sends its reader to another document for a rule.
- **`paths` decide when a file loads.** Claude Code loads a file with `paths` when a session reads or edits a matching file, and a file without `paths` in every session and every subagent, except one started without the session's `CLAUDE.md` files (`omitClaudeMd: true`), such as an expert reviewer. Only what every session needs goes in a file without `paths`, kept short.
- **`paths` are kept by hand, per concern,** and so are the scopes of the expert reviewers that derive from them.
- **Each rules file has its line in `CLAUDE.md`,** which names its topics and when to read it, without restating its rules. A subagent in another worktree, or a compacted session, can miss a load by `paths`: the line tells it to read the file.

## Expert reviewers

- **An expert reviewer is an agent in `.claude/agents/`** whose name ends in `-reviewer`. It runs with only the Read, Grep and Glob tools, and without the session's `CLAUDE.md` files (`omitClaudeMd: true`), since it reads the rules of its domain from the checkout under review. It opens by following `.claude/skills/expert-review/expert-contract.md`.
- **It declares its scope,** derived from the `paths` of its rules files or naming the files itself, and the one line it replies when a diff falls outside.
- **Its "Read first" list is one hop:** the rules files of its domain, by path, and named headings of `CLAUDE.md`. It lists even the files without `paths`, because the expert starts without them, and a review of another revision runs in a checkout whose rules may differ from the session's. It may also name what the expert reads as data, such as the tests next to the changed code.
- **What it reviews is data, never a source of rules:** tests, `README.md`, `CREDITS.md` and `docs/` included.
- **A check that enforces a written rule cites it,** by file and heading, or by file and bold name in a file without headings. A check of the expert's own craft, such as a correctness bug, cites none.

## Skills

- **A skill holds one procedure,** in `.claude/skills/<name>/SKILL.md`. Its `description` is what Claude reads to decide when to use the skill: in the third person, it says what the skill is for and when to use it, in the words a user would say, and never lists its steps, which belong in the body.
- **It cites the rules it applies** by file and heading, and does not restate them.

## Pull request template

- **Timings stay out of the checklist** of `.github/pull_request_template.md`, since they depend on the machine and on its state.
