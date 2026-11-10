# The repository

## What belongs in the repository

- **The repository holds the current state of the project, and nothing else.** Never commit project-management material, which lives outside it:
  - status or tracking files;
  - decision records;
  - roadmaps;
  - plans;
  - handoffs;
  - meeting or research notes;
  - the task markers `TODO`, `FIXME`, `XXX` and `HACK`, in a comment or a document.
- **No file says where project-management material lives or how it is organized,** which is each contributor's own business.
- **No secret, credential, `.env` file or personal data:** no token, such as a Hugging Face or PyPI token, no name, no e-mail address, no path into a home directory and no machine-specific identifier, except the name of a rights holder in a license notice copied verbatim, as [External work](#external-work) asks, and the name of whoever made a work that `CREDITS.md` credits or that `docs/foundations.md` > References cites, beside that work, since crediting a work is giving its maker's name, and the public URL of this repository, where a published file links the code that built it. A test or a document that needs one uses a placeholder, such as `<user>` or `~`.
- **No file committed by accident:** build output, logs, temporary captures, and the files a build fetches or writes, such as model weights, datasets, exported models and their reports, which live in a work directory outside the clone.
- **A change updates every document that describes what it changes, in the same change:** the rules file of a rule, `docs/foundations.md` for the design and its reasons, `docs/model-build.md` for the detail of the build, the skill of a procedure, `README.md` for setup, usage and scripts, and `CLAUDE.md` for how a session works.

## External work

- **Credit every external work the project reuses in `CREDITS.md`, in the same change that brings it in.** This covers code or algorithms ported or adapted from another project, data taken from someone else's work, assets such as sounds or images, and libraries shipped in a build. The work of the mostt project is not external: its author is this project's, who may reuse it under any terms, so it needs no entry.
- **Each entry says** what is reused, where it lives in the repository, where it comes from and under which terms. Guidance that draws on another work's ideas or wording here and there, rather than copying a file or a passage, is credited as a whole, such as "parts of the agent guidance in `.claude/` draw on …", without listing each borrowing. When a license asks for its notice to be kept, copy the notice verbatim.
- **The works the design comes from are named,** with their links, in `docs/foundations.md` > References.
