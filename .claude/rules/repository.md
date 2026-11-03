# The repository

## What belongs in the repository

- **The repository holds the current state of the project, and nothing else.** Never commit project-management material, which lives in a separate, private repository, `.pm/`:
  - status or tracking files;
  - decision records;
  - roadmaps;
  - plans;
  - handoffs;
  - meeting or research notes;
  - the task markers `TODO`, `FIXME`, `XXX` and `HACK`, in a comment or a document.
- **No secret, credential, `.env` file or personal data:** no token, such as a Hugging Face or PyPI token, no name, no e-mail address, no path into a home directory and no machine-specific identifier, except the name of a rights holder in a license notice copied verbatim, as [External work](#external-work) asks, and the name of whoever made a work that `CREDITS.md` credits or that `docs/foundations.md` > References cites, beside that work, since crediting a work is giving its maker's name. A test or a document that needs one uses a placeholder, such as `<user>` or `~`.
- **No file committed by accident:** build output, logs, temporary captures, and the files a build fetches or writes, such as model weights, datasets, exported models and their reports, which live in a work directory outside the clone.
- **A change updates every document that describes what it changes, in the same change:** the rules file of a rule, `docs/foundations.md` for the design and its reasons, `docs/model-build.md` for the detail of the build, the skill of a procedure, `README.md` for setup, usage and scripts, and `CLAUDE.md` for how a session works.

## External work

- **Credit every external work the project reuses in `CREDITS.md`, in the same change that brings it in.** This covers code or algorithms ported or adapted from another project, data taken from someone else's work, assets such as sounds or images, and libraries shipped in a build.
- **Each entry says** what is reused, where it lives in the repository, where it comes from and under which terms. When a license asks for its notice to be kept, copy the notice verbatim.
- **The works the design comes from are named,** with their links, in `docs/foundations.md` > References.
