# Recall qualification report

**Status: BLOCKED.** The requested native inference qualification did not
launch. The measured available memory was 3,421.1 MiB against the existing
6,144 MiB preflight gate. The model was not loaded and zero tokens were
processed.

Recall did not run its live system integration checks, create or restore a
model checkpoint, or claim parallel execution. All four pre-existing
worktree modifications were preserved on `copilot/stack-qualification`.
Nothing was committed or pushed. See `100K_QUALIFICATION_REPORT.md` and the
structured evidence files in this directory for details.
