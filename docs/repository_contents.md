# Repository contents

Keep source code and bundled TOV data, tests, scripts, the manual, root-level example inputs, six model templates, and one English notebook per stage: the workflow overview, quarkyonic–baryquark comparison, and six-model beta-equilibrium study.

The reference tables in `results/` support the workflow overview. Selected `Paper/` data support the six-model calculations and the beta derivative comparison. Saved reference results are historical: their sound speeds can use the previous polynomial method. The beta comparison explicitly retains both methods.

Generated figures, reports, density checkpoints, exploratory notebooks, duplicate notebook copies, generated input sweeps, and backups are local research files. `.gitignore` excludes them from new commits. They remain on disk; ignoring a file is not a backup.

Notebook outputs are omitted from version control. Full local copies from this cleanup, including embedded figures, are in `.local/repo-cleanup-2026-10-07/`. No physics code was changed by this cleanup.

The fixed-parameter beta notebook reuses compatible saved curves, which can retain historical thermodynamics. Use the opening comparison cells in the stage-3 notebook to recompute saved beta derivatives.
