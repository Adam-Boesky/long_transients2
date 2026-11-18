# Cluster jobs

One Slurm script per Python entry point, named after it: `jobs/fetch_ztf_products.sh` runs
`tools/fetch_ztf_products.py`, `jobs/build_field_tables.sh` runs `python -m Extracting.field_tables`. Submit from
the repo root on Cannon with nothing loaded in the submitting shell; every script sources `jobs/env.sh` itself.

Each script has, in order: a comment header with example `sbatch` lines and the environment variables it reads
(parameters are passed as `NAME=value sbatch ...`, never edited into the script); `#SBATCH` lines sized to what the
job measured (`seff`), logging to `$LT_DATA/logs/<name>_<jobid>.out`; `set -uo pipefail`; `source jobs/env.sh`;
one command. `env.sh` loads the toolchain and the `long_transients2` env, enters the repo, and exports `LT_DATA`,
`PS1_STACK_DIR` and single-thread BLAS.

Patterns: one field per array task (`FIELDS="..." sbatch --array=0-N`); `--dependency=afterany:<ids>` to start
after other jobs; `-J <name> --dependency=singleton` for a resumable campaign longer than the 3-day limit (N copies
run one after another). Smoke-test a new script on `-p test` before an array.
