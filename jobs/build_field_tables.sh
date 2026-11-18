#!/bin/bash
# One field's source tables (python -m Extracting.field_tables) per array task.
#   FIELDS="293 294 345" sbatch --array=0-2 jobs/build_field_tables.sh      task i builds the i-th field of FIELDS
#   FIELD=860 sbatch -p test jobs/build_field_tables.sh                     one field (smoke test: ~12 min, 13 GB)
#   REQUIRE="refunc" ...   fail instead of falling back when an optional product is missing
#   ... --dependency=afterany:<fetch job ids>   start once the downloads have ended
#
#SBATCH -J build_field_tables
#SBATCH -p shared
#SBATCH -c 8
#SBATCH --mem=20G
#SBATCH -t 0-02:00
#SBATCH -o /n/holystore01/LABS/berger_lab/Users/aboesky/long_transients2/logs/%x_%j.out
#SBATCH -e /n/holystore01/LABS/berger_lab/Users/aboesky/long_transients2/logs/%x_%j.err
set -uo pipefail
source "${SLURM_SUBMIT_DIR:-.}/jobs/env.sh"

if [ -n "${SLURM_ARRAY_TASK_ID:-}" ] && [ -n "${FIELDS:-}" ]; then
    FIELD=$(echo ${FIELDS} | tr ' ' '\n' | sed -n "$((SLURM_ARRAY_TASK_ID + 1))p")
fi
python3 -u -m Extracting.field_tables ${FIELD:?set FIELD or FIELDS} --n-jobs ${SLURM_CPUS_PER_TASK:-1} ${REQUIRE:+--require ${REQUIRE}}
