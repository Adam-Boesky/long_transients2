#!/bin/bash
# Filter one field per array task (python -m Source_Analysis.filter_field) from its tables under $LT_DATA/fields/<field>/.
#   FIELDS="345 370" sbatch --array=0-1 jobs/filter_field.sh
#   FIELD=345 sbatch -p test --mem=96G jobs/filter_field.sh      smoke test / memory measurement
#
#SBATCH -J filter_field
#SBATCH -p shared
#SBATCH -c 1
#SBATCH --mem=48G
#SBATCH -t 0-04:00
#SBATCH -o /n/holystore01/LABS/berger_lab/Users/aboesky/long_transients2/logs/%x_%j.out
#SBATCH -e /n/holystore01/LABS/berger_lab/Users/aboesky/long_transients2/logs/%x_%j.err
set -uo pipefail
source "${SLURM_SUBMIT_DIR:-.}/jobs/env.sh"

if [ -n "${SLURM_ARRAY_TASK_ID:-}" ] && [ -n "${FIELDS:-}" ]; then
    FIELD=$(echo ${FIELDS} | tr ' ' '\n' | sed -n "$((SLURM_ARRAY_TASK_ID + 1))p")
fi
python3 -u -m Source_Analysis.filter_field ${FIELD:?set FIELD or FIELDS}
