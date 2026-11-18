#!/bin/bash
# Gaia DR2 G < 16 stars for the bright-star mask, one parquet per field (tools/prefetch_bright_stars.py); resumable.
#   sbatch jobs/prefetch_bright_stars.sh                        the whole footprint (~485 archive queries)
#   FIELDS="345 370" sbatch -p test jobs/prefetch_bright_stars.sh
#
#SBATCH -J prefetch_bright_stars
#SBATCH -p shared
#SBATCH -c 1
#SBATCH --mem=4G
#SBATCH -t 1-00:00
#SBATCH -o /n/holystore01/LABS/berger_lab/Users/aboesky/long_transients2/logs/%x_%j.out
#SBATCH -e /n/holystore01/LABS/berger_lab/Users/aboesky/long_transients2/logs/%x_%j.err
set -uo pipefail
source "${SLURM_SUBMIT_DIR:-.}/jobs/env.sh"

python3 -u tools/prefetch_bright_stars.py ${FIELDS:+--fields ${FIELDS}}
