#!/bin/bash
# One download worker for ZTF deep reference products (tools/fetch_ztf_products.py), resumable: resubmit freely.
# IRSA allows one transfer per client IP and Cannon NATs each rack family through its own address
# (holy7c -> .4 via -p shared, holy8a -> .5 via -p sapphire), so run one worker per partition; a second worker on
# an already-served IP exits at once (--one-per-ip).
#   PRODUCTS="refunc" sbatch -p shared   jobs/fetch_ztf_products.sh
#   PRODUCTS="refunc" sbatch -p sapphire jobs/fetch_ztf_products.sh
#   FETCH_ARGS="--fields 643 --limit 6" ...   smoke test;   FETCH_ARGS="--reverse" ...   walk the manifest backwards
# A campaign longer than the 3-day limit: submit N copies with -J <name> --dependency=singleton (they run in turn).
# Progress: python tools/fetch_ztf_products.py --products refunc --status
#
#SBATCH -J fetch_ztf_products
#SBATCH -c 1
#SBATCH --mem=2G
#SBATCH -t 3-00:00
#SBATCH -o /n/holystore01/LABS/berger_lab/Users/aboesky/long_transients2/logs/%x_%j.out
#SBATCH -e /n/holystore01/LABS/berger_lab/Users/aboesky/long_transients2/logs/%x_%j.err
set -uo pipefail
source "${SLURM_SUBMIT_DIR:-.}/jobs/env.sh"

python3 -u tools/fetch_ztf_products.py --products ${PRODUCTS:-refpsfcat} --claim --one-per-ip ${FETCH_ARGS:-}
