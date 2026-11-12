#!/bin/bash
# One cooperating download worker. IRSA allows one transfer per client IP, and Cannon NATs each rack
# family through its own address (holy7c -> .4 via `shared`, holy8a -> .5 via `sapphire`), so submit one
# job per rack family; a duplicate on an already-served IP exits at once. Resumable: resubmit freely.
#   sbatch -p shared   tools/fetch_ztf_products.sh
#   sbatch -p sapphire tools/fetch_ztf_products.sh
#   PRODUCTS="refsexcat" sbatch ... ; FETCH_ARGS="--reverse" sbatch ... ; smoke test: FETCH_ARGS="--fields 643 --limit 6"
#SBATCH -J fetch_ztf
#SBATCH -c 1
#SBATCH --mem=2G
#SBATCH -t 3-00:00
#SBATCH -o /n/holystore01/LABS/berger_lab/Users/aboesky/long_transients2/ztf_data/fetch_logs/fetch_%j.out
#SBATCH -e /n/holystore01/LABS/berger_lab/Users/aboesky/long_transients2/ztf_data/fetch_logs/fetch_%j.err

module load gcc/12.2.0-fasrc01
module load python/3.12.5-fasrc01
source activate long_transients2

cd /n/home04/aboesky/berger/long_transients2
python3 -u tools/fetch_ztf_products.py --products ${PRODUCTS:-refpsfcat} --claim --one-per-ip ${FETCH_ARGS}
