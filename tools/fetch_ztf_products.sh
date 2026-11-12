#!/bin/bash
# One IRSA download stream; resumable, so requeue or resubmit freely.
#   sbatch tools/fetch_ztf_products.sh                       # refpsfcat, whole footprint
#   PRODUCTS="refsexcat" sbatch tools/fetch_ztf_products.sh  # another product
#   SHARD=0/2 sbatch tools/fetch_ztf_products.sh             # one of two hosts
#   FETCH_ARGS=--reverse sbatch tools/fetch_ztf_products.sh    # from the far end, while a faster host works forward
#SBATCH -p shared
#SBATCH -c 1
#SBATCH --mem=2G
#SBATCH -t 3-00:00
#SBATCH -o /n/holystore01/LABS/berger_lab/Users/aboesky/long_transients2/ztf_data/fetch_logs/fetch_%j.out
#SBATCH -e /n/holystore01/LABS/berger_lab/Users/aboesky/long_transients2/ztf_data/fetch_logs/fetch_%j.err

module load gcc/12.2.0-fasrc01
module load python/3.12.5-fasrc01
source activate long_transients2

cd /n/home04/aboesky/berger/long_transients2
python3 -u tools/fetch_ztf_products.py --products ${PRODUCTS:-refpsfcat} --shard ${SHARD:-0/1} ${FETCH_ARGS}
