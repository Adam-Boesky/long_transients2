# Sourced by every jobs/*.sh on Cannon: toolchain, conda env, repo root, data paths. Submit from the repo root.
module load gcc/12.2.0-fasrc01 python/3.12.5-fasrc01
source activate long_transients2
cd /n/home04/aboesky/berger/long_transients2
export PYTHONPATH=.
export LT_DATA=/n/holystore01/LABS/berger_lab/Users/aboesky/long_transients2                      # Extracting.utils.get_data_path on the cluster
export PS1_STACK_DIR=/n/holystore01/LABS/berger_lab/Users/aboesky/long_transients/panstarrs/stack_object/dataset
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1                                 # parallelism comes from --n-jobs / arrays
mkdir -p "$LT_DATA/logs"
export NUMBA_CACHE_DIR="${TMPDIR:-/tmp}/numba_cache_${SLURM_JOB_ID:-$$}"                           # per job: concurrent tasks race on the shared __pycache__ (Extracting.kron)
