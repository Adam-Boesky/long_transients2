#!/bin/bash
# One field's source tables (Extracting.field_tables) per array task: sbatch --array=<fields> Extracting/build_field_tables.sh
# with FIELDS="293 294 ..." in the environment, or sbatch --export=FIELD=860 for one field.

#SBATCH -p shared
#SBATCH -c 8
#SBATCH --mem=20G
#SBATCH -t 0-02:00
#SBATCH -o /n/holystore01/LABS/berger_lab/Users/aboesky/long_transients2/logs/field_tables_%A_%a.out
#SBATCH -e /n/holystore01/LABS/berger_lab/Users/aboesky/long_transients2/logs/field_tables_%A_%a.err

module load gcc/12.2.0-fasrc01
module load python/3.12.5-fasrc01
source activate long_transients2

cd /n/home04/aboesky/berger/long_transients2
export PS1_STACK_DIR=/n/holystore01/LABS/berger_lab/Users/aboesky/long_transients/panstarrs/stack_object/dataset
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

if [ -n "${SLURM_ARRAY_TASK_ID}" ] && [ -n "${FIELDS}" ]; then
    FIELD=$(echo ${FIELDS} | tr ' ' '\n' | sed -n "$((SLURM_ARRAY_TASK_ID + 1))p")
fi
FIELD=${FIELD:?set FIELD or FIELDS}
# the builder reads the flat layout: link the bulk download's per-field products and the old data dir's images
L=/n/holystore01/LABS/berger_lab/Users/aboesky/long_transients2/ztf_data
F=$(printf %06d ${FIELD})
for p in $L/field${F}/ztf_${F}_*.fits /n/holystore01/LABS/berger_lab/Users/aboesky/long_transients/ztf_data/ztf_${F}_*_refimg.fits; do
    [ -e "$p" ] && ln -sfn "$p" "$L/$(basename $p)"
done
python3 -u -m Extracting.field_tables ${FIELD} --n-jobs ${SLURM_CPUS_PER_TASK:-1} ${REQUIRE:+--require ${REQUIRE}}
