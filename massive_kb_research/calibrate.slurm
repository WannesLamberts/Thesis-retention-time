#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=50
#SBATCH -A ap_adrem_wb_thesis2425
#SBATCH --time=3:00:00
#SBATCH --partition=zen2


module purge
module load calcua/2024a
module load SciPy-bundle/2024.05-gfbf-2024a
module load scikit-learn/1.5.2-gfbf-2024a
module load matplotlib/3.9.2-gfbf-2024a

chmod +x main.py
./main.py "${VSC_SCRATCH}/results/psms" "${VSC_DATA}/results_calibrated" "${VSC_DATA}/Chronologer.tsv"
