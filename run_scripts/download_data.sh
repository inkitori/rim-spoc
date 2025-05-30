#!/bin/bash
#SBATCH --partition=ckpt
#SBATCH --account=cse
#SBATCH --job-name=download_data
#SBATCH --output=/gscratch/ark/anjo0/rim-spoc/run_scripts/download_data.out

CONDA_BASE=$(conda info --base) # This is a good way to get it if conda is in PATH

echo "CONDA_BASE detected as: ${CONDA_BASE}" # For debugging

# Source the conda.sh script
# The exact path might vary slightly based on your Conda version / installation type
# but etc/profile.d/conda.sh is standard
if [ -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]; then
    source "${CONDA_BASE}/etc/profile.d/conda.sh"
    echo "Sourced ${CONDA_BASE}/etc/profile.d/conda.sh"
else
    echo "ERROR: conda.sh not found at ${CONDA_BASE}/etc/profile.d/conda.sh"
    exit 1
fi

cd /gscratch/ark/anjo0/rim-spoc

conda activate spoc

python -m scripts.download_training_data --save_dir data/datasets --types fifteen --task_types RoomVisit
