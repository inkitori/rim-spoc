#!/bin/bash
#SBATCH --partition=gpu-l40s
#SBATCH --account=cse
#SBATCH --mem-per-gpu=64G
#SBATCH --cpus-per-gpu=1
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --time=23:59:00
#SBATCH --job-name=train
#SBATCH --output=/gscratch/ark/anjo0/rim-spoc/run_scripts/train.out

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

python -m training.offline.train_pl --dataset_version RoomVisit --wandb_project_name dl_project --wandb_entity_name 493_spoc_rim --data_dir data/datasets/fifteen --input_sensors raw_navigation_camera raw_manipulation_camera last_actions an_object_is_in_hand --precision 16-mixed --lr 0.00002 --model_version siglip_3 --per_gpu_batch 16 --output_dir data/results --model EarlyFusionCnnTransformer --num_nodes 1 
