#!/bin/bash
#SBATCH --partition=gpu-l40s
#SBATCH --account=cse
#SBATCH --mem-per-gpu=64G
#SBATCH --cpus-per-gpu=1
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=23:59:00
#SBATCH --job-name=eval
#SBATCH --output=/gscratch/ark/anjo0/rim-spoc/run_scripts/eval.out

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

git checkout main

conda activate spoc

python -m training.offline.online_eval --shuffle --eval_subset minival --output_basedir data/logs  \
    --test_augmentation --task_type PickupType  --input_sensors raw_navigation_camera raw_manipulation_camera last_actions an_object_is_in_hand  \
    --house_set objaverse --wandb_logging True --num_workers 1  --gpu_devices 0 --training_run_id s5d36lx7 \
    --local_checkpoint_dir data/results  --dataset_path 'data/datasets/fifteen/PickupType' \
    --wandb_project_name dl_project --wandb_entity_name 493_spoc_rim --ckptStep 6000
