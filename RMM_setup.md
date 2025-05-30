# RIM SPOC Installation Guide

This guide describes the steps to set up and run RIM SPOC. Follow the instructions below.

## 1. Create and Activate the Conda Environment

- First install Conda from [this link](https://www.anaconda.com/download/success)
- Create the environment with Python 3.10.16:
  ```
  conda create -n spoc python=3.10.16
  ```
- Activate the environment:
  ```
  conda activate spoc
  ```

## 2. Install System Dependencies

- Install SWIG (only if not already installed):
  ```
  conda install -c conda-forge swig
  ```

## 3. Install Python Dependencies

- Install Python requirements (if you don't have CUDA, remove the top `xformers` line):
  ```
  pip install -r requirements.txt
  ```
- Install ai2thor:
  ```
  pip install --extra-index-url https://ai2thor-pypi.allenai.org/ ai2thor==0+5d0ab8ab8760eb584c5ae659c2b2b951cab23246
  ```
- Install xformers (only if you have CUDA):
  ```
  pip install xformers==0.0.28.post1
  ```

## 4. Download Required Data and Models

- Download NLTK datasets:
  ```
  python -m nltk.downloader wordnet2022
  python -m nltk.downloader wordnet
  ```
- Download Objathor annotations and assets:
  ```
  python -m objathor.dataset.download_annotations --version 2023_07_28 --path data/objaverse_assets
  python -m objathor.dataset.download_assets --version 2023_07_28 --path data/objaverse_assets
  ```

- Download Objathor houses:
	```
	python -m scripts.download_objaverse_houses --save_dir data/objaverse_houses --subset val
	```
- Set the environment variable for Objaverse:
  ```
  conda env config vars set OBJAVERSE_HOUSES_DIR=data/objaverse_houses
  conda env config vars set OBJAVERSE_DATA_DIR=data/objaverse_assets
  conda env config vars set WANDB_DIR=data/wandb
  ```
- Reset Conda environment to save environment variables
	```
	conda deactivate
	conda activate spoc
	```

## 5. Download Training Data

- Download training data for the desired tasks:
  ```
  python -m scripts.download_training_data --save_dir data/datasets --types fifteen --task_types {TASK_TYPE_1 TASK_TYPE_2 TASK_TYPE_3 ...}
  ```

- For example, to download PickupType and RoomNav, you should run
  ```
  python -m scripts.download_training_data --save_dir data/datasets --types fifteen --task_types PickupType RoomNav
  ```

  You can also replace fifteen with all to instead download all objects

## 6. Run Training

- Execute training with the following command (customize parameters as needed):
  ```
  python -m training.offline.train_pl --dataset_version PickupType --wandb_project_name dl_project --wandb_entity_name 493_spoc_rim --data_dir data/datasets/fifteen --input_sensors raw_navigation_camera raw_manipulation_camera last_actions an_object_is_in_hand --precision 16-mixed --lr 0.0002 --model_version siglip_3 --per_gpu_batch 16 --output_dir data/results --model EarlyFusionCnnTransformer --num_nodes 1 --resume_local
  ```


## 7. Run Evaluation

- Execute evaluation with the following command (customize parameters as needed):
  ```
  python -m training.offline.online_eval --shuffle --eval_subset minival --output_basedir data/logs  \
  --test_augmentation --task_type RoomNav  --input_sensors raw_navigation_camera raw_manipulation_camera last_actions an_object_is_in_hand  \
  --house_set objaverse --wandb_logging True --num_workers 1  --gpu_devices 0 --training_run_id 81vcsg1l \
  --local_checkpoint_dir data/results  --dataset_path 'data/datasets/fifteen/RoomNav' \
  --wandb_project_name dl_project --wandb_entity_name 493_spoc_rim --ckptStep 6000
  ```

