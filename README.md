
# Vietnamese Sign Language Recognition - VSL400

This repository implements models and tools for translating Vietnamese Sign Language (VSL) using video and pose/keypoint data.

## Project layout

Top-level layout (trimmed to most relevant folders):

```text
LICENSE
Makefile
README.md
requirements.txt
src/                       # main project code and entrypoints
  ├── train.py             # training entrypoint
  ├── inference.py         # inference entrypoint
  ├── configs/             # YAML config templates for training / 
  ├── data/                # datasets and processing code
  ├── models/              # trained model binaries and checkpoints
  ├── utils/               # utility modules used across the project
  ├── visualization/       # plotting and visualization helpers
  ├── convert_model_to_onnx.py
  ├── convert_model_to_torchscript.py
  ├── evaluate_model.py
  ├── extract_keypoints.py
  ├── register_pipeline.py
```

Note: See `src/` for the actual script names and `src/configs/` for example YAMLs.

## Datasets


Typical layout for VSL-400 in this repo's data directory:

```text
vsl_400/
  cam_1/    # videos from camera 1
  cam_2/
  cam_3/
  cam_1.json
  cam_2.json
  cam_3.json
  gloss.csv
```



## Installation

1. Create a Python 3.9 environment (this project was developed against Python 3.9.x).

2. (Optional) Install PyTorchVideo if your chosen configs depend on it:

```powershell
cd src/libs
git clone https://github.com/facebookresearch/pytorchvideo.git
pip install -e pytorchvideo
cd ../../
```

3. Install Python requirements:

```powershell
pip install -r requirements.txt
```

If you use `wandb` or private Hugging Face models/datasets, log into those services before running training/inference.

## Data Processing
Step 1: Detect gesture boundaries (TBL)

```powershell
python src/data/temporal_boundary_localization.py --input_video video.mp4 --get_cut_time
```
Step 2: Segment and crop videos (BGSP)

```powershell 
python src/data/boundary_segmentation_pruning.py --input_video video.mp4 --cut_crop_video
```
Output: Individual segmented videos saved in ./video/ directory
## Configuration

Configs live in `src/configs/` separated by training/inference subfolders. Typical fields to update:

- data: dataset, modality, subset, data_dir 
- training: run_name, hub_model_id

## Training

From the project root you can start training with a config file:

```powershell
python src/train.py --config_path src/configs/training/config.yaml
```

This will read the YAML under `src/configs/` and run the training pipeline. Common issues to check:

- Ensure `data.data_dir` points to your local copy of the dataset.
- Make sure required pretrained weights are accessible (local path or HF hub).
- If using `wandb` set `report_to` in the config and log in with `wandb login`.

## Inference

Run inference (evaluation or producing predictions) with:

```powershell
python src/inference.py --config_path src/configs/inference/config.yaml
```

There are also helpers for model conversion and evaluation:

- `src/evaluate_model.py` — run evaluation metrics on predictions.
- `src/extract_keypoints.py` — utilities to extract pose/keypoint features from videos.



