# SEPOSE: SEPose: A Synthetic Event-based Human Pose Estimation Dataset for Pedestrian Monitoring

<figure>
  <img src="images/weathers.png" width="100%" />
  <figcaption>Summary of the maps within CARLA used to generate SEPose data</figcaption>
</figure>

A synthetic data generation pipeline for multi-human pose estimation using CARLA simulator from the POV of fixed traffic cameras spanning busy and light crowds and traffic caross diverse traffic and weather conditions in urban, suburban, and rural settings.

Supports both RGB and Dynamic Vision Sensor (DVS) event camera modalities.

## Overview

This repository provides tools to:
- Generate synthetic pose estimation data from CARLA simulator
- Produce ground-truth annotations in YOLO-pose format (16 keypoints)
- Create training datasets for both RGB and event camera inputs
- Train and evaluate YOLO pose models

### Keypoint Definition (16 joints)

| Index | Keypoint | Index | Keypoint |
|-------|----------|-------|----------|
| 0 | Head | 8 | Left Forearm |
| 1 | Right Eye | 9 | Hip Center |
| 2 | Left Eye | 10 | Right Thigh |
| 3 | Right Shoulder | 11 | Left Thigh |
| 4 | Left Shoulder | 12 | Right Shin |
| 5 | Right Upper Arm | 13 | Left Shin |
| 6 | Left Upper Arm | 14 | Right Foot |
| 7 | Right Forearm | 15 | Left Foot |

---

## Installation

### Prerequisites
- Python 3.7+
- CARLA Simulator 0.9.15
- NVIDIA GPU (recommended)

### Setup

```bash
# Clone repository
git clone https://github.com/Kaustav97/SEPose.git 
cd eventpose

# Install dependencies
pip install -r requirements.txt
```
Install the CARLA Simulator from https://carla.org/ to setup the CARLA server from precompiled binaries or the docker image.


## Data Generation

### 1. Start CARLA Server

```bash
# In a separate terminal
cd /path/to/carla
./CarlaUE4.sh -quality-level=Epic -RenderOffScreen
```

### 2. Run Data Collection

```bash
# Basic usage (uses configs/default.yaml)
python GenerateData.py --out-dir ./DATA

# Custom configuration
python GenerateData.py --config configs/my_experiment.yaml --out-dir ./DATA

# Override specific parameters via CLI
python GenerateData.py --out-dir ./DATA -n 50 -w 30  # 50 vehicles, 30 pedestrians
```

### Output Structure

```
DATA/Town10HD/
├── events/     # DVS event frames (PNG)
├── RGB/        # RGB camera frames (PNG)
├── GT/         # Visualization with skeleton overlay
└── Annot/      # YOLO-pose annotations (TXT)
```

---

## Configuration

All simulation parameters are controlled via YAML configuration files in `configs/`.

### Key Configuration Sections

```yaml
# configs/default.yaml

simulation:
  num_frames: 15000             # Total frames to capture
  weather_reset_interval: 300   # Seconds between scene resets
  data_collection_cooldown: 5.0 # Skip frames after reset to avoid artifacts

actors:
  vehicles:
    count: 30
    filter: "vehicle.*"
  walkers:
    count: 10
    percentage_running: 0.0
    percentage_crossing: 0.0

camera:
  dvs:
    positive_threshold: "0.7"
    negative_threshold: "0.7"
    sigma_positive_threshold: "0.7"
    sigma_negative_threshold: "0.7"
    refractory_period_ns: "330000"

weather_presets:
  sunny: { cloudiness: 10.0, precipitation: 0.0, ... }
  rainy: { cloudiness: 90.0, precipitation: 70.0, ... }
  # ... other weather presets
```

### CLI Arguments

| Argument | Description |
|----------|-------------|
| `--config` | Path to YAML config (default: `configs/default.yaml`) |
| `--out-dir` | Output directory for generated data |
| `-n, --number-of-vehicles` | Number of vehicles to spawn |
| `-w, --number-of-walkers` | Number of pedestrians to spawn |
| `--host` | CARLA server hostname |
| `-p, --port` | CARLA server port |

CLI arguments override config file values.

---

## Dataset Preparation

Convert raw data to YOLO training format (uses symlinks by default to save space):

```bash
# For event camera data
python prepare_yolo_dataset.py \
    --data-dir DATA/Town10HD \
    --output-dir datasets/eventpose_events \
    --modality events

# For RGB data
python prepare_yolo_dataset.py \
    --data-dir DATA/Town10HD \
    --output-dir datasets/eventpose_rgb \
    --modality rgb
```

### Options

| Argument | Default | Description |
|----------|---------|-------------|
| `--modality` | required | `rgb` or `events` |
| `--train-ratio` | 0.8 | Train/val split ratio |
| `--seed` | 42 | Random seed for reproducibility |
| `--copy` | false | Copy files instead of symlinks |

### Output Structure

```
datasets/eventpose_events/
├── data.yaml           # YOLO configuration
├── images/
│   ├── train/          # Training images (symlinks)
│   └── val/            # Validation images
└── labels/
    ├── train/          # Training annotations
    └── val/            # Validation annotations
```

---

## Training

### YOLOv8-Pose

```bash
# Install ultralytics
pip install ultralytics

# Train on event data
yolo pose train \
    data=datasets/eventpose_events/data.yaml \
    model=yolov8n-pose.pt \
    epochs=100 \
    imgsz=640 \
    batch=16

# Train on RGB data
yolo pose train \
    data=datasets/eventpose_rgb/data.yaml \
    model=yolov8n-pose.pt \
    epochs=100
```

### YOLOv11-Pose

```bash
# Train with YOLOv11
yolo pose train \
    data=datasets/eventpose_events/data.yaml \
    model=yolo11n-pose.pt \
    epochs=100 \
    imgsz=640
```

<!-- ### Training Tips

- **Batch size**: Reduce if OOM (e.g., `batch=8`)
- **Image size**: Default 640, increase for better accuracy
- **Pretrained weights**: Start from COCO-pretrained for faster convergence
- **Augmentation**: YOLO applies mosaic, mixup, HSV augmentation by default -->

---

## Evaluation

```bash
# Validate model
yolo pose val \
    data=datasets/eventpose_events/data.yaml \
    model=runs/pose/train/weights/best.pt

# Inference on images
yolo pose predict \
    model=runs/pose/train/weights/best.pt \
    source=path/to/images
```

---

## Annotation Format

Annotations follow YOLO-pose format:

```
<class> <x_c> <y_c> <w> <h> <kp1_x> <kp1_y> <kp1_v> ... <kp16_x> <kp16_y> <kp16_v>
```

- All coordinates normalized to [0, 1]
- Visibility flags: `0` = not visible, `2` = visible
- One line per person

---

## Project Structure

```
eventpose/
├── GenerateData.py          # Main data generation script
├── config.py                # Configuration loader
├── prepare_yolo_dataset.py  # Dataset preparation for YOLO
├── draw_skeleton.py         # Skeleton visualization utilities
├── validate_yolo_pose.py    # Annotation validation
├── configs/
│   └── default.yaml         # Default simulation config
├── DATA/                    # Generated data (gitignored)
└── datasets/                # Prepared YOLO datasets
```

---

## Citation

```bibtex
@article{chanda2025seposesyntheticeventbasedhuman,
      title={SEPose: A Synthetic Event-based Human Pose Estimation Dataset for Pedestrian Monitoring}, 
      author={Kaustav Chanda and Aayush Atul Verma and Arpitsinh Vaghela and Yezhou Yang and Bharatesh Chakravarthi},
      year={2025},
      eprint={2507.11910},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2507.11910},
}
```

<!-- ## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details. -->

## Acknowledgments

- [CARLA Simulator](https://carla.org/) for the simulation environment
- [Ultralytics](https://github.com/ultralytics/ultralytics) for YOLO implementation
