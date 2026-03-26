#!/usr/bin/env python
"""Prepare YOLO-pose dataset from CARLA simulation data.

Converts CARLA data (RGB or event frames + annotations) into YOLO training format.
Creates train/val splits with symlinks to save disk space.

Usage:
    python prepare_yolo_dataset.py --data-dir DATA/Town10HD --output-dir datasets/pose_rgb --modality rgb
    python prepare_yolo_dataset.py --data-dir DATA/Town10HD --output-dir datasets/pose_events --modality events
"""

import argparse
import os
import random
import shutil
from pathlib import Path
from typing import List, Tuple

import yaml


# CARLA skeleton keypoints (16 keypoints)
KEYPOINT_NAMES = [
    "head",           # crl_Head__C
    "right_eye",      # crl_eye__R
    "left_eye",       # crl_eye__L
    "right_shoulder", # crl_shoulder__R
    "left_shoulder",  # crl_shoulder__L
    "right_upper_arm",# crl_arm__R
    "left_upper_arm", # crl_arm__L
    "right_forearm",  # crl_foreArm__R
    "left_forearm",   # crl_foreArm__L
    "hip_center",     # crl_hips__C
    "right_thigh",    # crl_thigh__R
    "left_thigh",     # crl_thigh__L
    "right_shin",     # crl_leg__R
    "left_shin",      # crl_leg__L
    "right_foot",     # crl_foot__R
    "left_foot",      # crl_foot__L
]

# Skeleton connections for visualization
SKELETON = [
    [0, 1], [0, 2],           # head to eyes
    [0, 3], [0, 4],           # head to shoulders
    [3, 5], [5, 7],           # right arm
    [4, 6], [6, 8],           # left arm
    [3, 9], [4, 9],           # shoulders to hip
    [9, 10], [9, 11],         # hip to thighs
    [10, 12], [12, 14],       # right leg
    [11, 13], [13, 15],       # left leg
]


def find_matching_frames(
    data_dir: Path,
    modality: str
) -> List[Tuple[str, Path, Path]]:
    """Find frames that have both image and annotation files.
    
    Args:
        data_dir: Path to data directory (e.g., DATA/Town10HD)
        modality: 'rgb' or 'events'
    
    Returns:
        List of (frame_id, image_path, label_path) tuples
    """
    annot_dir = data_dir / "Annot"
    
    if modality == "rgb":
        image_dir = data_dir / "RGB"
        image_suffix = "_RGB.png"
    else:  # events
        image_dir = data_dir / "events"
        image_suffix = ".png"
    
    # Get all annotation frame IDs
    annot_files = list(annot_dir.glob("*.txt"))
    
    matched = []
    for annot_path in annot_files:
        frame_id = annot_path.stem
        
        # Construct image path
        if modality == "rgb":
            image_path = image_dir / f"{frame_id}_RGB.png"
        else:
            image_path = image_dir / f"{frame_id}.png"
        
        # Check if image exists
        if image_path.exists():
            # Check if annotation is non-empty
            if annot_path.stat().st_size > 0:
                matched.append((frame_id, image_path, annot_path))
    
    return sorted(matched, key=lambda x: x[0])


def split_dataset(
    frames: List[Tuple[str, Path, Path]],
    train_ratio: float = 0.8,
    seed: int = 42
) -> Tuple[List, List]:
    """Split frames into train and validation sets.
    
    Args:
        frames: List of (frame_id, image_path, label_path) tuples
        train_ratio: Fraction of data for training
        seed: Random seed for reproducibility
    
    Returns:
        (train_frames, val_frames) tuple
    """
    random.seed(seed)
    shuffled = frames.copy()
    random.shuffle(shuffled)
    
    split_idx = int(len(shuffled) * train_ratio)
    return shuffled[:split_idx], shuffled[split_idx:]


def create_dataset_structure(
    output_dir: Path,
    train_frames: List[Tuple[str, Path, Path]],
    val_frames: List[Tuple[str, Path, Path]],
    use_symlinks: bool = True
) -> None:
    """Create YOLO dataset directory structure with symlinks.
    
    Args:
        output_dir: Output directory for dataset
        train_frames: Training frame tuples
        val_frames: Validation frame tuples
        use_symlinks: Use symlinks instead of copying files
    """
    # Create directories
    for split in ["train", "val"]:
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)
    
    def link_or_copy(src: Path, dst: Path):
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        if use_symlinks:
            dst.symlink_to(src.resolve())
        else:
            shutil.copy2(src, dst)
    
    # Process train frames
    for frame_id, img_path, label_path in train_frames:
        img_dst = output_dir / "images" / "train" / f"{frame_id}.png"
        label_dst = output_dir / "labels" / "train" / f"{frame_id}.txt"
        link_or_copy(img_path, img_dst)
        link_or_copy(label_path, label_dst)
    
    # Process val frames
    for frame_id, img_path, label_path in val_frames:
        img_dst = output_dir / "images" / "val" / f"{frame_id}.png"
        label_dst = output_dir / "labels" / "val" / f"{frame_id}.txt"
        link_or_copy(img_path, img_dst)
        link_or_copy(label_path, label_dst)


def generate_data_yaml(
    output_dir: Path,
    num_keypoints: int = 16
) -> None:
    """Generate YOLO data.yaml configuration file.
    
    Args:
        output_dir: Output directory for dataset
        num_keypoints: Number of keypoints per person
    """
    data_config = {
        "path": str(output_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": {
            0: "person"
        },
        "nc": 1,  # number of classes
        "kpt_shape": [num_keypoints, 3],  # [num_keypoints, dims] where dims=3 for x,y,visibility
        "flip_idx": [
            0,        # head (center, no flip)
            2, 1,     # eyes (swap left/right)
            4, 3,     # shoulders (swap)
            6, 5,     # upper arms (swap)
            8, 7,     # forearms (swap)
            9,        # hip center (no flip)
            11, 10,   # thighs (swap)
            13, 12,   # shins (swap)
            15, 14,   # feet (swap)
        ],
        "skeleton": SKELETON,
        "keypoint_names": KEYPOINT_NAMES,
    }
    
    yaml_path = output_dir / "data.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(data_config, f, default_flow_style=False, sort_keys=False)
    
    print(f"Created: {yaml_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare YOLO-pose dataset from CARLA data"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Path to data directory (e.g., DATA/Town10HD)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for YOLO dataset"
    )
    parser.add_argument(
        "--modality",
        type=str,
        choices=["rgb", "events"],
        required=True,
        help="Input modality: 'rgb' or 'events'"
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Fraction of data for training (default: 0.8)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible splits (default: 42)"
    )
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Copy files instead of creating symlinks"
    )
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    
    # Validate input directory
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    
    print(f"Input: {data_dir}")
    print(f"Output: {output_dir}")
    print(f"Modality: {args.modality}")
    print(f"Train ratio: {args.train_ratio}")
    print(f"Seed: {args.seed}")
    print(f"Use symlinks: {not args.copy}")
    print()
    
    # Find matching frames
    print("Finding matching frames...")
    frames = find_matching_frames(data_dir, args.modality)
    print(f"Found {len(frames)} frames with both images and annotations")
    
    if len(frames) == 0:
        print("ERROR: No matching frames found!")
        print(f"  Check that {data_dir / 'Annot'} has .txt files")
        if args.modality == "rgb":
            print(f"  Check that {data_dir / 'RGB'} has *_RGB.png files")
        else:
            print(f"  Check that {data_dir / 'events'} has *.png files")
        return
    
    # Split dataset
    print("Splitting dataset...")
    train_frames, val_frames = split_dataset(
        frames, args.train_ratio, args.seed
    )
    print(f"  Train: {len(train_frames)} frames")
    print(f"  Val: {len(val_frames)} frames")
    
    # Create directory structure
    print("Creating dataset structure...")
    create_dataset_structure(
        output_dir, train_frames, val_frames,
        use_symlinks=not args.copy
    )
    
    # Generate data.yaml
    print("Generating data.yaml...")
    generate_data_yaml(output_dir)
    
    print()
    print("=" * 50)
    print("Dataset preparation complete!")
    print(f"  Location: {output_dir.resolve()}")
    print()
    print("To train YOLOv8-pose:")
    print(f"  yolo pose train data={output_dir.resolve()}/data.yaml model=yolov8n-pose.pt epochs=100")
    print()
    print("To train YOLOv11-pose:")
    print(f"  yolo pose train data={output_dir.resolve()}/data.yaml model=yolo11n-pose.pt epochs=100")


if __name__ == "__main__":
    main()
