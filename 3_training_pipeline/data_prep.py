#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
YOLOv8 Dataset Preparation Script

This script automates the process of preparing a custom dataset for YOLOv8 training.
It converts Pascal VOC XML annotations to the YOLO .txt format and then splits the
dataset into training, validation, and test sets.

Assumptions:
- Your initial dataset has images (e.g., .jpg) and annotations (.xml) in the same directory.
- The XML files have the same name as their corresponding image files.

Example Usage:
    python data_prep.py \
        --input-dir /path/to/raw_data \
        --output-dir /path/to/yolo_dataset \
        --split-ratio 0.8 0.1 0.1 \
        --class-names "cat,dog,person"
"""

import os
import shutil
import random
import argparse
import xml.etree.ElementTree as ET
from pathlib import Path
from tqdm import tqdm
import yaml

def convert_voc_to_yolo(xml_file: Path, class_mapping: dict, img_size: tuple):
    """
    Converts a single Pascal VOC XML annotation to the YOLO format.

    Args:
        xml_file (Path): Path to the XML annotation file.
        class_mapping (dict): A dictionary mapping class names to integer IDs.
        img_size (tuple): A tuple containing the (width, height) of the image.

    Returns:
        list: A list of strings, where each string is a YOLO formatted annotation line.
    """
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    img_width, img_height = img_size
    
    yolo_annotations = []
    
    for obj in root.findall('object'):
        class_name = obj.find('name').text
        if class_name not in class_mapping:
            continue
            
        class_id = class_mapping[class_name]
        
        bndbox = obj.find('bndbox')
        xmin = float(bndbox.find('xmin').text)
        ymin = float(bndbox.find('ymin').text)
        xmax = float(bndbox.find('xmax').text)
        ymax = float(bndbox.find('ymax').text)
        
        # YOLO format calculations
        dw = 1.0 / img_width
        dh = 1.0 / img_height
        x_center = (xmin + xmax) / 2.0 * dw
        y_center = (ymin + ymax) / 2.0 * dh
        width = (xmax - xmin) * dw
        height = (ymax - ymin) * dh
        
        yolo_annotations.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
        
    return yolo_annotations

def process_dataset(input_dir: Path, output_dir: Path, class_names: list):
    """
    Processes the entire dataset, converting all annotations.
    """
    class_mapping = {name: i for i, name in enumerate(class_names)}
    
    # Create a temporary directory for converted labels
    temp_labels_dir = output_dir / "temp_labels"
    temp_labels_dir.mkdir(parents=True, exist_ok=True)

    print("🔍 Finding image and annotation files...")
    image_files = list(input_dir.glob('*.jpg')) + list(input_dir.glob('*.png'))

    print("🔄 Converting annotations from VOC XML to YOLO TXT format...")
    for img_path in tqdm(image_files, desc="Converting Annotations"):
        xml_path = img_path.with_suffix('.xml')
        
        if not xml_path.exists():
            print(f"Warning: Annotation file not found for {img_path.name}, skipping.")
            continue

        # This part requires an image library like PIL or OpenCV to get image size.
        # For simplicity, we assume a common size or parse it from XML if available.
        # A robust implementation would use: from PIL import Image; w, h = Image.open(img_path).size
        try:
            tree = ET.parse(xml_path)
            size_node = tree.getroot().find('size')
            img_width = int(size_node.find('width').text)
            img_height = int(size_node.find('height').text)
        except Exception:
            print(f"Warning: Could not determine image size for {img_path.name}. Skipping.")
            continue

        yolo_lines = convert_voc_to_yolo(xml_path, class_mapping, (img_width, img_height))
        
        if yolo_lines:
            label_path = temp_labels_dir / f"{img_path.stem}.txt"
            with open(label_path, 'w') as f:
                f.write('\n'.join(yolo_lines))

def split_and_organize_dataset(input_dir: Path, output_dir: Path, split_ratio: tuple):
    """
    Splits the dataset and organizes it into the final YOLO directory structure.
    """
    train_ratio, val_ratio, test_ratio = split_ratio
    temp_labels_dir = output_dir / "temp_labels"

    image_files = list(input_dir.glob('*.jpg')) + list(input_dir.glob('*.png'))
    # Filter out images for which we couldn't create a label
    valid_stems = {p.stem for p in temp_labels_dir.glob('*.txt')}
    images_to_process = [p for p in image_files if p.stem in valid_stems]
    
    random.shuffle(images_to_process)
    
    total_files = len(images_to_process)
    train_end = int(total_files * train_ratio)
    val_end = train_end + int(total_files * val_ratio)
    
    splits = {
        'train': images_to_process[:train_end],
        'val': images_to_process[train_end:val_end],
        'test': images_to_process[val_end:]
    }
    
    print("🗂️  Creating final YOLO directory structure and splitting files...")
    for split_name, file_list in splits.items():
        img_dir = output_dir / split_name / 'images'
        lbl_dir = output_dir / split_name / 'labels'
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        
        for img_path in tqdm(file_list, desc=f"Processing {split_name} set"):
            lbl_path = temp_labels_dir / f"{img_path.stem}.txt"
            
            # Copy files to their final destination
            shutil.copy(img_path, img_dir / img_path.name)
            shutil.copy(lbl_path, lbl_dir / lbl_path.name)
            
    # Clean up temporary directory
    shutil.rmtree(temp_labels_dir)

def create_yaml_file(output_dir: Path, class_names: list):
    """
    Creates the data.yaml file required for YOLO training.
    """
    data = {
        'train': str(output_dir / 'train' / 'images'),
        'val': str(output_dir / 'val' / 'images'),
        'test': str(output_dir / 'test' / 'images'),
        'nc': len(class_names),
        'names': class_names
    }
    
    yaml_path = output_dir / 'data.yaml'
    with open(yaml_path, 'w') as f:
        yaml.dump(data, f, sort_keys=False, default_flow_style=False)
    print(f"✅ Created dataset YAML file at: {yaml_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLOv8 Dataset Preparation Script")
    parser.add_argument('--input-dir', type=str, required=True, help='Directory with raw images and XML annotations.')
    parser.add_argument('--output-dir', type=str, required=True, help='Directory to save the YOLO formatted dataset.')
    parser.add_argument('--split-ratio', nargs=3, type=float, default=[0.8, 0.1, 0.1], help='Train, validation, test split ratio (e.g., 0.8 0.1 0.1).')
    parser.add_argument('--class-names', type=str, required=True, help='Comma-separated list of class names in order (e.g., "cat,dog,person").')

    args = parser.parse_args()

    # Validate split ratio
    if sum(args.split_ratio) != 1.0:
        raise ValueError("Split ratios must sum to 1.0")

    input_path = Path(args.input_dir)
    output_path = Path(args.output_dir)
    class_list = [name.strip() for name in args.class_names.split(',')]

    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)

    # Run the preparation steps
    process_dataset(input_path, output_path, class_list)
    split_and_organize_dataset(input_path, output_path, tuple(args.split_ratio))
    create_yaml_file(output_path, class_list)

    print("🎉 Dataset preparation complete!")
