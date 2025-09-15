#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
YOLOv8 Training Script

This script provides a command-line interface to train a YOLOv8 model 
for object detection using the Ultralytics library. It allows customization of 
the model, dataset, and various training hyperparameters.

Example Usage:
    python train.py --model yolov8n.yaml --weights yolov8n.pt --data coco128.yaml --epochs 100 --batch 16 --imgsz 640
"""

import argparse
import torch
from ultralytics import YOLO

def train_yolo(args):
    """
    Initializes and trains a YOLOv8 model based on provided arguments.
    
    Args:
        args: An object containing the command-line arguments.
    """
    # Check for GPU availability and set the device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🚀 Training on device: {device}")

    # --- 1. Load the Model ---
    # Load a pre-trained model if weights are provided, otherwise load from a YAML configuration
    if args.weights:
        print(f"Loading pre-trained model from: {args.weights}")
        model = YOLO(args.weights)
    else:
        print(f"Building a new model from YAML: {args.model}")
        model = YOLO(args.model)

    # Move model to the selected device
    model.to(device)

    # --- 2. Start Training ---
    print("Starting YOLOv8 training...")
    try:
        results = model.train(
            data=args.data,
            epochs=args.epochs,
            batch=args.batch,
            imgsz=args.imgsz,
            patience=50,       # Stop training if no improvement is seen for 50 epochs
            project="runs/train",  # Directory to save training runs
            name="yolov8_custom_training", # Name of the specific run
            exist_ok=True,     # Overwrite existing project/name directory
            verbose=True,      # Print detailed logs
        )
        print("✅ Training completed successfully!")
        print(f"📊 Results saved to: {results.save_dir}")

    except Exception as e:
        print(f"❌ An error occurred during training: {e}")

if __name__ == "__main__":
    # --- 3. Argument Parsing ---
    parser = argparse.ArgumentParser(description="YOLOv8 Training Script")
    
    parser.add_argument(
        '--model', 
        type=str, 
        default='yolov8n.yaml', 
        help='Path to the model configuration YAML file (e.g., yolov8n.yaml).'
    )
    parser.add_argument(
        '--weights', 
        type=str, 
        default='yolov8n.pt', 
        help='Path to pre-trained model weights (e.g., yolov8n.pt).'
    )
    parser.add_argument(
        '--data', 
        type=str, 
        required=True, 
        help='Path to the dataset configuration YAML file (e.g., coco128.yaml).'
    )
    parser.add_argument(
        '--epochs', 
        type=int, 
        default=100, 
        help='Total number of training epochs.'
    )
    parser.add_argument(
        '--batch', 
        type=int, 
        default=16, 
        help='Batch size for training. Use -1 for auto-batch size.'
    )
    parser.add_argument(
        '--imgsz', 
        type=int, 
        default=640, 
        help='Input image size for training (e.g., 640 for 640x640).'
    )

    args = parser.parse_args()
    
    # Run the training function
    train_yolo(args)
