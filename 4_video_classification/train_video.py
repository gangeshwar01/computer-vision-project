#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Training Script for the Video Classifier

This script handles the entire pipeline for training the CNN+LSTM video classifier:
1. Loads and preprocesses video data using a custom Dataset.
2. Sets up the model, optimizer, and loss function.
3. Implements a training loop to train the model.
4. Implements an evaluation loop to validate the model's performance.
5. Saves the best model checkpoint based on validation accuracy.

Example Usage (assuming a dataset structure like UCF-101):
    python train_video.py \
        --data-path /path/to/ucf101/videos \
        --epochs 20 \
        --batch-size 8 \
        --num-frames 16
"""

import os
import argparse
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from tqdm import tqdm
import cv2  # OpenCV

from video_classifier import VideoClassifier

# --- 1. Custom Dataset for Videos ---

class VideoDataset(Dataset):
    """
    A custom PyTorch Dataset to load videos, sample frames, and apply transforms.
    Assumes a directory structure where each subdirectory is a class:
    - data_path/
        - class_1/
            - video_1.avi
            - video_2.avi
        - class_2/
            - video_3.avi
    """
    def __init__(self, data_path: str, num_frames: int, transform=None):
        self.data_path = data_path
        self.num_frames = num_frames
        self.transform = transform
        self.video_paths = []
        self.labels = []

        # Find all video files and their corresponding class labels
        self.class_names = sorted(os.listdir(data_path))
        self.class_to_idx = {name: i for i, name in enumerate(self.class_names)}

        for class_name in self.class_names:
            class_dir = os.path.join(data_path, class_name)
            for video_file in os.listdir(class_dir):
                self.video_paths.append(os.path.join(class_dir, video_file))
                self.labels.append(self.class_to_idx[class_name])

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        label = self.labels[idx]
        
        # Use OpenCV to read the video
        cap = cv2.VideoCapture(video_path)
        frames = []
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # Simple sampling: take frames at equal intervals
        indices = torch.linspace(0, total_frames - 1, self.num_frames).long()
        
        frame_idx = 0
        captured_count = 0
        while cap.isOpened() and captured_count < self.num_frames:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx in indices:
                # Convert from BGR (OpenCV default) to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
                captured_count += 1
            frame_idx += 1
        cap.release()
        
        # Apply transformations if provided
        if self.transform:
            # Apply same transform to all frames
            frames = torch.stack([self.transform(frame) for frame in frames])
        
        return frames, label


# --- 2. Training and Evaluation Functions ---

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """Trains the model for one epoch."""
    model.train()
    running_loss = 0.0
    
    for inputs, labels in tqdm(dataloader, desc="Training"):
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        
    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss

def evaluate(model, dataloader, criterion, device):
    """Evaluates the model on the validation set."""
    model.eval()
    running_loss = 0.0
    corrects = 0
    
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Evaluating"):
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            corrects += torch.sum(preds == labels.data)
            
    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = corrects.double() / len(dataloader.dataset)
    return epoch_loss, epoch_acc.item()


# --- 3. Main Execution Block ---

def main(args):
    """Main function to run the training pipeline."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Using device: {device}")

    # Define transformations for the frames
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Create Datasets and DataLoaders
    # Note: A real-world scenario requires separate train/val/test directories.
    # Here, we'll split the main dataset for demonstration.
    full_dataset = VideoDataset(args.data_path, args.num_frames, transform=transform)
    
    # 80-20 train-validation split
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    num_classes = len(full_dataset.class_names)
    print(f"Found {num_classes} classes.")

    # Initialize the model, criterion, and optimizer
    model = VideoClassifier(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_acc = 0.0
    os.makedirs("checkpoints", exist_ok=True)

    print("--- Starting Training ---")
    for epoch in range(args.epochs):
        print(f"\n--- Epoch {epoch+1}/{args.epochs} ---")
        
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        
        # Save the best model
        if val_acc > best_acc:
            best_acc = val_acc
            checkpoint_path = os.path.join("checkpoints", "best_video_classifier.pt")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"✨ New best model saved to {checkpoint_path} with accuracy: {best_acc:.4f}")

    print("--- Training Finished ---")
    print(f"Best validation accuracy: {best_acc:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Video Classifier Training Script")
    parser.add_argument('--data-path', type=str, required=True, help='Path to the video dataset directory.')
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs.')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size for training.')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate.')
    parser.add_argument('--num-frames', type=int, default=16, help='Number of frames to sample per video.')
    
    args = parser.parse_args()
    main(args)
