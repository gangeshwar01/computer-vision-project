# Computer Vision Model Exploration 🤖

This repository contains a collection of studies, implementations, and training pipelines for various state-of-the-art computer vision tasks. The project explores the evolution of YOLO architectures, analyzes SOTA models, provides a robust training pipeline for object detection, and implements a baseline model for video classification.

-----

## 📂 Repository Structure

The project is organized into four main modules, each focusing on a specific area of study:

```
computer-vision-project/
├── README.md                 # You are here!
│
├── 1_yolo_architectures/
│   ├── arch_summary.md       # Detailed notes on YOLO's architectural evolution
│   └── diagrams/             # Architecture diagrams
│
├── 2_sota_analysis/
│   └── sota_report.md        # Analysis of SOTA models in CV tasks
│
├── 3_training_pipeline/
│   ├── train.py              # Main script for training YOLOv8
│   ├── data_prep.py          # Utility script to prepare custom datasets
│   └── requirements.txt      # Dependencies for the YOLO pipeline
│
└── 4_video_classification/
    ├── video_classifier.py   # CNN+LSTM model definition
    ├── train_video.py        # Script to train the video classifier
    └── requirements.txt      # Dependencies for video classification
```

-----

## 🎯 Key Areas of Study

### 1\. YOLO Architectures

This module traces the architectural evolution of the YOLO family, from YOLOv3 to the latest SOTA versions like YOLOv9. It breaks down the key components—Backbone, Neck, and Head—and highlights significant improvements such as the move to anchor-free designs and more efficient backbones.

➡️ **See the detailed analysis in [`1_yolo_architectures/arch_summary.md`](https://www.google.com/search?q=1_yolo_architectures/arch_summary.md).**

### 2\. State-of-the-Art (SOTA) Analysis

A comprehensive report analyzing the leading models in three core computer vision tasks:

  * **Image Classification:** Vision Transformers (ViT) vs. modern ConvNets.
  * **Object Detection:** A comparison of YOLO, RT-DETR, and DINO, focusing on the speed vs. accuracy trade-off.
  * **Object Segmentation:** An overview of foundational models like SAM and instance segmentation leaders like Mask2Former.

➡️ **Read the full report in [`2_sota_analysis/sota_report.md`](https://www.google.com/search?q=2_sota_analysis/sota_report.md).**

### 3\. YOLOv8 Training Pipeline

A practical and reusable pipeline for training YOLOv8 models on custom datasets.

  * **`data_prep.py`**: A utility script to convert annotations from Pascal VOC (.xml) to YOLO (.txt) format and automatically split the data into train, validation, and test sets.
  * **`train.py`**: A flexible command-line script to launch training runs, allowing customization of models, datasets, and hyperparameters.

### 4\. Video Classification

An implementation of a baseline video classification model.

  * **`video_classifier.py`**: Defines a classic CNN + LSTM architecture, where a pre-trained ResNet50 extracts frame-level features and an LSTM models their temporal sequence.
  * **`train_video.py`**: A complete script to train and evaluate the model on a video dataset, including a custom PyTorch `Dataset` for loading and processing videos.

-----

## 🚀 Setup and Usage

### Prerequisites

  * Python 3.8+
  * Git
  * (Optional but recommended) NVIDIA GPU with CUDA for training.

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/computer-vision-project.git
    cd computer-vision-project
    ```
2.  **Install dependencies for each module:**
      * **For the YOLO Training Pipeline:**
        ```bash
        pip install -r 3_training_pipeline/requirements.txt
        ```
      * **For Video Classification:**
        ```bash
        pip install -r 4_video_classification/requirements.txt
        ```

### Running the Scripts

#### 1\. Preparing a Custom Dataset for YOLO

```bash
python 3_training_pipeline/data_prep.py \
    --input-dir /path/to/raw_data \
    --output-dir /path/to/yolo_dataset \
    --class-names "class1,class2,class3"
```

#### 2\. Training the YOLOv8 Model

```bash
python 3_training_pipeline/train.py \
    --weights yolov8n.pt \
    --data /path/to/yolo_dataset/data.yaml \
    --epochs 100 \
    --batch 16
```

#### 3\. Training the Video Classification Model

```bash
python 4_video_classification/train_video.py \
    --data-path /path/to/video_dataset \
    --epochs 20 \
    --batch-size 8 \
    --num-frames 16
```

-----

## 📊 Expected Results

Running the training scripts will generate output directories (`runs/` for YOLO, `checkpoints/` for the video model) containing:

  * **Trained model weights** (`best.pt`).
  * **Performance metrics** and logs.
  * **Visualizations** like loss curves and confusion matrices.
