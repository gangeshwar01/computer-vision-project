#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Video Classification Model using CNN + LSTM

This script defines the neural network architecture for video classification.
It consists of two main parts:
1. A pre-trained CNN (ResNet50) to extract spatial features from each frame.
2. An LSTM network to learn the temporal patterns from the sequence of frame features.

The model is designed to take a sequence of video frames as input and output a
prediction for the entire video clip's class.
"""

import torch
import torch.nn as nn
import torchvision.models as models

class CnnFeatureExtractor(nn.Module):
    """
    A feature extractor using a pre-trained CNN (ResNet50).
    It freezes the weights of the convolutional layers and uses the network
    up to the penultimate layer to get a feature vector for each frame.
    """
    def __init__(self, feature_size: int = 2048):
        super(CnnFeatureExtractor, self).__init__()
        # Load a pre-trained ResNet50 model
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        
        # Remove the final classification layer (the 'fc' layer)
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        
        # Freeze the parameters of the pre-trained layers
        for param in self.resnet.parameters():
            param.requires_grad = False
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to extract features from a batch of frames.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size * num_frames, C, H, W).
            
        Returns:
            torch.Tensor: Feature vectors of shape (batch_size * num_frames, feature_size).
        """
        # Get the features from the ResNet backbone
        with torch.no_grad():
            features = self.resnet(x)
        
        # Flatten the features to a vector
        features = features.view(features.size(0), -1)
        return features

class VideoClassifier(nn.Module):
    """
    The main video classifier model that combines the CNN extractor and LSTM.
    """
    def __init__(self, num_classes: int, feature_size: int = 2048, hidden_size: int = 512, num_lstm_layers: int = 2):
        """
        Initializes the model layers.
        
        Args:
            num_classes (int): The number of classes to predict.
            feature_size (int): The size of the feature vector from the CNN.
            hidden_size (int): The number of features in the LSTM hidden state.
            num_lstm_layers (int): The number of recurrent layers in the LSTM.
        """
        super(VideoClassifier, self).__init__()
        self.feature_extractor = CnnFeatureExtractor(feature_size)
        
        self.lstm = nn.LSTM(
            input_size=feature_size,
            hidden_size=hidden_size,
            num_layers=num_lstm_layers,
            batch_first=True,  # Input tensor shape: (batch_size, seq_len, input_size)
            dropout=0.5
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_size // 2, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the model.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_frames, C, H, W).
        
        Returns:
            torch.Tensor: The final classification scores of shape (batch_size, num_classes).
        """
        batch_size, num_frames, c, h, w = x.shape
        
        # 1. Reshape and extract features from all frames
        # Input to CNN: (batch_size * num_frames, C, H, W)
        frame_features = self.feature_extractor(x.view(-1, c, h, w))
        
        # 2. Reshape features back into a sequence
        # Input to LSTM: (batch_size, num_frames, feature_size)
        sequence_features = frame_features.view(batch_size, num_frames, -1)
        
        # 3. Pass the sequence through the LSTM
        # We only need the output of the last time step
        # lstm_out shape: (batch_size, num_frames, hidden_size)
        # hidden_state shape: (num_layers, batch_size, hidden_size)
        lstm_out, (hidden_state, _) = self.lstm(sequence_features)
        
        # We use the hidden state of the last layer
        last_hidden_state = hidden_state[-1]
        
        # 4. Pass the final hidden state to the classifier
        output = self.classifier(last_hidden_state)
        
        return output

if __name__ == '__main__':
    # --- Test the model with dummy data ---
    print("🧪 Testing the VideoClassifier model...")
    
    # Model parameters
    NUM_CLASSES = 101  # Example: UCF-101 dataset
    FEATURE_SIZE = 2048 # From ResNet50
    
    # Dummy input parameters
    BATCH_SIZE = 4
    NUM_FRAMES = 16  # Sequence length
    CHANNELS = 3
    HEIGHT, WIDTH = 224, 224
    
    # Create a dummy input tensor
    dummy_input = torch.randn(BATCH_SIZE, NUM_FRAMES, CHANNELS, HEIGHT, WIDTH)
    
    # Instantiate the model
    model = VideoClassifier(num_classes=NUM_CLASSES, feature_size=FEATURE_SIZE)
    
    print(f"\nModel Architecture:\n{model}\n")
    
    # Pass the dummy input through the model
    print(f"Input tensor shape: {dummy_input.shape}")
    output = model(dummy_input)
    print(f"Output tensor shape: {output.shape}")
    
    # Check if output shape is correct
    assert output.shape == (BATCH_SIZE, NUM_CLASSES)
    print("\n✅ Model test passed! Output shape is correct.")
