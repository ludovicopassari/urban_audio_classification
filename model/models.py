import torch
import torch.nn as nn
import numpy as np

class TorchModel(nn.Module):
    def __init__(self, input_shape=(513, 345, 2), num_classes=10, seed=42):
        super().__init__()
        
        # Numero di canali in input
        in_channels = input_shape[2]
        
        # Primo blocco convoluzionale
        self.conv1 = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # Riduce dimensioni di 2x
            nn.Dropout2d(0.1)
        )
        
        # Secondo blocco convoluzionale
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.1)
        )
        
        # Terzo blocco convoluzionale
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.2)
        )
        
        # Quarto blocco convoluzionale
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.2)
        )
        
        # Global Average Pooling per ridurre parametri
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
    
    def _get_conv_output(self, shape):
        # Dummy input per calcolare la dimensione del flatten
        dummy_input = torch.zeros(1, shape[2], shape[0], shape[1])
        output = self.features(dummy_input)
        return output.numel() // output.shape[0]  # dimensione per batch
    
    def forward(self, x):
        
        # Input: (batch_size, 1, n_mels, time_steps)
        x = self.conv1(x)    # (batch_size, 32, n_mels/2, time_steps/2)
        x = self.conv2(x)    # (batch_size, 64, n_mels/4, time_steps/4)
        x = self.conv3(x)    # (batch_size, 128, n_mels/8, time_steps/8)
        x = self.conv4(x)    # (batch_size, 256, n_mels/16, time_steps/16)
        
        # Global Average Pooling
        x = self.global_avg_pool(x)  # (batch_size, 256, 1, 1)
        x = x.view(x.size(0), -1)    # (batch_size, 256)
        
        # Classification
        x = self.classifier(x)       # (batch_size, n_classes)
        return x

