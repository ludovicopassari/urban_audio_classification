import torch
import torch.nn as nn
import numpy as np

class TorchModel(nn.Module):
    def __init__(self, input_shape=(513, 345, 2), num_classes=10, seed=42):
        super().__init__()
        
        # Numero di canali in input
        in_channels = input_shape[2]
        
        # Convolutional layers
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25),
            
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25),

            
        )
        
        # Calcolo dinamico della dimensione dopo i convoluzionali
        #self.flattened_size = self._get_conv_output(input_shape)
        self.global_pool = nn.AdaptiveAvgPool2d((1,1))

        # fully connected layer
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
    
    def _get_conv_output(self, shape):
        # Dummy input per calcolare la dimensione del flatten
        dummy_input = torch.zeros(1, shape[2], shape[0], shape[1])
        output = self.features(dummy_input)
        return output.numel() // output.shape[0]  # dimensione per batch
    
    def forward(self, x):
        # Se input in formato TF (batch, H, W, C), converte in (batch, C, H, W)
        if x.ndim == 4 and x.shape[-1] == self.features[0].in_channels:
            x = x.permute(0, 3, 1, 2)
        
        x = self.features(x)
        x = self.global_pool(x)
        x = self.classifier(x)
        return x

