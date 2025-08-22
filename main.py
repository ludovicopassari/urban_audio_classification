import torch
from torch.utils.data import DataLoader
from dataset_utils import UrbanSoundDataset

from pathlib import Path

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset_utils.AudioDS import AudioDS

from pathlib import Path
from model.models import TorchModel
import os
import logging
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime


batch_size = 32
num_classes = 10
learning_rate = 0.001
num_epochs = 50
weight_decay = 1e-4

#base dataset dir
dataset_dir = Path("dataset")

# datasets for training and testing
training_data = AudioDS(data_path="dataset", folds=[1,2,3,4,5,6,7,8,9], sample_rate=44100, feature_ext_type='mel-spectrogram', training=True)
test_data = AudioDS(data_path="dataset", folds=[10], sample_rate=44100, feature_ext_type='mel-spectrogram')

# dataloader to wrap dataset with an iterable
train_loader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

#setting device for computation
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Using device:", device)

# instance of model
model = TorchModel(input_shape=( 257, 690, 1),num_classes=num_classes)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

# Estrai metriche
train_acc_history = []
val_acc_history   = []
train_loss_history = []
val_loss_history   = []

def train_loop(dataloader, model, loss_fn, optimizer):
    pass

def train_model(model, train_loader, val_loader, num_epochs=50, learning_rate=0.001):
    """Funzione di training del modello"""
    size = len(train_loader.dataset)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
    
    best_val_acc = 0.0
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (data, targets) in enumerate(train_loader):
            data, targets = data.to(device), targets.to(device)

            # this reset the gradients of model params
            optimizer.zero_grad()
            #this forward input throwgh model
            outputs = model(data)
            #compute loss
            loss = criterion(outputs, targets)

            #compute gradient's params based on loss_function
            loss.backward()
            #update model's params
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += targets.size(0)
            train_correct += (predicted == targets).sum().item()

            if batch_idx % 100 == 0:
                loss, current = loss.item(), batch_idx * batch_size + len(data)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
        
        train_acc = 100 * train_correct / train_total

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for data, targets in val_loader:
                data, targets = data.to(device), targets.to(device)
                outputs = model(data)
                loss = criterion(outputs, targets)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += targets.size(0)
                val_correct += (predicted == targets).sum().item()
        
        val_acc = 100 * val_correct / val_total
        
        # Scheduler step
        scheduler.step(val_loss)
        
        print(f'Epoch [{epoch+1}/{num_epochs}]')
        print(f'Train Loss: {train_loss/len(train_loader):.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss/len(val_loader):.4f}, Val Acc: {val_acc:.2f}%')
        print('-' * 50)


        #tracking values
        train_acc_history.append(train_acc)
        train_loss_history.append(train_loss/len(train_loader))

        val_acc_history.append(val_acc)
        val_loss_history.append(val_loss/len(val_loader))


        
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_audio_classifier.pth')
    
    return model



model = train_model(model, train_loader, test_loader,num_epochs=num_epochs)

epochs= list(range(1, num_epochs + 1))

# genera stringa con data e ora
now = datetime.now()
timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")  # esempio: 2025-08-22_15-30-45

# --- Plot Accuracy ---
plt.figure(figsize=(10,5))
plt.plot(epochs, train_acc_history, label="Train Accuracy")
plt.plot(epochs, val_acc_history, label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Accuracy per Epoch")
plt.legend()
plt.grid(True)
# salva il grafico con data e ora nel nome
plt.savefig(f"accuracy_{timestamp}.png", dpi=300)
plt.show()

# --- Plot Loss ---
plt.figure(figsize=(10,5))
plt.plot(epochs, train_loss_history, label="Train Loss")
plt.plot(epochs, val_loss_history, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss per Epoch")
plt.legend()
plt.grid(True)

# salva il grafico con data e ora nel nome
plt.savefig(f"error_{timestamp}.png", dpi=300)
plt.show()