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


batch_size = 4
num_classes = 10
learning_rate = 0.001
num_epochs = 28
weight_decay = 1e-4

#base dataset dir
dataset_dir = Path("dataset")

# datasets for training and testing
training_data = AudioDS(data_path="dataset", folds=[1,2,3,4,5,6,7,8,9], sample_rate=44100, feature_ext_type='linear-spectrogram', training=True)
test_data = AudioDS(data_path="dataset", folds=[10], sample_rate=44100, feature_ext_type='linear-spectrogram', training=True)

# dataloader to wrap dataset with an iterable
train_loader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

#setting device for computation
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Using device:", device)

# instance of model
model = TorchModel(input_shape=(128, 345, 1),num_classes=num_classes)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)



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
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_audio_classifier.pth')
    
    return model




""" def train_model(model, train_loader, test_loader, criterion, optimizer, device, num_epochs=50, patience=5):

    best_test_loss = float("inf")
    patience_counter = 0
    history = {"train_loss": [], "val_loss": [], "accuracy": []}

    logger.info(f"Start Training. lr={learning_rate} weight_decay= {weight_decay}")
    for epoch in range(num_epochs):

        model.train()
        running_loss = 0.0

        for batch_idx, (spectrogram, labels) in enumerate(train_loader):
            spectrogram, labels = spectrogram.to(device), labels.to(device)
            
            # forward
            outputs = model(spectrogram)
            loss = criterion(outputs, labels)
            
            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
                
            running_loss += loss.item()
            if batch_idx % 10 == 0:
                logger.info(f"[Train] Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx}], Loss: {loss.item():.4f}")
        
        avg_train_loss = running_loss / len(train_loader)
        logger.warning(f"[Training Epoch Resume] Epoch [{epoch+1}/{num_epochs}] Training Loss: {avg_train_loss:.4f}")

        model.eval()
        test_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for spettrogram, labels in test_loader:
                spettrogram, labels = spettrogram.to(device), labels.to(device)
                    
                outputs = model(spettrogram)
                loss = criterion(outputs, labels)
                test_loss += loss.item()
                    
                # Predizioni
                _, predicted = torch.max(outputs, dim=1)  # [batch]
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            
        avg_test_loss = test_loss / len(test_loader)
        accuracy = 100 * correct / total

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_test_loss)
        history["accuracy"].append(accuracy)
        
        # implements early stopping mechanism
        if avg_test_loss < best_test_loss:
            best_test_loss = avg_test_loss
            patience_counter = 0
            torch.save(model.state_dict(), "best_model.pth")
            logger.info("Model improved. Saved best_model.pth")
        else:
            patience_counter += 1
            logger.warning(f"No improvement. Patience counter: {patience_counter}/{patience}")

            if patience_counter >= patience:
                logger.warning("Early stopping triggered!")
                logger.warning("Training Ended.")
                break
        
        logger.warning(f"[Test Epoch Resume] Epoch [{epoch+1}/{num_epochs}] Test Loss: {avg_test_loss:.4f}, Accuracy: {accuracy:.2f}%")

    logger.warning("Training Ended.") """

model = train_model(model, train_loader, test_loader,num_epochs=50)

