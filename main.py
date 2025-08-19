import torch
from torch.utils.data import DataLoader
from dataset_utils import UrbanSoundDataset

from pathlib import Path

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset_utils import UrbanSoundDataset

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

log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

# Nome file log
log_file = os.path.join(log_dir, "training.log")

# Configurazione logging
logging.basicConfig(
    level=logging.INFO,  # INFO, DEBUG, WARNING, ERROR
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file),  # salva su file
        logging.StreamHandler()         # mostra a video
    ]
)

logger = logging.getLogger()

#base dataset dir
dataset_dir = Path("dataset")

# datasets for training and testing
training_data = UrbanSoundDataset(dataset_dir, [1,2,3,4,5,6,7,8,9], sample_rate=16000)
test_data = UrbanSoundDataset(dataset_dir, [10],sample_rate=16000)

# dataloader to wrap dataset with an iterable
train_loader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

#setting device for computation
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Using device:", device)

# instance of model
model = TorchModel(input_shape=(128, 173, 1),num_classes=num_classes).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)


def train_model(model, train_loader, test_loader, criterion, optimizer, device, num_epochs=50, patience=5):

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

    logger.warning("Training Ended.")

history = train_model(model, train_loader, test_loader, criterion, optimizer, device, num_epochs=50, patience=7)

logger.info(f"History : {history}")

# Crea DataFrame
df = pd.DataFrame(history)
df.to_csv("training_history.csv", index=False)