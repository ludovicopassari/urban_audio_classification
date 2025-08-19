from dataset_utils import UrbanSoundDataset
from torch.utils.data import DataLoader
from pathlib import Path
import matplotlib.pyplot as plt

dataset_dir = Path("dataset")

training_data = UrbanSoundDataset(dataset_dir, [6], sample_rate=16000)
train_loader = DataLoader(training_data, batch_size=4, shuffle=True)

for batch_idx, (waveforms, labels) in enumerate(train_loader):
    print(waveforms.shape)
    print(labels)

    spec_single = waveforms[0, 0]  
    spec_np = spec_single.detach().numpy()

    plt.figure(figsize=(12, 6))
    plt.imshow(spec_np, 
               aspect='auto',
               origin='lower',
               cmap='viridis',
               interpolation='nearest')
    
    plt.colorbar(label='Magnitude (dB)')
    plt.xlabel('Time Frames')
    plt.ylabel('Frequency Bins')
    plt.tight_layout()
    plt.show()
    break
