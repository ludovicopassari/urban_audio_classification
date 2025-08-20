
from dataset_utils import UrbanSoundDataset
from torch.utils.data import DataLoader
from pathlib import Path
import matplotlib.pyplot as plt

dataset_dir = Path("dataset")

training_data = UrbanSoundDataset(dataset_dir, [6], train=True )
train_loader = DataLoader(training_data, batch_size=4, shuffle=True)

for batch_idx, ((waveforms,no_mod), labels) in enumerate(train_loader):
    print(waveforms.shape)
    print(labels)

    spec_single = no_mod[0, 0]  
    spec_np = spec_single.detach().numpy()


    plt.figure(figsize=(10, 5))
    plt.imshow(spec_np, 
               aspect='auto',
               origin='lower',
               cmap='magma',
               interpolation='nearest')
    
    plt.colorbar(label='Magnitude (dB)')
    plt.xlabel('Time Frames')
    plt.ylabel('Frequency Bins')
    plt.tight_layout()
    plt.show()
