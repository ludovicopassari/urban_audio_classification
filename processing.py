from dataset_utils.AudioDS import AudioDS
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch


def plot_waveform(waveform, sr, title="Waveform", ax=None):
    waveform = waveform.numpy()

    num_channels, num_frames = waveform.shape
    time_axis = torch.arange(0, num_frames) / sr

    if ax is None:
        _, ax = plt.subplots(num_channels, 1)
    ax.plot(time_axis, waveform[0], linewidth=1)
    ax.grid(True)
    ax.set_xlim([0, time_axis[-1]])
    ax.set_title(title)


def plot_spectrogram(specgram, title=None, ylabel="freq_bin", ax=None):
    if ax is None:
        _, ax = plt.subplots(1, 1)
    if title is not None:
        ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.imshow(specgram, origin="lower", aspect="auto", interpolation="nearest")


#test params
batch_size=1

train_dataset = AudioDS(data_path="dataset", folds=[1,2,3,4,5,6,7,8,9,10], sample_rate=44100, feature_ext_type='mel-spectrogram', training=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size,shuffle=True)


for spec, label in train_loader:

    print(spec.shape)
    fig, axs = plt.subplots(2, 1, figsize=(12, 6))


    plot_spectrogram(spec[0,0], title="Mel Scale Spectrogram", ax=axs[1])
    

    plt.tight_layout()
    plt.show()
    break