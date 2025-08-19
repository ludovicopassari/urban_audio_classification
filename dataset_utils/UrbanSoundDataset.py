import pandas as pd

import torchaudio
import torch
from torch.utils.data import Dataset
from torchaudio.transforms import Spectrogram, MelSpectrogram, MFCC
from typing import List, Union
from pathlib import Path

class UrbanSoundDataset(Dataset):
    def __init__(self, dataset_dir: Union[str, Path], folds: List[int], sample_rate = 16000)-> None:
        self.dataset_dir = Path(dataset_dir)
        self.folds = folds
        self.target_sample_rate = sample_rate
        self.metadata = self.__load_metadata()
        
    def __len__(self):
        return len(self.metadata)

    def __getitem__(self,idx):
        """
        Get examlple from dataset ad ID 'idx'.
        Return preprocessed example an its lable 

        """
        row = self.metadata.iloc[idx]
        audio_path = self.dataset_dir / f"fold{row['fold']}" / row['slice_file_name']

        # this loads audio example in memory
        waveform, sample_rate = torchaudio.load(audio_path) #[channels, samples]

        #this resample audio if the sample rate is different from the one i use
        if sample_rate != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=self.target_sample_rate)
            waveform = resampler(waveform)
            
        max_len = self.target_sample_rate * 4 # max number of samples in 4s audio
        if waveform.shape[1] < max_len:
            pad_len = max_len - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, pad_len))  # add padding to right
        else:
            waveform = waveform[:, :max_len]  # cut if the audio is too long

        if waveform.shape[0] == 1:
            waveform = waveform.repeat(2, 1)    # create second channel by duplicating the first one
        elif waveform.shape[0] > 2:
            waveform = waveform[:2, :]  # if the audio have more than 2 channel this slice only the first 2. """

        


        mel_transform =  torchaudio.transforms.MelSpectrogram(
            sample_rate=self.target_sample_rate,
            n_fft= 1024,
            hop_length=256,
            #win_length=512,
            n_mels=256, # numero di bande Mel
            window_fn=torch.hann_window,
            power=2.0,
            f_min=10, f_max=8000

        )
        mel_spectrogram = mel_transform(waveform) 
        mel_spectrogram = torch.log1p(mel_spectrogram) 
        
        mel_min = mel_spectrogram.min()
        mel_max = mel_spectrogram.max()
        mel_spectrogram = (mel_spectrogram - mel_min) / (mel_max - mel_min + 1e-6)

        print(row['slice_file_name'])
        #print(row['fold'])
        
        label = row['classID']

        return mel_spectrogram, label

    def __load_metadata(self):
        metadata_file_path = self.dataset_dir / "UrbanSound8K.csv"
        df = pd.read_csv(metadata_file_path)
        df = df[df['fold'].isin(self.folds)]
        return df


