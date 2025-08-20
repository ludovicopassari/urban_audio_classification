import pandas as pd

import torchaudio
import torch
from torch.utils.data import Dataset
from torchaudio.transforms import Spectrogram, MelSpectrogram, MFCC
from typing import List, Union
from pathlib import Path


class UrbanSoundDataset(Dataset):
    def __init__(self, dataset_dir: str, 
                folds: List[int], 
                sample_rate:int=22050,
                max_duration:float = 4.0, 
                train:bool=False,
                n_mels=200, n_fft=2048, hop_length=256 )-> None:

        
        self.dataset_dir = Path(dataset_dir)
        self.folds = folds
        self.sample_rate = sample_rate
        self.metadata = self.__load_metadata()
        self.train = train
        
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            power=2.0,
            f_max= sample_rate / 2,
            f_min=20,

        )
        self.time_masking = torchaudio.transforms.TimeMasking(time_mask_param=10)
        self.freq_masking = torchaudio.transforms.FrequencyMasking(freq_mask_param=10)
        self.db_transform = torchaudio.transforms.AmplitudeToDB()

        
    def __len__(self):
        return len(self.metadata)

    def __getitem__(self,idx):

        row = self.metadata.iloc[idx]
        audio_path = self.dataset_dir / f"fold{row['fold']}" / row['slice_file_name']

        # this loads audio example in memory
        waveform, sample_rate = torchaudio.load(audio_path) #[channels, samples]
        label = row['classID']

        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True) 

        if self.train and torch.rand(1).item() < 0.5:
            snr_db = torch.randint(low=5, high=20, size=(1,)).item()
            waveform =self.add_noise_gaussian(waveform, snr_db)

        spec = self.__preprocess_waveform(waveform, sample_rate, n_fft=2048)

        return spec, label

    def __load_metadata(self):
        metadata_file_path = self.dataset_dir / "UrbanSound8K.csv"
        df = pd.read_csv(metadata_file_path)
        df = df[df['fold'].isin(self.folds)]
        return df
    
    def __load_audio(self, path:str):
        pass

    def __preprocess_waveform(self,
        waveform: torch.Tensor,
        sample_rate: int,
        target_sample_rate: int = 22050,
        duration: int = 4,
        n_fft: int = 1024,
        hop_length: int = 512,
        n_mels: int = 128):  

        # 1. Resample se necessario
        if sample_rate != target_sample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sample_rate)
            waveform = resampler(waveform)
        
        # 2. Padding o cut a durata fissa
        max_len = target_sample_rate * duration
        if waveform.shape[1] < max_len:
            pad_len = max_len - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, pad_len))
        else:
            waveform = waveform[:, :max_len]
            

       
        mel_spectrogram = self.mel_transform(waveform)
        mel_spectrogram_db = self.db_transform(mel_spectrogram)

        if self.train and torch.rand(1).item() < 0.5:
            mel_spectrogram_db = self.time_masking(mel_spectrogram_db)
            mel_spectrogram_db = self.freq_masking(mel_spectrogram_db)

        # 6. Normalizzazione [0,1]
        mel_db = (mel_spectrogram_db - mel_spectrogram_db.min()) / (mel_spectrogram_db.max() - mel_spectrogram_db.min())

        return mel_db
    

    def add_noise_gaussian(self,speech, snr_db):
        # Generiamo rumore casuale
        noise = torch.randn_like(speech)

        # Calcoliamo potenza segnale e rumore
        power_speech = speech.pow(2).mean()
        power_noise = noise.pow(2).mean()

        # Calcoliamo il fattore di scala del rumore per ottenere SNR desiderato
        snr = 10 ** (snr_db / 10)
        scale = torch.sqrt(power_speech / (snr * power_noise))
        noise_scaled = noise * scale

        # Sommiamo rumore al segnale
        noisy_speech = speech + noise_scaled
        return noisy_speech
