from torch.utils.data import Dataset
import torchaudio
from pathlib import Path
import pandas as pd
import torch
import librosa
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler

class AudioDS(Dataset):
    def __init__(self, data_path, folds, sample_rate, feature_ext_type:str, max_duration=4,training=False, aug=True):
        self._data_path = Path(data_path)
        self._folds = folds
        self._sample_rate = sample_rate
        self._max_duration = max_duration
        self._train = training
        self._target_len = sample_rate * max_duration
        self._feature_ext_type = feature_ext_type
        self._metadata = self._load_metadata()
        self._augmentation = aug

        #feature extraction 
        n_fft = 1024
        win_length = None
        hop_length = 512
        n_mels = 128
        
        self.mel_transform= torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            center=True,
            pad_mode="reflect",
            power=2.0,
            norm="slaney",
            n_mels=n_mels,
            mel_scale="htk",
            f_max= self._sample_rate /2,
            f_min=0
            
        )
        self.spectrogram_transform = torchaudio.transforms.Spectrogram(n_fft=512)
        self.db_transform = None

        #data augmentation
        self.time_masking = torchaudio.transforms.TimeMasking(time_mask_param=40)
        self.freq_masking = torchaudio.transforms.FrequencyMasking(freq_mask_param=40)
        self.pich_shifter= None


    def __len__(self):
        return len(self._metadata)
    
    def __getitem__(self,idx):
        row = self._metadata.iloc[idx]
        label = row['classID']

        #build sample path
        audio_full_path = self._data_path / f"fold{row['fold']}" / row['slice_file_name']

        raw_waveform, raw_sample_rate = self._load_sample(audio_full_path)
        waveform = raw_waveform

        if raw_sample_rate != self._sample_rate:
            waveform = self._resample_waveform(waveform=raw_waveform, orig_freq=raw_sample_rate, new_freq= self._sample_rate)

        if self._target_len != waveform.shape[0]:
            waveform = self._fix_lenght(waveform)

        if self._train and self._augmentation and torch.rand(1).item() < 0.3:
            
            snr_db = torch.randint(low=10, high=20, size=(1,)).item()
            waveform =self._add_noise_gaussian(waveform, snr_db)

            rand_shift = torch.randint(low=1000, high=6400, size=(1,)).item()
            waveform = self.random_shift(waveform,rand_shift)

            #waveform = self.pich_shifter(waveform)


        waveform = waveform / waveform.abs().max()  # scala tra -1 e 1
      

        if self._feature_ext_type == 'linear-spectrogram':
            raw_spectrogram = self.spectrogram_transform(waveform)
        elif self._feature_ext_type == 'mel-spectrogram':
            raw_spectrogram = self.mel_transform(waveform)

        spec= librosa.power_to_db(raw_spectrogram)
        spec = (spec - spec.min()) / (spec.max() - spec.min())

        #print("Valore massimo:", spec.max().item())
        #print("Valore minimo:", spec.min().item())

        spec = torch.tensor(spec, dtype=torch.float32)

        if self._augmentation and self._train and torch.rand(1).item() < 0.3:
            #spec = self.freq_masking(spec)
            spec = self.time_masking(spec)

        return spec, label
    

    def random_shift(self, waveform, max_shift):
        """Applica uno shift temporale casuale al waveform."""
        shift = torch.randint(-max_shift, max_shift + 1, (1,)).item()
        if shift == 0:
            return waveform  # niente shift
        
        # numero di canali e campioni
        channels, samples = waveform.shape

        if shift > 0:
            # shift a destra
            pad = torch.zeros((channels, shift), device=waveform.device)
            waveform = torch.cat([pad, waveform[:, :-shift]], dim=1)
        else:  # shift < 0
            # shift a sinistra
            pad = torch.zeros((channels, -shift), device=waveform.device)
            waveform = torch.cat([waveform[:, -shift:], pad], dim=1)
        
        return waveform
    
    def _add_noise_gaussian(self,speech, snr_db):
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


    def _load_sample(self, path):

        samples, sr = torchaudio.load(path) #[channels, samples]
        
        if samples.shape[0] > 1:
            samples = samples.mean(dim=0, keepdim=True) 
    
        return samples, sr

    def _resample_waveform(self, waveform, orig_freq, new_freq):
        #stando alla documentazione 'sinc_interp_kaiser' è più preciso alle alte frequenze
        resampler = torchaudio.transforms.Resample(orig_freq=orig_freq, new_freq=new_freq, resampling_method='sinc_interp_kaiser')
        return resampler(waveform)
    
    def _fix_lenght(self, waveform):
        
        #numero di campioni nel segnale audio raw
        num_samples = waveform.shape[1]

        #se il segnale audio è più lungo di quello che mi serve
        if num_samples > self._target_len:
            if self._train:
                start = torch.randint(0, num_samples - self._target_len + 1, (1,)).item()
            else:
                start = (num_samples - self._target_len) // 2
            waveform = waveform[:, start:start + self._target_len]
        elif num_samples < self._target_len:
            # Pad con zeri
            pad_len = self._target_len - num_samples
            waveform = torch.nn.functional.pad(waveform, (0, pad_len))
            
        return waveform



    def _process_waveform(self, waveform):
        pass

    def _load_metadata(self):
        metadata_file_path = self._data_path / "UrbanSound8K.csv"
        df = pd.read_csv(metadata_file_path)
        df = df[df['fold'].isin(self._folds)]
        return df