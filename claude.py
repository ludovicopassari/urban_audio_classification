import os
import pandas as pd
import torch
import torchaudio
import matplotlib.pyplot as plt


def preprocess_waveform(
        waveform: torch.Tensor,
        sample_rate: int,
        target_sample_rate: int = 16000,
        duration: int = 4,
        n_fft: int = 512,
        hop_length: int = 256,
        n_mels: int = 256
    ):
        """
        Applica il preprocessing a un tensore waveform.
        
        Args:
            waveform: torch.Tensor di forma (channels, samples)
            sample_rate: frequenza di campionamento originale
            target_sample_rate: frequenza target per il resampling
            duration: durata massima in secondi
            n_fft, hop_length, n_mels: parametri per MelSpectrogram
        
        Returns:
            mel_db: torch.Tensor di forma (1, n_mels, time) normalizzato
        """
        
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

        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True) 

        # 4. Calcolo Mel-spectrogram
        mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=target_sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            window_fn=torch.hann_window
        )(waveform)

        # 5. Log-scaling
        mel_spectrogram_db = torch.log1p(mel_spectrogram)

        # 6. Normalizzazione [0,1]
        mel_db = (mel_spectrogram_db - mel_spectrogram_db.min()) / (mel_spectrogram_db.max() - mel_spectrogram_db.min())

        # 7. Usa solo un canale → (1, n_mels, time)
        mel_db = mel_db[0:1, :, :]

        return mel_db

# Parametri dataset
metadata_file = "./dataset/UrbanSound8K.csv"
audio_dir = "./dataset"
target_class = "engine_idling"#"car_horn"#"gun_shot"#"jackhammer"#"engine_idling"#"siren"#"street_music"#"children_playing"#"air_conditioner"#"dog_bark"   # Cambia qui la classe che vuoi visualizzare
target_sample_rate = 22050



def plot_mel_spectrogram(spec, title, subplot_idx, total_plots, duration=4, sample_rate=22050, hop_length=256, n_mels=256):
    plt.subplot(2, 5, subplot_idx)  # 10 fold → 2 righe x 5 colonne
    
    # Calcolo asse del tempo (in secondi)
    num_frames = spec.shape[-1]
    time_axis = (num_frames * hop_length) / sample_rate
    
    # Mostra spettrogramma con assi uniformi
    plt.imshow(
        spec[0].numpy(),
        cmap='magma',
        origin='lower',
        aspect='auto',
        extent=[0, time_axis, 0, n_mels]
    )
    
    plt.title(title)
    plt.xlabel("Tempo (s)")
    plt.ylabel("Mel bins")

def main():
    metadata = pd.read_csv(metadata_file)

    plt.figure(figsize=(20, 8))
    
    for fold in range(1, 11):
        # Trova una riga con la classe target in questo fold
        row = metadata[(metadata["fold"] == fold) & (metadata["class"] == target_class)].iloc[0]
        audio_path = os.path.join(audio_dir, f"fold{fold}", row["slice_file_name"])

        # Carica audio
        waveform, sr = torchaudio.load(audio_path)
        

        # Spettrogramma mel
        mel_spec = preprocess_waveform(waveform=waveform,sample_rate=sr)
        
        # Plot
        plot_mel_spectrogram(mel_spec, f"Fold {fold}", fold, 10)

    plt.suptitle(f"Mel-spectrograms della classe '{target_class}' per ogni fold")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
