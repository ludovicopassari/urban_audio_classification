from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import librosa
import os
import torchaudio
import torch

"""

RangeIndex: 8732 entries, 0 to 8731
Data columns (total 8 columns):
 #   Column           Non-Null Count  Dtype  
---  ------           --------------  -----  
 0   slice_file_name  8732 non-null   object 
 1   fsID             8732 non-null   int64  
 2   start            8732 non-null   float64
 3   end              8732 non-null   float64
 4   salience         8732 non-null   int64  
 5   fold             8732 non-null   int64  
 6   classID          8732 non-null   int64  
 7   class            8732 non-null   object 
dtypes: float64(2), int64(4), object(2)
memory usage: 545.9+ KB

"""


def main():
    base_path = Path(__file__).resolve().parent.parent
    dataset_path = base_path / 'dataset'

    df = pd.read_csv(dataset_path / "UrbanSound8K.csv")

    #drop unused columns to save memory 
    df.drop(['fsID', 'start', 'end', 'salience','slice_file_name'], axis='columns', inplace= True)

    class_count = df.value_counts(subset=['class']).sort_values()
    
    grouped = df.groupby(by=['fold', 'class']).size().unstack(fill_value=0)

    max_freqs = []

    for fold in range(1, 11):
        fold_path = os.path.join(dataset_path, f"fold{fold}")
        for file in os.listdir(fold_path):
            if file.endswith(".wav"):
                file_path = os.path.join(fold_path, file)
                waveform, sr = torchaudio.load(file_path)
                waveform = waveform.mean(dim=0)  # media canale se stereo
                
                # FFT
                fft = torch.fft.rfft(waveform)
                
                magnitude = torch.abs(fft)
                
                # Frequenze corrispondenti
                freqs = torch.fft.rfftfreq(len(waveform), d=1/sr)
                
                # frequenza con massima energia
                max_freq = freqs[magnitude.argmax()].item()
                max_freqs.append(max_freq)

    # Plot istogramma delle frequenze dominanti
    plt.figure(figsize=(8,5))
    plt.hist(max_freqs, bins=50, color='skyblue', edgecolor='k')
    plt.title("Distribuzione della frequenza dominante nei file audio")
    plt.xlabel("Frequenza (Hz)")
    plt.ylabel("Numero di file")
    plt.show()

    """ data = []

    for fold in range(1, 11):
        fold_path = os.path.join(dataset_path, f"fold{fold}")
        for file in os.listdir(fold_path):
            if file.endswith(".wav"):
                file_path = os.path.join(fold_path, file)
                try:
                    info = torchaudio.info(file_path)
                    sr = info.sample_rate
                    data.append({"file": file, "fold": fold, "samplerate": sr})
                except Exception as e:
                    print(f"Errore su {file_path}: {e}")

    df = pd.DataFrame(data)

    # conteggi delle frequenze
    freq_counts = df["samplerate"].value_counts().sort_index()

    # grafico a barre
    plt.figure(figsize=(8,5))
    freq_counts.plot(kind="bar")
    plt.title("Frequenze di campionamento nel dataset UrbanSound8K")
    plt.xlabel("Frequenza di campionamento (Hz)")
    plt.ylabel("Numero di file")
    plt.xticks(rotation=0)
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()
  
    grouped.plot(kind='bar', stacked=True, figsize=(12,6), edgecolor='black')
    plt.title("Distribuzione delle classi in ciascun fold")
    plt.xlabel("Fold")
    plt.ylabel("Numero di esempi")
    plt.legend(title="Classe", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=0)
    plt.tight_layout()
   
    plt.figure(figsize=(8,6))
    class_count.plot(kind='bar')
    plt.title("Numero di esempi per classe")
    plt.xlabel("Classe")
    plt.ylabel("Conteggio")
    plt.tight_layout()
    plt.show()  """
  
    


   
if __name__ == "__main__":
    main()