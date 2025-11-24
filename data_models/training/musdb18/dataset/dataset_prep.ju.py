# %%
import os
import librosa
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt 

# %%



# %% [md]
##  Main funkction

# %%   



class DataPrep:

    def __init__(self,
                 dataset_path,
                 segment_duration=30,
                 sr=44100,
                 hop_length=1024,
                 fmin=32.7,
                 n_bins=360,
                 bins_per_octave=60):

        self.dataset_path = dataset_path
        self.segment_duration = segment_duration
        self.sr = sr
        self.hop_length = hop_length
        self.fmin = fmin
        self.n_bins = n_bins
        self.bins_per_octave = bins_per_octave

        # listy na dane
        self.X_list = []
        self.Y_list = []

    # -------------------------
    #   HELPER: CQT
    # -------------------------
    def seg_to_cqt(self, y):
        C = np.abs(librosa.cqt(
            y=y,
            sr=self.sr,
            hop_length=self.hop_length,
            fmin=self.fmin,
            n_bins=self.n_bins,
            bins_per_octave=self.bins_per_octave
        ))
        C_db = librosa.amplitude_to_db(C, ref=np.max)
        return C_db

    # -------------------------
    #   HELPER: normalizacja segmentu
    # -------------------------
    @staticmethod
    def normalize_segment(y):
        return y / (np.max(np.abs(y)) + 1e-9)

    # -------------------------
    #   GŁÓWNA METODA
    # -------------------------
    def build(self):

        print(tf.__version__) 
        for split in ["train", "test"]:  
            split_path = os.path.join(self.dataset_path, split)

            for track_name in os.listdir(split_path):
                track_path = os.path.join(split_path, track_name)

                # musi być katalog z utworem
                if not os.path.isdir(track_path):
                    continue

                # lista plików w utworze
                files = os.listdir(track_path)

                # sprawdzamy czy są stem’y
                if "mixture.wav" not in files:
                    continue

                mix_path    = os.path.join(track_path, "mixture.wav")
                bass_path   = os.path.join(track_path, "bass.wav")
                drums_path  = os.path.join(track_path, "drums.wav")
                other_path  = os.path.join(track_path, "other.wav")
                vocals_path = os.path.join(track_path, "vocals.wav")

                print("Przetwarzam:", track_path)

        # --- dalej Twoje segmentowanie i CQT ---
                # load
                mix, sr = librosa.load(mix_path, sr=self.sr, mono=True) 
                bass,_   = librosa.load(bass_path,   sr=self.sr, mono=True)
                drums,_  = librosa.load(drums_path,  sr=self.sr, mono=True)
                other,_  = librosa.load(other_path,  sr=self.sr, mono=True)
                vocals,_ = librosa.load(vocals_path, sr=self.sr, mono=True)

                # segmentation
                segment_samples = int(self.segment_duration * self.sr)
                num_segments = len(mix) // segment_samples

                for seg_idx in range(num_segments):

                    start = seg_idx * segment_samples
                    stop = start + segment_samples

                    # waveform segments
                    mix_seg    = self.normalize_segment(mix[start:stop])
                    bass_seg   = self.normalize_segment(bass[start:stop])
                    drums_seg  = self.normalize_segment(drums[start:stop])
                    other_seg  = self.normalize_segment(other[start:stop])
                    vocals_seg = self.normalize_segment(vocals[start:stop])

                    # CQT for each segment
                    mix_cqt    = self.seg_to_cqt(mix_seg)
                    bass_cqt   = self.seg_to_cqt(bass_seg)
                    drums_cqt  = self.seg_to_cqt(drums_seg)
                    other_cqt  = self.seg_to_cqt(other_seg)
                    vocals_cqt = self.seg_to_cqt(vocals_seg)

                    # shape (n_bins, T, 1)
                    X_seg = mix_cqt[..., np.newaxis]

                    # shape (n_bins, T, 4)
                    Y_seg = np.stack(
                        [bass_cqt, drums_cqt, other_cqt, vocals_cqt],
                        axis=-1
                    )

                    self.X_list.append(X_seg)
                    self.Y_list.append(Y_seg)

        # stack all
        X = np.stack(self.X_list, axis=0)
        Y = np.stack(self.Y_list, axis=0)

        return X, Y
    def save_data(self, path="/home/ciona/projects/RCOLM/data/raw_data/musdb18/dataset"):
        X, Y = self.build()
        np.savez_compressed(path, X=X, Y=Y)
        print("Zapisano dataset:", path)
        

dp = DataPrep(dataset_path = "/home/ciona/projects/RCOLM/data/raw_data/musdb18")
 
dp.save_data()



# data = np.load("/home/ciona/projects/RCOLM/data/raw_data/musdb18/dataset/dataset.npz")
# 
# print(data.files)  # jakie są klucze? (np. ['X', 'Y'])
# 
# X = data["X"]
# Y = data["Y"]
# 
# print("X shape:", X.shape)
# print("Y shape:", Y.shape)
# 
