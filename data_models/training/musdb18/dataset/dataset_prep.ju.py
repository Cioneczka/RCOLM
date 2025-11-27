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




class DataPrepSTFT:

    def __init__(self,
                 dataset_path,
                 segment_duration=30,
                 sr=44100,
                 n_fft=2048,
                 hop_length=1024,
                 patch_h=360,
                 patch_w=1288,
                 hop_frames=1288):

        self.dataset_path = dataset_path
        self.segment_duration = segment_duration
        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.patch_h = patch_h
        self.patch_w = patch_w
        self.hop_frames = hop_frames

        self.X_list = []
        self.Y_list = []

    # -------------------------
    #   Helper: STFT → dB
    # -------------------------
    def to_db(self, audio):
        S = librosa.stft(audio,
                         n_fft=self.n_fft,
                         hop_length=self.hop_length)
        mag = np.abs(S)
        S_db = librosa.amplitude_to_db(mag, ref=1.0, top_db=80)
        return S_db

    # -------------------------
    #  GŁÓWNA METODA
    # -------------------------
    def build(self):

        for split in ["train", "test"]:
            split_path = os.path.join(self.dataset_path, split)

            for track_name in os.listdir(split_path):
                track_path = os.path.join(split_path, track_name)
                if not os.path.isdir(track_path):
                    continue

                if "mixture.wav" not in os.listdir(track_path):
                    continue
                print(f"\n Loading track:{track_name}")
                # Load audios
                mix, _ = librosa.load(os.path.join(track_path, "mixture.wav"), sr=self.sr, mono=True)
                bass,_ = librosa.load(os.path.join(track_path, "bass.wav"),    sr=self.sr, mono=True)
                drums,_= librosa.load(os.path.join(track_path, "drums.wav"),   sr=self.sr, mono=True)
                other,_= librosa.load(os.path.join(track_path, "other.wav"),   sr=self.sr, mono=True)
                vocals,_=librosa.load(os.path.join(track_path, "vocals.wav"),  sr=self.sr, mono=True)

                segment_samples = int(self.segment_duration * self.sr)
                num_segments = len(mix) // segment_samples

                for seg_idx in range(num_segments):
                    start = seg_idx * segment_samples
                    stop = start + segment_samples

                    # Normalize segment
                    peak = np.max(np.abs(mix[start:stop])) + 1e-9

                    mix_seg    = mix[start:stop]    / peak
                    bass_seg   = bass[start:stop]   / peak
                    drums_seg  = drums[start:stop]  / peak
                    other_seg  = other[start:stop]  / peak
                    vocals_seg = vocals[start:stop] / peak

                    # STFT → dB
                    mix_db    = self.to_db(mix_seg)
                    bass_db   = self.to_db(bass_seg)
                    drums_db  = self.to_db(drums_seg)
                    other_db  = self.to_db(other_seg)
                    vocals_db = self.to_db(vocals_seg)

                    # -----------------------
                    # Patchowanie (360 × 1288)
                    # -----------------------
                    H, W = mix_db.shape

                    for t in range(0, W - self.patch_w + 1, self.hop_frames):

                        X_patch = mix_db[:self.patch_h, t:t+self.patch_w]
                        Y_patch = np.stack([
                            bass_db[:self.patch_h, t:t+self.patch_w],
                            drums_db[:self.patch_h, t:t+self.patch_w],
                            other_db[:self.patch_h, t:t+self.patch_w],
                            vocals_db[:self.patch_h, t:t+self.patch_w]
                        ], axis=-1)

                        X_patch = X_patch[..., None]  # (360,1288,1)

                        self.X_list.append(X_patch)
                        self.Y_list.append(Y_patch)

        X = np.stack(self.X_list, axis=0)
        Y = np.stack(self.Y_list, axis=0)
        return X, Y

    def save_data(self, out_path):
        X, Y = self.build()
        print(f"Saving data...")
        np.savez_compressed(out_path, X=X, Y=Y)
        print("Dataset saved to:", out_path)


dp = DataPrepSTFT(dataset_path = "/home/ciona/projects/RCOLM/data/raw_data/musdb18")
 
dp.save_data("/home/ciona/projects/RCOLM/data/raw_data/musdb18/dataset/dataset.npz")



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
