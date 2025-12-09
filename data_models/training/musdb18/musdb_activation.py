import os
import sys
sys.path.append("/home/ciona/projects/RCOLM/data_models/MyModels")  
from mymodels import MyModels
import tensorflow as tf
import librosa 
import numpy as np
import matplotlib as plt



# Normalizacja taka jak podczas treningu -----------------------
def to_norm(db):
    return (db + 80.0) / 80.0

def to_db(norm):
    return norm * 80.0 - 80.0


# trzeba to przerobić tak, żeby nie wchodziła ścieżka wav, a sam plik
def source_activation(model_path, wav_path,
                          patch_h=360, patch_w=1288,
                          hop=1288, sr=44100):   # <<< hop = 1288

    print("Ładowanie modelu…")
    model = tf.keras.models.load_model(model_path, compile=False)

    print("Ładowanie pliku WAV:", wav_path)
    audio, sr = librosa.load(wav_path, sr=sr, mono=True)

    print("Liczenie STFT…")
    S = librosa.stft(audio, n_fft=2048, hop_length=1024)  # <<< 2048, 1024
    mag = np.abs(S)
    phase = np.angle(S)

    S_db = librosa.amplitude_to_db(mag, ref=1.0, top_db=80)
    H, W = S_db.shape
    print("Wymiary spektrogramu:", S_db.shape)

    patches = []
    positions = []

    for t in range(0, W - patch_w + 1, hop):
        patch = S_db[:patch_h, t:t+patch_w]  # <<< identyczny slicing jak w DataPrep
        patch = patch[..., None]
        patches.append(patch)
        positions.append(t)

    patches = np.array(patches, dtype=np.float32)
    patches = to_norm(patches)

    print("Patchy do przetworzenia:", len(patches))
    print("Przepuszczam przez model…")
    preds = model.predict(patches, batch_size=2, verbose=1)

    preds_db = to_db(preds)

    stems_mag = [np.zeros((H, W), dtype=np.float32) for _ in range(4)]
    count = np.zeros(W, dtype=np.float32)

    for (t, pred) in zip(positions, preds_db):
        for s in range(4):
            patch_db = pred[..., s]                 # (360,1288)
            full_db = np.zeros((H, patch_w)) - 80.0
            full_db[:patch_h, :] = patch_db
            stems_mag[s][:, t:t+patch_w] += librosa.db_to_amplitude(full_db)
        count[t:t+patch_w] += 1

    count[count == 0] = 1.0
    stems_mag = [m / count for m in stems_mag]

    #os.makedirs(out_dir, exist_ok=True)
    names = ["vocals", "drums", "bass", "other"]

    print("Konwertuję na audio (ISTFT)…")

    stems_audio = []
    for s, name in enumerate(names):
        complex_S = stems_mag[s] * np.exp(1j * phase)
        audio_s = librosa.istft(complex_S, hop_length=1024)
        stems_audio.append(audio_s)

    print("Gotowe.")

    return stems_audio


