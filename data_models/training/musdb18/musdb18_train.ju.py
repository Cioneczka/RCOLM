# %%
import numpy as np 
import tensorflow as tf
import keras 
from sklearn.model_selection import train_test_split
import librosa
import datetime
import matplotlib.pyplot as plt
import sys
sys.path.append("/home/ciona/projects/RCOLM/data_models/MyModels")  
from mymodels import MyModels

import os

from .paths import (
    SAVE_INPUT_PATH_FILE,
    SAVE_MEL_DIR,
    SAVE_CHROMA_DIR,
    MUSDB_DIR,
)
# %% [md]
##  Main funkction

# %%
class Train:





    @staticmethod
    def create_dataset(X, y, batch_size):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)

        print("Creating dataset from X.shape =", X.shape, " y.shape =", y.shape, flush=True)

        # Dataset tworzony jawnie na CPU
        with tf.device('/CPU:0'):
            ds = tf.data.Dataset.from_tensor_slices((X, y))

        ds = ds.shuffle(buffer_size=min(1024, len(X)))
        ds = ds.batch(batch_size)
        ds = ds.prefetch(tf.data.AUTOTUNE)
        return ds
    @staticmethod
    def save_model(model, model_path):
        timestamp = datetime.datetime.now()
        filename = f"unet_model{timestamp}.h5"
        full_path = os.path.join(model_path, filename)
        model.save(full_path)
        print(f"Model saved as:{full_path}")


    @staticmethod
    def training():
        DATA_PATH = "/home/ciona/projects/RCOLM/data/raw_data/musdb18/dataset/dataset.npz"
        data = np.load(DATA_PATH)
        X = data["X"].astype("float32")
        y = data["Y"].astype("float32")

        print("X shape:", X.shape)
        print("y shape:", y.shape)



        # Wejściowy kształt (np. 360x216x1)
        H, W, C_in = X.shape[1], X.shape[2], X.shape[3]
        C_out = y.shape[3]   # liczba kanałów wyjściowych (stemy / maski)
        n = X.shape[0]

        input_shape = (H, W, C_in)

        # Budowa modelu (jeśli build_unet ma num_bins, to przekaż C_out)
        model = MyModels.build_unet(input_shape, num_bins=C_out)
        model.compile(
            optimizer='adam',
            loss='mae',
            metrics=[
                tf.keras.metrics.MeanSquaredError(name="mse"),
                tf.keras.metrics.MeanAbsoluteError(name="mae"),
            ],
        )



        print("Dataset fragment:")
        print("X shape:", X.shape)
        print("y shape:", y.shape)




        # po przycięciu trzeba zaktualizować n
        n = X.shape[0]

        chunk_size = 100
        epochs_per_chunk = 20
        FLOOR_DB = -80
        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            print(f"\nTrenuję na próbkach {start}:{end}")
            
            X_chunk = X[start:end]
            y_chunk = y[start:end]
            X_chunk = (X_chunk - FLOOR_DB) / -FLOOR_DB
            y_chunk = (y_chunk - FLOOR_DB) / -FLOOR_DB
            print(f"\nPróbki znormalizowane do [0,1]")

            X_train, X_test, y_train, y_test = train_test_split(
                X_chunk, y_chunk, test_size=0.3, random_state=42
            )
                # batch_size koniecznie mały
            train_ds = Train.create_dataset(X_train, y_train, batch_size=1)
            test_ds  = Train.create_dataset(X_test,  y_test,  batch_size=1)

            print("Datasety gotowe, start fit...")

            model.fit(
                train_ds,
                epochs=epochs_per_chunk,
                verbose=1,
                validation_data=test_ds,
            )

        return model        


    

# %%
model_path = "/home/ciona/projects/RCOLM/data_models/saved/musdb18/"
# show_pred_vs_true(model, X_test, y_test, sample_idx=5, stem_idx=2)
model  = Train.training()
Train.save_model(model, model_path)
FLOOR_DB = -80.0

def load_data_npz(path, n_samples=8):
    # mmap_mode='r' → dane są mapowane z dysku, nie ładowane w całości do RAM
    data = np.load(path, mmap_mode='r')

    # najpierw bierzemy tylko mały kawałek
    X_small = data["X"][:n_samples].astype(np.float32)
    Y_small = data["Y"][:n_samples].astype(np.float32)

    # przycinamy, na wszelki wypadek (już tylko na small)
    X_small = np.clip(X_small, FLOOR_DB, 0.0)
    Y_small = np.clip(Y_small, FLOOR_DB, 0.0)

    # skalowanie dB -> [0,1]
    X_small = (X_small - FLOOR_DB) / -FLOOR_DB  # (-80 -> 0, 0 -> 1)
    Y_small = (Y_small - FLOOR_DB) / -FLOOR_DB

    return X_small, Y_small




def stress_test():
    X_small, Y_small = load_data_npz(
        "/home/ciona/projects/RCOLM/data/raw_data/musdb18/dataset/dataset_5_s.npz"
    )

    print("X_small:", X_small.shape, "Y_small:", Y_small.shape)

    model = MyModels.build_unet((360, 216, 1), num_bins=4)
    model.compile(optimizer='adam', loss='mae')

    history = model.fit(
        X_small, Y_small,
        batch_size=2,     # mniejszy batch, mniej VRAM
        epochs=200,
        verbose=0
    )

    print("Final loss:", history.history["loss"][-1])

    # Jeden przykład do sprawdzenia
    idx = 0
    y_true = Y_small[idx:idx+1]           # (1, 360, 216, 4)
    y_pred = model.predict(X_small[idx:idx+1])

    print("y_true mean:", float(y_true.mean()), "y_pred mean:", float(y_pred.mean()))
    return model, (X_small, Y_small, y_pred)



# %%  
# opis modelu oraz epoch przejsciowy
input_bins = 5
input_shape = (256, 256, 1)

model = MyModels.build_unet(input_shape, num_bins = input_bins)
model.summary()
# 4. Testowy forward pass na losowym tensorze
x_test = tf.random.normal((2, *input_shape))  # batch_size = 2
y_pred = model(x_test)
print("Wejście:", x_test.shape)
print("Wyjście:", y_pred.shape)

# 5. Test treningu – fejkowe dane
y_fake = tf.random.uniform(y_pred.shape)    

model.compile(optimizer='adam', loss='binary_crossentropy')

history = model.fit(
    x_test,
    y_fake,
    epochs=1,
    batch_size=2,
    verbose=1
)


# %%

""" na szybko sklejona funkcja do testów  """
import os
import numpy as np
import librosa
import soundfile as sf
import tensorflow as tf

# Normalizacja taka jak podczas treningu -----------------------
def to_norm(db):
    return (db + 80.0) / 80.0

def to_db(norm):
    return norm * 80.0 - 80.0


# ---------------------------------------------------------------
#  GŁÓWNA FUNKCJA
# ---------------------------------------------------------------

def split_wav_into_stems(model_path, wav_path, out_dir,
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

    os.makedirs(out_dir, exist_ok=True)
    names = ["vocals", "drums", "bass", "other"]

    print("Konwertuję na audio (ISTFT)…")

    stems_audio = []
    for s, name in enumerate(names):
        complex_S = stems_mag[s] * np.exp(1j * phase)
        audio_s = librosa.istft(complex_S, hop_length=1024)
        stems_audio.append(audio_s)

    print("Gotowe.")
    plt.figure(figsize=(14, 5))
    librosa.display.waveshow(y_foreground, sr=sr)
    plt.title('Reconstructed Melody (Vocals)')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Amplitude')
    plt.tight_layout()
    plt.show()
    return stems_audio


