# %%
import numpy as np 
import tensorflow as tf
import keras 
from sklearn.model_selection import train_test_split
import librosa

import matplotlib.pyplot as plt
import sys
sys.path.append("/home/ciona/projects/RCOLM/data_models/MyModels")  
from mymodels import MyModels


# %% [md]
##  Main funkction

# %%

class MUSDB_augmentation:
    def change_volume(x, min_gain, max_gain):
        db = np.random.uniform(min_gain, max_gain)
        factor  = 10 ** (db/20)
        return x * factor
    
    def noise(x, noise_level):
        noise = np.random.randn(*x.shape)*nosie_level
        return x + noise

    
    def time_stretch(x, rate=1.1):
        return librosa.effects.time_stretch(x, rate)


    def bandpass(x, low, high, sr=44100):
        b,a = butter(4, [low/(sr/2), high/(sr/2)], btype='band')
        return lfilter(b, a, x)

    def train():
        print('dupa')

# %%   

class Train:
    @staticmethod 
    def create_dataset(X, y, batch_size=2):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)

        print("Creating dataset from X.shape =", X.shape, " y.shape =", y.shape, flush=True)

        ds = tf.data.Dataset.from_tensor_slices((X, y))
        ds = ds.shuffle(buffer_size=min(64, len(X)))
        ds = ds.batch(batch_size)
        ds = ds.prefetch(tf.data.AUTOTUNE)
        return ds


    @staticmethod
    def training():
        DATA_PATH = "/home/ciona/projects/RCOLM/data_models/training/musdb18/dataset/dataset.npz"
        data = np.load(DATA_PATH)
        X = data["X"]
        y = data["Y"]
        #normalization
        
        print("X min:", np.min(X))
        print("X max:", np.max(X))
        print("X mean:", np.mean(X))
        print("Is NaN:", np.isnan(X).any())
        print("Is inf:", np.isinf(X).any())

        print("\ny min:", np.min(y))
        print("y max:", np.max(y))
        print("y mean:", np.mean(y))
        print("Is NaN:", np.isnan(y).any())
        print("Is inf:", np.isinf(y).any())


        print("X shape:", X.shape)
        print("y shape:", y.shape)
        
        X = (X+80.0)/80.0
        y = (y+80)/80   
# augmentation


# 🔹 10% danych
        ratio = 0.1
        n = X.shape[0]
        n_subset = int(n * ratio)

        idx = np.random.permutation(n)[:n_subset]  # losowe indeksy
        X = X[idx]
        y = y[idx]

        print("Dataset fragment:")
        print("X shape:", X.shape)
        print("y shape:", y.shape)        

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # Wejściowy kształt (tu U-Net na pełnym rozmiarze 360x216)
        H, W, C_in = X.shape[1], X.shape[2], X.shape[3]
        C_out = y.shape[3]   # 4

        input_shape = (H, W, C_in)   # (360, 216, 1)
        input_bins = C_out           # 4

        # Uważaj: to są duże macierze – batch_size koniecznie mały
        train_ds = Train.create_dataset(X_train, y_train, batch_size=1)
        test_ds  = Train.create_dataset(X_test,  y_test,  batch_size=1)

        print("Dataset ready, compiling model...")

        model = MyModels.build_unet(input_shape, num_bins=input_bins)
        model.compile(optimizer='adam', loss='mse', metrics = 
        [tf.keras.metrics.MeanSquaredError(name="mse"),
        tf.keras.metrics.MeanAbsoluteError(name="mae"),])

        print("Model compiled, starting fit...")

        history = model.fit(
            train_ds,
            epochs=10,
            verbose=1,
            validation_data=test_ds,
        )





        return model, history


# %%

Train.training()

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
