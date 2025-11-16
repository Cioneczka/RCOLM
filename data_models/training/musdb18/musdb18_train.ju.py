# %%
import numpy as np 
import tensorflow as tf
import keras 
from sklearn.model_selection import train_test_split
import librosa

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

    def training():
        DATA_PATH = "/home/ciona/projects/RCOLM/data_models/training/musdb18/dataset/dataset.npz"
        min_gain = -6
        max_gain = 6
        x=0
        noise_level = 0.01
        
        data = np.load(DATA_PATH)
        X = data["X"]
        y = data["Y"]
        
        X_train, y_train, X_test, y_test = train_test_split(X, y, test_size = 0.3)
        
        for x, y in zip(X, y):
            print('s')


        

# %%


