
import numpy as np
import librosa
from spleeter.separator import Separator
import soundfile as sf

# Inicjalizacja modelu (2 źródła: wokal + tło)
separator = Separator('spleeter:2stems')

# Wczytaj WAV jako tablicę numpy
y, sr = librosa.load('/home/ciona/projects/RCOLM/data/raw_data/GTZAN/genres_original/blues/blues.00003.wav',
                     sr=44100, mono=False)

# Upewnij się, że sygnał ma wymiar [samples, channels]
if y.ndim == 1:
    y = np.expand_dims(y, axis=1)

# Separacja (zwraca słownik z waveformami)
prediction = separator.separate(y)

# Wyniki: wokal i akompaniament
vocals = prediction['vocals']
backing = prediction['accompaniment']

print(vocals.shape, backing.shape)

# Opcjonalnie zapisz wynik
sf.write('vocals.wav', vocals, sr)
sf.write('backing.wav', backing, sr)

