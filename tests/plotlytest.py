
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# Wczytanie sygnału audio
y, sr = librosa.load(librosa.ex('trumpet'))

# Parametry
n_fft = 2048
hop_length = 1024

# ===== Obliczenia =====

# 1. STFT (liniowy)
S = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
S_mag = np.abs(S)
S_db = librosa.amplitude_to_db(S_mag, ref=np.max)

# 2. Spektrogram melowy
S_mel = librosa.feature.melspectrogram(
    y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=128
)
S_mel_db = librosa.power_to_db(S_mel, ref=np.max)

# 3. Chromagram
chroma = librosa.feature.chroma_stft(
    y=y, sr=sr, hop_length=hop_length
)

# 4. Constant-Q Transform (CQT)
CQT = librosa.cqt(y, sr=sr, hop_length=hop_length)
CQT_db = librosa.amplitude_to_db(np.abs(CQT), ref=np.max)

# ===== Wizualizacja =====

fig, axs = plt.subplots(5, 1, figsize=(12, 14))

# STFT liniowy (amplituda)
img1 = librosa.display.specshow(
    S_mag, sr=sr, hop_length=hop_length,
    x_axis='time', y_axis='linear', ax=axs[0]
)
axs[0].set(title='Spektrogram liniowy (amplituda)')
fig.colorbar(img1, ax=axs[0])

# STFT w skali dB
img2 = librosa.display.specshow(
    S_db, sr=sr, hop_length=hop_length,
    x_axis='time', y_axis='linear', ax=axs[1]
)
axs[1].set(title='Spektrogram liniowy w skali dB')
fig.colorbar(img2, ax=axs[1], format='%+2.0f dB')

# Spektrogram melowy
img3 = librosa.display.specshow(
    S_mel_db, sr=sr, hop_length=hop_length,
    x_axis='time', y_axis='mel', ax=axs[2]
)
axs[2].set(title='Spektrogram melowy')
fig.colorbar(img3, ax=axs[2], format='%+2.0f dB')

# Chromagram
img4 = librosa.display.specshow(
    chroma, sr=sr, hop_length=hop_length,
    x_axis='time', y_axis='chroma', ax=axs[3]
)
axs[3].set(title='Chromagram')
fig.colorbar(img4, ax=axs[3])

# CQT
img5 = librosa.display.specshow(
    CQT_db, sr=sr, hop_length=hop_length,
    x_axis='time', y_axis='cqt_note', ax=axs[4]
)
axs[4].set(title='Spektrogram Constant-Q (CQT)')
fig.colorbar(img5, ax=axs[4], format='%+2.0f dB')

plt.tight_layout()
plt.show()

