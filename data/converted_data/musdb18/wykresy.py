
import numpy as np
import matplotlib.pyplot as plt

# Wczytanie datasetu
data = np.load("dataset.npz")  # np. "dataset.npz"
X = data["X"]  # tablica X
Y = data["Y"]  # tablica Y

# Sprawdzenie kształtów
print("X.shape =", X.shape)
print("Y.shape =", Y.shape)

# Teraz możesz użyć tej samej wizualizacji co wcześniej
idx = 5  # numer próbki
stem_names = ["bass", "drums", "other", "vocals"]

fig, axes = plt.subplots(1, 1 + Y.shape[-1], figsize=(18, 4))

# X – miks
axes[0].imshow(X[idx, :, :, 0], aspect="auto", origin="lower")
axes[0].set_title("Input X (mixture)")
axes[0].set_xlabel("Time")
axes[0].set_ylabel("Frequency")

# Y – stemsy
for i in range(Y.shape[-1]):
    axes[i+1].imshow(Y[idx, :, :, i], aspect="auto", origin="lower")
    axes[i+1].set_title(f"Y – {stem_names[i]}")
    axes[i+1].set_xlabel("Time")

plt.tight_layout()
plt.show()

