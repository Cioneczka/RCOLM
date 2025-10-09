
import os
import io
import base64
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt

def mel_to_disk_and_base64(
    audio_path: str,
    save_dir: str,
    sr: int = 22050,
    duration: float = 30.0,
    start_frac: float = 0.1,
    n_mels: int = 128,
    n_fft: int = 2048,
):
    """
    Zwraca dict: {"save_path": str, "img_src": "data:image/png;base64,..."}.
    Jednocześnie zapisuje PNG na dysk i zwraca obraz w base64 do wstawienia na stronę.
    """
    os.makedirs(save_dir, exist_ok=True)

    # fragment audio (np. od 10% długości, przez 'duration' sekund)
    total_duration = librosa.get_duration(filename=audio_path)
    offset = max(0.0, total_duration * start_frac)

    y, sr = librosa.load(audio_path, sr=sr, duration=duration, offset=offset, mono=True)
    spec = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, n_mels=n_mels)
    mel_db = librosa.power_to_db(spec, ref=np.max)

    # nazwa pliku docelowego
    base = os.path.splitext(os.path.basename(audio_path))[0]
    save_path = os.path.join(save_dir, f"{base}_mel.png")

    # rysuj jeden raz, potem zapisz i wyślij jako base64
    fig, ax = plt.subplots(figsize=(8, 3))
    librosa.display.specshow(mel_db, sr=sr, x_axis="time", y_axis="mel", ax=ax)
    ax.set(title="Mel-Spectrogram")
    plt.tight_layout()

    # zapis na dysk
    fig.savefig(save_path, format="png", bbox_inches="tight")

    # zapis do pamięci -> base64 (dla <img src="...">)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode("utf-8")
    buf.close()

    img_src = f"data:image/png;base64,{img_base64}"
    return {"save_path": save_path, "img_src": img_src}




def chroma_to_disk_and_base64(
    audio_path: str,
    save_dir: str,
    sr: int = 22050,
    duration: float = 30.0,
    start_frac: float = 0.1,
    method: str = "cqt",          # "cqt" lub "stft"
    n_chroma: int = 12,
    hop_length: int = 512,
    n_fft: int = 2048,            # używane przy STFT
):
    """
    Generuje chromagram, zapisuje PNG na dysk i zwraca base64 gotowe do <img src="...">.

    Zwraca dict:
      {
        "save_path": "/sciezka/do/pliku.png",
        "img_src": "data:image/png;base64,..."
      }
    """
    os.makedirs(save_dir, exist_ok=True)

    # fragment audio (od start_frac * długości)
    total_duration = librosa.get_duration(filename=audio_path)
    offset = max(0.0, total_duration * start_frac)
    y, sr = librosa.load(audio_path, sr=sr, duration=duration, offset=offset, mono=True)

    # --- chroma ---
    if method.lower() == "stft":
        chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_chroma=n_chroma, n_fft=n_fft, hop_length=hop_length)
    else:
        # domyślna metoda CQT
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr, n_chroma=n_chroma, hop_length=hop_length)

    # nazwa pliku
    base = os.path.splitext(os.path.basename(audio_path))[0]
    suffix = f"chroma_{method.lower()}"
    save_path = os.path.join(save_dir, f"{base}_{suffix}.png")

    # rysowanie
    fig, ax = plt.subplots(figsize=(8, 3))
    librosa.display.specshow(chroma, x_axis="time", y_axis="chroma", sr=sr, hop_length=hop_length, ax=ax)
    ax.set(title=f"Chromagram ({method.upper()})")
    plt.tight_layout()

    # zapis na dysk
    fig.savefig(save_path, format="png", bbox_inches="tight")

    # wersja base64
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode("utf-8")
    buf.close()

    img_src = f"data:image/png;base64,{img_base64}"
    return {"save_path": save_path, "img_src": img_src}
