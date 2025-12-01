# Lekka baza z Pythonem
FROM python:3.11-slim

# Zależności systemowe potrzebne m.in. do audio (librosa, soundfile, ffmpeg itd.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ffmpeg \
    libsndfile1 \
 && rm -rf /var/lib/apt/lists/*

# Ustaw katalog roboczy
WORKDIR /app

# Najpierw requirements, żeby cache się lepiej wykorzystywał
COPY requirements.txt .

# Instalacja zależności Pythona
RUN pip install --no-cache-dir -r requirements.txt

# Skopiowanie całej reszty kodu
COPY . .

# Zmienna na port (Azure Container Apps go używają)
ENV PORT=8000

# Domyślne polecenie startowe:
# zakładam, że:
#  - główny plik to app.py
#  - obiekt Flask nazywa się "app" (app = Flask(__name__))
CMD ["gunicorn", "-b", "0.0.0.0:8000", "app:app"]

