# Bazowy Python (slim + system deps later)
FROM python:3.11-slim

# Instalacja systemowych zależności dla:
# - matplotlib (libfreetype, libpng, fontconfig)
# - audio (ffmpeg, libsndfile)
# - kompilacji pip (build-essential)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ffmpeg \
    libsndfile1 \
    libfreetype6-dev \
    libpng-dev \
    pkg-config \
    libfontconfig1 \
    && rm -rf /var/lib/apt/lists/*

# Katalog roboczy
WORKDIR /app

# Requirements najpierw
COPY requirements.txt .

# Instalacja zależności Pythona
RUN pip install --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r requirements.txt

# Skopiowanie całego projektu
COPY . .

# Port dla Azure
ENV PORT=8000

# Uruchamianie przez gunicorn
CMD ["gunicorn", "-b", "0.0.0.0:8000", "app:app"]

