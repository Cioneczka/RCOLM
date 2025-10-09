
import os
import hashlib
from datetime import datetime
from werkzeug.utils import secure_filename
import sqlite3
def save_uploaded_wav(file, upload_dir):
    """
    Zapisuje przesłany plik WAV na dysku.
    :param file: obiekt pliku 
    :param upload_dir: katalog docelowy (domyślnie data/uploads)
    :return: pełna ścieżka do zapisanego pliku
    """
    # check if dir exists 
    os.makedirs(upload_dir, exist_ok=True)

    original_file_name = secure_filename(file.filename)
    #hashed name generating
    content = file.read() if hasattr(file, "read") else file
    file.seek(0) if hasattr(file, "seek") else None
    file_hash = hashlib.sha256(content).hexdigest()[:16]
    safe_name = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{file_hash}.wav"
    save_path = os.path.join(upload_dir, safe_name)
    # for deduplication 
    sha256 = file_hash
    

    # File save sive 
    with open(save_path, "wb") as f:
        f.write(content)

    print(f"✅ Zapisano plik: {save_path}")
    return save_path, original_file_name, sha256

#Insert inputed track to track table


def insert_to_tracks(original_name, storage_path, mime, sr, duration_sec, sha256, mode,
                     db_name="db/database.db"):
    conn = sqlite3.connect(db_name, timeout=10)
    cur = conn.cursor()

    # tryb WAL zwiększa odporność na blokady
    cur.execute("PRAGMA journal_mode=WAL;")
    cur.execute("PRAGMA synchronous=NORMAL;")

    try:
        cur.execute("""
            INSERT INTO tracks (
                original_name,
                storage_path,
                mime,
                sr,
                duration_sec,
                sha256,
                mode
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (original_name, storage_path, mime, sr, duration_sec, sha256, mode))

        conn.commit()
    except sqlite3.OperationalError as e:
        print("❌ Błąd bazy danych:", e)
        raise
    finally:
        conn.close()


