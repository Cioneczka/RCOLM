
import sqlite3

def init_db(db_name="database.db"):
    conn = sqlite3.connect(db_name)
    cur = conn.cursor()

    # 1️⃣ Tabela: pliki audio
    cur.execute("""
    CREATE TABLE IF NOT EXISTS tracks (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        file_path TEXT UNIQUE,
        file_name TEXT,
        genre TEXT,
        duration REAL,
        sample_rate INTEGER,
        key TEXT,
        scale TEXT,
        confidence REAL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)

    # 3️⃣ Tabela: modele ML
    cur.execute("""
    CREATE TABLE IF NOT EXISTS models (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        model_name TEXT,
        model_type TEXT,
        dataset_name TEXT,
        accuracy REAL,
        model_path TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)

    # 4️⃣ Tabela: wyniki predykcji
    cur.execute("""
    CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        track_id INTEGER,
        model_id INTEGER,
        predicted_genre TEXT,
        predicted_key TEXT,
        predicted_scale TEXT,
        confidence REAL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (track_id) REFERENCES tracks(id),
        FOREIGN KEY (model_id) REFERENCES models(id)
    )
    """)

    conn.commit()
    conn.close()
    print(f"✅ Baza danych '{db_name}' została zainicjalizowana.")

if __name__ == "__main__":
    init_db()
