import sqlite3




def find_melspec_with_track_id(track_id, db_name="db/database.db"):
    conn = sqlite3.connect(db_name, timeout=10)
    cur = conn.cursor()

    cur.execute("PRAGMA journal_mode=WAL;")
    cur.execute("PRAGMA synchronous=NORMAL;")

    try:
        cur.execute("""
            SELECT plot_path 
            FROM plots 
            WHERE track_id = ? 
              AND plot_type = 'mel'
        """, (track_id,))

        row = cur.fetchone()
        if row:
            plot_path = row[0]
            return plot_path
        else:
            print(f"⚠️ Brak wpisu dla track_id={track_id} i plot_type='mel'")
            return None

    except sqlite3.Error as e:
        print(f"❌ Błąd bazy danych: {e}")
        return None

    finally:
        conn.close()
