
-- ===============================================
--  RCOLM — Database Initialization Script (SQLite)
-- ===============================================

CREATE TABLE IF NOT EXISTS tracks (
  id            INTEGER PRIMARY KEY AUTOINCREMENT,
  original_name TEXT,
  storage_path  TEXT NOT NULL,
  mime          TEXT,
  sr            INTEGER,
  duration_sec  REAL,
  sha256        TEXT UNIQUE,
  mode          TEXT,
  created_at    TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS models (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  model_name TEXT,
  model_type TEXT,
  dataset_name TEXT,
  accuracy REAL,
  model_path TEXT,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS predictions_gtzan_1 (
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
);

CREATE TABLE IF NOT EXISTS plots (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  track_id INTEGER,
  plot_path TEXT,
  plot_type TEXT,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY (track_id) REFERENCES tracks(id)
);

CREATE TABLE IF NOT EXISTS predictions_musdb18 (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  track_id INTEGER,
  model_id INTEGER,
  target_name TEXT,
  file_path TEXT,
  sdr REAL,
  sir REAL,
  sar REAL,
  confidence REAL,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY (track_id) REFERENCES tracks(id),
  FOREIGN KEY (model_id) REFERENCES models(id)
);

CREATE TABLE IF NOT EXISTS musbd_odict_keys (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  track_id INTEGER,
  file_path TEXT,
  file_name TEXT,
  odict_keys_json TEXT,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY (track_id) REFERENCES tracks(id)
);
 
