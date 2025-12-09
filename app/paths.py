from pathlib import Path 

BASE_PATH = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_PATH / "data"
INPUT_DIR_PATH = DATA_PATH / "input"

#SAVE DIRS
SAVE_INPUT_PATH_FILE = INPUT_DIR_PATH / "wavs"
SAVE_MEL_DIR = INPUT_DIR_PATH / "plots" / "mel"
SAVE_CHROMA_DIR = INPUT_DIR_PATH / "plots" / "chroma"

#MODEL DIRS
GTZAN_DIR = BASE_PATH / "data_models" / "saved" / "gtzan_v1"
MUSDB_DIR = BASE_PATH / "data_models" / "saved" / "musdb18"

#MODEL FILES
GTZAN_KERAS_FILE_DIR = GTZAN_DIR / "model.keras"
GTZAN_MODEL_DIR = BASE_PATH / "data_models" / "Mymodels" 

MUSDB_MODEL_FILE_PATH = MUSDB_DIR / "unet_model2025-11-25 15:46:24.041779.h5"
