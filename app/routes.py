
from flask import Blueprint, render_template, request
from data.static_data_collectors.data_extractors import Extractors
from data_models.training.CNN_GTZAN.GTZAN_model import MLP_gtzan
from data_models.training.musdb18.musdb_activation import source_activation
from lib.file_save import Gtzan_db
from lib.save_plot import (
    mel_to_disk_and_base64,
    chroma_to_disk_and_base64,
    insert_to_plots,
)
from lib.find_melspec import find_melspec_with_track_id
import librosa
# IMPORT PATHS
from .paths import (
    SAVE_INPUT_PATH_FILE,
    SAVE_MEL_DIR,
    SAVE_CHROMA_DIR,
    GTZAN_DIR,
    MUSDB_DIR,
    GTZAN_KERAS_FILE_DIR,
    MUSDB_MODEL_FILE_PATH,
)
from lib.main_waveform import signal_to_list

bp = Blueprint("main", __name__)


@bp.route("/")
def hello():
    return render_template("home.html")


@bp.route("/about")
def app():
    return render_template("about.html")


@bp.route("/contact")
def contact():
    return render_template("contact.html")


@bp.route("/licenses")
def licenses():
    return render_template("licenses.html")


@bp.route("/analyze", methods=["POST"])
def analyze():
    uploaded = request.files.get("file")
    if uploaded is None:
        return "No file part", 400

    if uploaded.filename == "":
        return "No file selected", 400

    file = request.files["file"]

    filepath, original_name, sha256 = Gtzan_db.save_uploaded_wav(
        file, str(SAVE_INPUT_PATH_FILE)
    )

#  Generating melspectogram, saving it on disk and in db + return to result template
    mel_result = mel_to_disk_and_base64(filepath, str(SAVE_MEL_DIR))
    saved_mel_path = mel_result["save_path"]
    mel_img_src = mel_result["img_src"]

    # Generating chromagram, saving it on disk and in db + return to result template
    chroma_result = chroma_to_disk_and_base64(filepath, str(SAVE_CHROMA_DIR))
    saved_chroma_path = chroma_result["save_path"]
    chroma_img_src = chroma_result["img_src"]

    # audio duration
    duration = librosa.get_duration(filename=filepath)
    duration = "%.2f" % duration

    # basic information about file
    tempo, sr = Extractors.tempo_estimator(filepath)
    #key, scale = Extractors.key_extractor(filepath)

    # machine learning model – ścieżka z paths.py
    gtzan_dir = str(GTZAN_KERAS_FILE_DIR)
    
    # TODO: zmienić na dynamiczny link do bazy
    mel_image_path = saved_mel_path

    # loading model with metadata
    model, meta = MLP_gtzan.load_model(gtzan_dir)
    mime = original_name.split(".")[1]
    # activating genre prediction function
    genre_predictions = MLP_gtzan.predict_from_path(model, meta, mel_image_path)
    # activating split funcktion 
    stems, sr = source_activation(MUSDB_MODEL_FILE_PATH, filepath)  
    stem_names = ["Vocals", "Drums", "Bass", "Other"]
    stems_chart_data = []

    for name, audio in zip(stem_names, stems):
        times, value = signal_to_list(stems, sr)
        stems_chart_data.append(
            {"name": name,
             "times": times,
             "values": value,
             })


    #  przekazujemy mel_img_src do szablonu HTML
    return render_template(
        "results.html",
        original_name=original_name,
        tempo=tempo,
        sr=sr,
        stems=stems,
        # key=key,
        # scale=scale,
        genre_predictions=genre_predictions,
        duration=duration,
        mime=mime,
        mel_img_src=mel_img_src,  # obraz w base64
        mel_image_path=saved_mel_path,  # path obrazu
        chroma_img_src=chroma_img_src,
        saved_chroma_path=saved_chroma_path,
        stems_chart_data=stems_chart_data,
    )

