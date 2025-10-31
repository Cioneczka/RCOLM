from flask import Blueprint
from flask import render_template, request, Flask
from data.static_data_collectors.data_extractors import Extractors
from data_models.training.CNN_GTZAN.GTZAN_model import MLP_gtzan
from lib.file_save import Gtzan_db
from lib.save_plot import mel_to_disk_and_base64, chroma_to_disk_and_base64, insert_to_plots
from lib.find_melspec import find_melspec_with_track_id
import librosa




bp = Blueprint("main", __name__)

@bp.route("/")
def hello():
    return render_template("home.html")

@bp.route("/app")
def app():
    return render_template("app.html")

@bp.route("/contact")
def contact():
    return render_template("contact.html")



@bp.route("/analyze", methods=["POST"])
def analyze():
    if "file" not in request.files:
        return "No file part", 400

    file = request.files["file"]
    if file.filename == "":
        return "No file selected", 400

    save_input_file_path = "/home/ciona/projects/RCOLM/data/input/wavs/"
    file = request.files["file"]
    filepath, original_name, sha256 = Gtzan_db.save_uploaded_wav(file, save_input_file_path)

    #  Generating melspectogram, saving it on disk and in db + return to result template
    save_mel_dir = "/home/ciona/projects/RCOLM/data/input/plots/mel"
    mel_result = mel_to_disk_and_base64(filepath, save_mel_dir)
    saved_mel_path = mel_result["save_path"]
    mel_img_src = mel_result["img_src"]

    #Generating chromagram, saving it on disk and in db + return to result template
    save_chroma_dir = "/home/ciona/projects/RCOLM/data/input/plots/chroma"
    chroma_result = chroma_to_disk_and_base64(filepath, save_chroma_dir)
    saved_chroma_path = chroma_result["save_path"]
    chroma_img_src = chroma_result["img_src"]
    
    # audio duration
    duration = librosa.get_duration(filename=filepath)
    duration = "%.2f" % duration

    # basic information about file 
    tempo, sr = Extractors.tempo_estimator(filepath)
    key, scale = Extractors.key_extractor(filepath)

    # machine learning model
    save_dir = "/home/ciona/projects/RCOLM/data_models/saved/gtzan_v1/"

    # TODO: zmienić na dynamiczny link do bazy
    mel_image_path = saved_mel_path

    # loading model with metadata 
    model, meta = MLP_gtzan.load_model(save_dir)
    mime = original_name.split(".")[1]
    
    # activating prediction function
    genre_predictions = MLP_gtzan.predict_from_path(model, meta, mel_image_path)
    track_id = Gtzan_db.insert_to_tracks(original_name, filepath, mime, sr, duration, sha256, scale)

    insert_to_plots(track_id, saved_mel_path, "mel")
    insert_to_plots(track_id, saved_chroma_path, "chroma")
    
    #  przekazujemy mel_img_src do szablonu HTML
    return render_template(
        "results.html",
        original_name=original_name,
        tempo=tempo,
        sr=sr,
        key=key,
        scale=scale,
        genre_predictions=genre_predictions,
        duration=duration,
        mime=mime,
        mel_img_src=mel_img_src,      #obraz w base64    
        mel_image_path=saved_mel_path,   #path obrazu 
        chroma_img_src=chroma_img_src,
        saved_chroma_path=saved_chroma_path
    )

