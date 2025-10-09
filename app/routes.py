from flask import Blueprint
from flask import render_template, request, Flask
from data.static_data_collectors.data_extractors import Extractors
from data_models.training.CNN_GTZAN.GTZAN_model import MLP_gtzan
from lib.file_save import save_uploaded_wav, insert_to_tracks





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

    save_input_file_path = "/home/ciona/projects/RCOLM/data/input/"
    file = request.files["file"]
    filepath, original_name, sha256 = save_uploaded_wav(file, save_input_file_path)

    #basic information about file 
    tempo, sr = Extractors.tempo_estimator(filepath)
    key,scale = Extractors.key_extractor(filepath)


    #machine learning methods
    gtzan_path = "/home/ciona/projects/RCOLM/data/converted_data/GTZAN/"
    save_dir =  "/home/ciona/projects/RCOLM/data_models/saved/gtzan_v1"

    #na pozniej do wywalenia -- trzeba to zmienic na dynamiczny link do bazy !!!
    image_path = "/home/ciona/projects/RCOLM/tests/png/blue_train.png"

    #loading model with metadata 
    model, meta = MLP_gtzan.load_model(save_dir)
    mime = original_name.split(".")[1]

    #activating prediction function
    genre_predictions = MLP_gtzan.predict_from_path(model, meta, image_path)
    insert_to_tracks(original_name, filepath, mime,  sr, "future_duration", sha256, scale)



    return render_template("results.html", 
                           original_name=original_name, 
                           tempo=tempo, 
                           sr=sr, 
                           key=key, 
                           scale=scale, 
                           genre_predictions=genre_predictions
                           )


