from flask import Blueprint
from flask import render_template

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



