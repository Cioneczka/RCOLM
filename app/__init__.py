from flask import Flask
from app.routes import bp


def create_app():
    app = Flask(__name__)
    app.register_blueprint(bp)
    app.config["TEMPLATES_AUTO_RELOAD"] = True
    return app


# To jest obiekt, którego szuka gunicorn przy "app:app"
app = create_app()

