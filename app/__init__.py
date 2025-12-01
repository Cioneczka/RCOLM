from flask import Flask
from app.routes import bp


def create_app():
    app = Flask(
        __name__,
        template_folder="app/templates",
        static_folder="app/static",
    )
    app.register_blueprint(bp)
    app.config["TEMPLATES_AUTO_RELOAD"] = True
    return app


# To jest obiekt, którego szuka gunicorn: app:app
app = create_app()

