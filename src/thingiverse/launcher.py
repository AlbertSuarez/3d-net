from src.config import THINGIVERSE_FLASK_PORT
from src.thingiverse.api import app


def run_app():
    app.run(host='0.0.0.0', port=THINGIVERSE_FLASK_PORT)
