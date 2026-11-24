from __future__ import annotations

from app import create_app
from config import APP_DEBUG, APP_HOST, APP_PORT

app = create_app()
server = app.server

if __name__ == "__main__":
    app.run(debug=APP_DEBUG, host=APP_HOST, port=APP_PORT)
