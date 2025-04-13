"""Webapp Functionality Module.

This module defines a Flask web application that handles audio file uploads
and renders the main index page.
"""

import requests
from flask import Flask, render_template, request, jsonify
from db import db, ml_client_host

app = Flask(__name__)


@app.route("/")
def home():
    """Render the main index page."""
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    """Upload an audio file and store its processing status in the database.

    Returns:
        A JSON response with an error message if no audio file is provided,
        or a success message including the uploaded file's name.
    """
    if "audio" not in request.files:
        return {"error": "No audio file provided"}, 400

    audio_file = request.files["audio"]
    filename = audio_file.filename
    file_record = {"audio_file_name": filename, "result": "processing"}
    db.file.insert_one(file_record)
    
    response = requests.post(
        f"{ml_client_host}/analyze", files={"audio": file_record}, timeout=100
    )
    return jsonify(response.json()), response.status_code



if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)