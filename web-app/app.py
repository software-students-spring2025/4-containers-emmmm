"""Webapp Functionality"""

from flask import Flask, render_template as rt, request
import requests
from db import db

app = Flask(__name__)


@app.route("/")
def home():

    """ index route"""
    return rt("index.html")

@app.route("/upload", methods=["POST"])
def upload():
    if "audio" not in request.files:
        return {"error": "No audio file provided"}, 400
    audio_file = request.files["audio"]
    file_record = {"result": "processing"}
    # Consider saving the file or storing its metadata if needed
    db.file.insert_one(file_record)
    return {"message": "Audio uploaded successfully"}, 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
