"""Main execution entry."""

import os
from datetime import datetime
import pymongo
from flask import Flask, request, jsonify

from emotion_analyzer import analyze_emotion

app = Flask(__name__)
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
connection = pymongo.MongoClient(MONGO_URI)
db = connection["emmmm"]

@app.route("/analyze", methods=["POST"])
def main():
    """The main function to run the overall function."""
    if "audio" not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files["audio"]
    audio_data = file.read()

    emotion = analyze_emotion(audio_data)
    emotion["timestamp"] = datetime.utcnow()
    inserted = db.sound_events.insert_one(emotion)
    emotion["_id"] = str(inserted.inserted_id)
    return jsonify({"status": "success", "emotion": emotion})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
