from flask import Flask, request, jsonify
from emotion_analyzer import analyze_emotion
from datetime import datetime, timezone
import os
import pymongo
import tempfile

app = Flask(__name__)
mongo_uri = os.environ.get("MONGO_URI", "mongodb://mongodb:27017/")
client = pymongo.MongoClient(mongo_uri)
db = client["emmmm"]


@app.route("/analyze", methods=["POST"])
def analyze():
    if "audio" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files["audio"]
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    file.save(temp_file.name)
    temp_file.close()

    try:
        emotion = analyze_emotion(temp_file.name)
        print("DEBUG: Returned emotion:", emotion) 
        
        result = {"emotion": emotion, "timestamp": datetime.now(timezone.utc)}
        print("DEBUG: Final result document:", result)  

        inserted = db.sound_result.insert_one(result)
        result["_id"] = str(inserted.inserted_id)
        return jsonify({"status": "success", "result": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(temp_file.name):
            os.unlink(temp_file.name)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=6000)
