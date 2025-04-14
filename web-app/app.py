import os
import json
import requests
from flask import Flask, render_template, request, jsonify
import pymongo
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Get environment variables with defaults
mongo_uri = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
ml_client_host = os.getenv("ML_CLIENT_HOST", "http://ml-client:6000")

# Connect to MongoDB
try:
    client = pymongo.MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
    # Verify connection works
    client.server_info()
    db = client["emmmm"]
    print("Connected to MongoDB successfully")
except pymongo.errors.ServerSelectionTimeoutError as err:
    print(f"MongoDB connection error: {err}")
    db = None

@app.route("/")
def home():
    """Render the main index page."""
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():
    """
    Handle audio file uploads, send to ML client for processing,
    and return the analysis results.
    """
    # Check if audio file exists in request
    if "audio" not in request.files:
        return jsonify({"error": "No audio file uploaded"}), 400
    audio = request.files.get("audio")
    # Check if filename is empty
    if audio.filename == "":
        return jsonify({"error": "No selected file"}), 400
    try:
        # Send file to ML client for analysis
        response = requests.post(
            f"{ml_client_host}/analyze", 
            files={"audio": (audio.filename, audio.stream, audio.content_type)},
            timeout=60  # Increased timeout for ML processing
        )
        # Check if response is valid
        try:
            result = response.json()
            return jsonify(result), response.status_code
        except json.JSONDecodeError:
            return jsonify({
                "error": "ML Client did not return valid JSON",
                "raw_response": response.text
            }), 500
            
    except requests.RequestException as error:
        return jsonify({
            "error": f"Failed to connect to ML client: {str(error)}"
        }), 500
@app.route("/health", methods=["GET"])
def health_check():
    """
    Simple health check endpoint to verify the service is running.
    """
    status = {
        "status": "ok",
        "service": "web",
        "mongodb_connected": db is not None,
    }
    # Check ML client connection
    try:
        ml_response = requests.get(f"{ml_client_host}/health", timeout=5)
        status["ml_client_connected"] = ml_response.status_code == 200
    except:
        status["ml_client_connected"] = False
    return jsonify(status)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
    
