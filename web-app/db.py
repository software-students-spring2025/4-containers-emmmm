"""
db.py - Database configuration for the Flask web app.
"""

import os
import pymongo
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Use MONGO_URI; fallback to localhost if not defined.
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
connection = pymongo.MongoClient(MONGO_URI)
ml_client_host = os.getenv("ML_CLIENT_HOST", "http://localhost:6000")


# Access database
db = connection["emmmm"]
