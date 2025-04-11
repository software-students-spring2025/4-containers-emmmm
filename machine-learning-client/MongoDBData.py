from pymongo import MongoClient
from datetime import datetime

client = MongoClient("mongodb://mongo:27017")  
db = client["ml_client"]
collection = db["audio_analysis"]

def save_metadata(audio_file, result):
    document = {
        "filename": audio_file,
        "result": result,
        "timestamp": datetime.utcnow()
    }
    collection.insert_one(document)
    print(f"Saved metadata to MongoDB: {document}")
