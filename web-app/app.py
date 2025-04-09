"""Webapp Functionality"""

from flask import Flask, render_template as rt, request
import requests
from db import db

app = Flask(__name__)


@app.route("/")
def home():

    """ index route"""
    return rt("index.html")

@app.route("/upload",methods=["POST"])
def upload():
    """ Send audio file to database"""
    if request.method == "POST":
        audio = request.form.get("formData")
        file = {"audio": audio, "result": "processing"}
        db.file.insert_one(file)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
<<<<<<< HEAD:web-app/App.py
    
=======
>>>>>>> 1951c5fab92dc74e7b00b82f1215fb432eb8c922:web-app/app.py
