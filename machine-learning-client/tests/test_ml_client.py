"""
tests for the machine learning client component.
"""

import os
import json
import tempfile
from unittest.mock import patch, MagicMock
import io
import pytest

# Import the Flask app from main.py
from main import app


@pytest.fixture
def client():
    """Create a test client for the Flask app."""
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


@patch("main.analyze_emotion")
def test_analyze_success(mock_analyze, client):
    """Test successful audio analysis and storage."""
    # Mock the emotion analyzer to return "HAPPY"
    mock_analyze.return_value = "HAPPY"
    
    # Create dummy audio data
    dummy_audio = b"mock audio data"
    
    # Create a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp:
        temp.write(dummy_audio)
        temp_name = temp.name
    
    try:
        # Prepare the file for upload
        with open(temp_name, "rb") as f:
            data = {"audio": (io.BytesIO(f.read()), "test_audio.wav", "audio/wav")}
            
            # Make the request
            response = client.post("/analyze", data=data, content_type="multipart/form-data")
        
        # Check response
        assert response.status_code == 200
        result = json.loads(response.data)
        assert result["status"] == "success"
        assert result["result"]["emotion"] == "HAPPY"
        assert "_id" in result["result"]
        
        # Verify emotion analyzer was called
        mock_analyze.assert_called_once()
        
    finally:
        # Clean up the temporary file
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def test_analyze_no_file(client):
    """Test error handling when no file is uploaded."""
    response = client.post("/analyze")
    assert response.status_code == 400
    result = json.loads(response.data)
    assert "error" in result
    assert "No file part" in result["error"]


@patch("main.analyze_emotion")
def test_analyze_with_error(mock_analyze, client):
    """Test handling of errors during analysis."""
    # Mock analyzer to raise an exception
    mock_analyze.side_effect = Exception("Analysis failed")
    
    # Create dummy audio data
    dummy_audio = b"mock audio data"
    
    # Create a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp:
        temp.write(dummy_audio)
        temp_name = temp.name
    
    try:
        # Prepare the file for upload
        with open(temp_name, "rb") as f:
            data = {"audio": (io.BytesIO(f.read()), "test_audio.wav", "audio/wav")}
            
            # Make the request
            response = client.post("/analyze", data=data, content_type="multipart/form-data")
        
        # Check response
        assert response.status_code == 500
        result = json.loads(response.data)
        assert "error" in result
        assert "Analysis failed" in result["error"]
        
    finally:
        # Clean up the temporary file
        if os.path.exists(temp_name):
            os.unlink(temp_name)
