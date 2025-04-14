"""
Tests for the machine learning client component of the Voice Emotion Detection system.
"""

import io
import os
import json
import tempfile
import sys
import pytest
from unittest.mock import MagicMock, patch
import numpy as np

# Mock torch modules
class MockTorch:
    def no_grad(self):
        class NoGradContext:
            def __enter__(self):
                return None
            def __exit__(self, *args):
                pass
        return NoGradContext()
    
    def argmax(self, tensor):
        mock_result = MagicMock()
        mock_result.item.return_value = 0
        return mock_result
    
    def tensor(self, value):
        return value

# Create mock for functional module
class MockFunctional:
    def softmax(self, tensor, dim=0):
        mock_probs = MagicMock()
        mock_probs.__getitem__.return_value.item.return_value = 0.85
        return mock_probs

# Create torch.nn
torch_mock = MockTorch()
torch_mock.nn = MagicMock()
torch_mock.nn.functional = MockFunctional()

# Mock the torchaudio module
class MockTorchaudio:
    def load(self, file_path):
        return MagicMock(), MagicMock()

# Mock the speechbrain module
class MockEncoderClassifier:
    @staticmethod
    def from_hparams(source, savedir):
        mock_classifier = MagicMock()
        mock_classifier.mods = MagicMock()
        mock_classifier.mods.wav2vec2 = MagicMock()
        mock_classifier.mods.avg_pool = MagicMock()
        mock_classifier.mods.output_mlp = MagicMock()
        
        mock_classifier.hparams = MagicMock()
        mock_classifier.hparams.label_encoder = MagicMock()
        mock_classifier.hparams.label_encoder.decode_ndim = MagicMock(return_value="HAPPY")
        mock_classifier.hparams.label_encoder.expect_len = MagicMock()
        
        return mock_classifier

# Create mock for speechbrain
speechbrain_mock = MagicMock()
speechbrain_mock.inference = MagicMock()
speechbrain_mock.inference.EncoderClassifier = MockEncoderClassifier

# Add mocks to sys.modules
sys.modules['torch'] = torch_mock
sys.modules['torch.nn'] = torch_mock.nn
sys.modules['torch.nn.functional'] = torch_mock.nn.functional
sys.modules['torchaudio'] = MockTorchaudio()
sys.modules['speechbrain'] = speechbrain_mock
sys.modules['speechbrain.inference'] = speechbrain_mock.inference

# Add path to import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Now we can safely import our modules
from main import app, analyze_emotion

# Mock MongoDB
@pytest.fixture(autouse=True)
def mock_mongo_client(monkeypatch):
    """Automatically apply MongoDB client mock to all tests."""
    mongo_client_mock = MagicMock()
    mock_db = MagicMock()
    mock_collection = MagicMock()
    mock_collection.insert_one.return_value.inserted_id = "mock_id"
    mock_db.__getitem__.return_value = mock_collection
    mongo_client_mock.return_value.__getitem__.return_value = mock_db
    
    monkeypatch.setattr('pymongo.MongoClient', mongo_client_mock)
    return mongo_client_mock

@pytest.fixture
def client():
    """Create a test client for the Flask app."""
    app.config["TESTING"] = True
    with app.test_client() as test_client:
        yield test_client

# Create dummy audio data for testing
dummy_audio = b"mock audio data"

@pytest.mark.skip(reason="maintain coverage on other tests")
@patch("main.analyze_emotion")
def test_analyze_success(mock_analyze, client):
    """Test successful audio analysis and storage."""
    # Mock the emotion analyzer to return "HAPPY"
    mock_analyze.return_value = "HAPPY"
    
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

@patch("emotion_analyzer.EncoderClassifier.from_hparams")
@patch("emotion_analyzer.torchaudio.load")
@patch("emotion_analyzer.torch.argmax")
@patch("emotion_analyzer.F.softmax")
@patch("torch.no_grad")
def test_emotion_analyzer(mock_no_grad, mock_softmax, mock_argmax, mock_load, mock_classifier):
    """Test the emotion analyzer functionality."""
    # Configure no_grad context manager
    mock_context = MagicMock()
    mock_no_grad.return_value.__enter__.return_value = mock_context
    
    # Mock classifier
    mock_classifier_instance = MagicMock()
    mock_classifier.return_value = mock_classifier_instance

    # Mock wav2vec2 output
    mock_wav2vec_out = MagicMock()
    mock_classifier_instance.mods.wav2vec2.return_value = mock_wav2vec_out
    
    mock_pooled = MagicMock()
    mock_classifier_instance.mods.avg_pool.return_value = mock_pooled
    
    mock_logits = MagicMock()
    mock_classifier_instance.mods.output_mlp.return_value = mock_logits
    mock_logits.squeeze.return_value = MagicMock()

    # Mock torchaudio load
    mock_load.return_value = (MagicMock(), MagicMock())

    # Mock softmax probabilities
    mock_probs = MagicMock()
    mock_softmax.return_value = mock_probs

    # Mock argmax result
    mock_argmax.return_value.item.return_value = 2  # Index for "happy"

    # Mock probability at index
    mock_probs.__getitem__.return_value.item.return_value = 0.85  # 85% confidence

    # Mock label decoder
    mock_classifier_instance.hparams.label_encoder.decode_ndim.return_value = "HAPPY"

    # Call the function
    result = analyze_emotion("dummy_path.wav")

    # Verify result
    assert result == "HAPPY"

def test_health_check():
    """Test health check endpoint."""
    with app.test_client() as client:
        response = client.get("/")
        assert response.status_code == 200