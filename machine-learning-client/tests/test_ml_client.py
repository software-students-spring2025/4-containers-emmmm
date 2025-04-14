import io
import unittest
from unittest.mock import patch, MagicMock
import sys
import os

# Import the Flask app from main.py
from main import app, analyze_emotion

# Dummy audio content for testing
dummy_audio = b"dummy audio content"

class TestMLClient(unittest.TestCase):
    
    def setUp(self):
        """Set up test client."""
        self.app = app
        self.app.config["TESTING"] = True
        self.client = self.app.test_client()
    
    @patch("main.analyze_emotion")
    @patch("main.db.sound_result.insert_one")
    def test_analyze_success(self, mock_insert, mock_analyze):
        """Test successful audio analysis and storage."""
        # Mock the emotion analyzer
        mock_analyze.return_value = "HAPPY"
        
        # Mock MongoDB insert
        mock_insert.return_value = MagicMock(inserted_id="mock_id")
        
        # Create test request
        data = {"audio": (io.BytesIO(dummy_audio), "test_audio.wav", "audio/wav")}
        
        # Make the request
        response = self.client.post("/analyze", data=data, content_type="multipart/form-data")
        
        # Check response
        self.assertEqual(response.status_code, 200)
        json_data = response.get_json()
        self.assertEqual(json_data["status"], "success")
        self.assertEqual(json_data["result"]["emotion"], "HAPPY")
        self.assertEqual(json_data["result"]["_id"], "mock_id")
        
        # Verify emotion analyzer was called
        mock_analyze.assert_called_once()
        
        # Verify MongoDB insert was called
        mock_insert.assert_called_once()
        
    def test_analyze_no_file(self):
        """Test error handling when no file is uploaded."""
        response = self.client.post("/analyze")
        self.assertEqual(response.status_code, 400)
        json_data = response.get_json()
        self.assertIn("error", json_data)
        self.assertIn("No file part", json_data["error"])
        
    @patch("main.analyze_emotion")
    def test_analyze_with_error(self, mock_analyze):
        """Test handling of errors during analysis."""
        # Mock analyzer to raise an exception
        mock_analyze.side_effect = Exception("Analysis failed")
        
        # Create test request
        data = {"audio": (io.BytesIO(dummy_audio), "test_audio.wav", "audio/wav")}
        
        # Make the request
        response = self.client.post("/analyze", data=data, content_type="multipart/form-data")
        
        # Check response
        self.assertEqual(response.status_code, 500)
        json_data = response.get_json()
        self.assertIn("error", json_data)
        self.assertIn("Analysis failed", json_data["error"])
        
class TestEmotionAnalyzer(unittest.TestCase):
    """Test cases for the emotion analyzer component."""

    @patch("emotion_analyzer.EncoderClassifier.from_hparams")
    @patch("emotion_analyzer.torchaudio.load")
    @patch("emotion_analyzer.torch.argmax")
    @patch("emotion_analyzer.F.softmax")
    def test_analyze_emotion(
        self, mock_softmax, mock_argmax, mock_load, mock_classifier
    ):
        """Test the emotion analyzer functionality."""
        # Mock classifier
        mock_classifier_instance = MagicMock()
        mock_classifier.return_value = mock_classifier_instance
        
        # Mock classifier components
        mock_classifier_instance.mods = MagicMock()
        mock_classifier_instance.mods.wav2vec2 = MagicMock(return_value=MagicMock())
        mock_classifier_instance.mods.avg_pool = MagicMock(return_value=MagicMock())
        mock_classifier_instance.mods.output_mlp = MagicMock(return_value=MagicMock())
        
        # Mock label encoder
        mock_classifier_instance.hparams = MagicMock()
        mock_classifier_instance.hparams.label_encoder = MagicMock()
        mock_classifier_instance.hparams.label_encoder.decode_ndim = MagicMock(return_value="HAPPY")
        
        # Mock torchaudio load
        mock_load.return_value = (MagicMock(), MagicMock())
        
        # Mock softmax and argmax results
        probs = MagicMock()
        mock_softmax.return_value = probs
        mock_argmax.return_value = MagicMock(item=MagicMock(return_value=2))
        probs.__getitem__.return_value = MagicMock(item=MagicMock(return_value=0.85))
        
        # Call the function
        result = analyze_emotion("dummy_path.wav")
        
        # Verify result
        self.assertEqual(result, "HAPPY")
        
        # Verify classifier was called
        mock_classifier.assert_called_once()
        mock_classifier_instance.mods.wav2vec2.assert_called_once()
        
        # Verify label decoding
        mock_classifier_instance.hparams.label_encoder.decode_ndim.assert_called_once()

if __name__ == "__main__":
    unittest.main()
