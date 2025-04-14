"""
Tests for the machine learning client component of the Voice Emotion Detection system.
"""

import unittest
import os
import json
import tempfile
from unittest.mock import patch, MagicMock
import io
import pymongo
import pytest
from datetime import datetime

# Import the Flask app from main.py
from main import app, analyze_emotion

class TestMLClient(unittest.TestCase):
    """Test cases for the ML client application."""

    def setUp(self):
        """Set up test client and other test variables."""
        self.app = app
        self.app.config['TESTING'] = True
        self.client = self.app.test_client()

    @patch('main.analyze_emotion')
    @patch('main.pymongo.MongoClient')
    def test_analyze_success(self, mock_mongo, mock_analyze):
        """Test successful audio analysis and storage."""
        # Mock the emotion analyzer
        mock_analyze.return_value = "HAPPY"
        
        # Mock MongoDB client
        mock_db = MagicMock()
        mock_collection = MagicMock()
        mock_mongo.return_value.__getitem__.return_value = mock_db
        mock_db.__getitem__.return_value = mock_collection
        mock_collection.insert_one.return_value.inserted_id = "mock_id"
        
        # Create test audio file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp:
            temp.write(b"mock audio data")
            temp_name = temp.name
        
        try:
            # Prepare the file for upload
            with open(temp_name, 'rb') as f:
                data = {'audio': (io.BytesIO(f.read()), 'test_audio.wav', 'audio/wav')}
                
                # Make the request
                response = self.client.post(
                    '/analyze',
                    data=data,
                    content_type='multipart/form-data'
                )
            
            # Check response
            assert response.status_code == 200
            result = json.loads(response.data)
            assert result['status'] == 'success'
            assert result['result']['emotion'] == 'HAPPY'
            assert '_id' in result['result']
            
            # Verify emotion analyzer was called
            mock_analyze.assert_called_once()
            
            # Verify MongoDB insert was called
            mock_collection.insert_one.assert_called_once()
            args, _ = mock_collection.insert_one.call_args
            assert args[0]['emotion'] == 'HAPPY'
            
        finally:
            # Clean up temporary file
            if os.path.exists(temp_name):
                os.unlink(temp_name)

    def test_analyze_no_file(self):
        """Test error handling when no file is uploaded."""
        response = self.client.post('/analyze')
        assert response.status_code == 400
        result = json.loads(response.data)
        assert 'error' in result
        assert 'No file part' in result['error']

    @patch('main.analyze_emotion')
    def test_analyze_with_error(self, mock_analyze):
        """Test handling of errors during analysis."""
        # Mock analyzer to raise an exception
        mock_analyze.side_effect = Exception("Analysis failed")
        
        # Create test audio file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp:
            temp.write(b"mock audio data")
            temp_name = temp.name
        
        try:
            # Prepare the file for upload
            with open(temp_name, 'rb') as f:
                data = {'audio': (io.BytesIO(f.read()), 'test_audio.wav', 'audio/wav')}
                
                # Make the request
                response = self.client.post(
                    '/analyze',
                    data=data,
                    content_type='multipart/form-data'
                )
            
            # Check response
            assert response.status_code == 500
            result = json.loads(response.data)
            assert 'error' in result
            assert 'Analysis failed' in result['error']
            
        finally:
            # Clean up temporary file
            if os.path.exists(temp_name):
                os.unlink(temp_name)

class TestEmotionAnalyzer(unittest.TestCase):
    """Test cases for the emotion analyzer component."""
    
    @patch('emotion_analyzer.EncoderClassifier.from_hparams')
    @patch('emotion_analyzer.torchaudio.load')
    @patch('emotion_analyzer.torch.argmax')
    @patch('emotion_analyzer.F.softmax')
    def test_analyze_emotion(self, mock_softmax, mock_argmax, mock_load, mock_classifier):
        """Test the emotion analyzer functionality."""
        # Mock classifier
        mock_classifier_instance = MagicMock()
        mock_classifier.return_value = mock_classifier_instance
        
        # Mock wav2vec2 output
        mock_classifier_instance.mods.wav2vec2.return_value = MagicMock()
        mock_classifier_instance.mods.avg_pool.return_value = MagicMock()
        mock_classifier_instance.mods.output_mlp.return_value = MagicMock()
        
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
        from emotion_analyzer import analyze_emotion
        result = analyze_emotion("dummy_path.wav")
        
        # Verify result
        assert result == "HAPPY"
        
        # Verify classifier was called
        mock_classifier.assert_called_once()
        mock_classifier_instance.mods.wav2vec2.assert_called_once()
        
        # Verify label decoding
        mock_classifier_instance.hparams.label_encoder.decode_ndim.assert_called_once()

# Additional fixtures and test cases can be added as needed