"""
Tests for the web application component of the Voice Emotion Detection system.
"""

import json
import unittest
from unittest.mock import patch, MagicMock
import io
import pytest
import requests
from web-app.app import app as flask_app

class TestWebApp(unittest.TestCase):
    """Test cases for the web application."""

    def setUp(self):
        """Set up test client and other test variables."""
        self.app = flask_app
        self.app.config['TESTING'] = True
        self.client = self.app.test_client()
        self.app_context = self.app.app_context()
        self.app_context.push()

    def tearDown(self):
        """Clean up after tests."""
        self.app_context.pop()

    def test_home_route(self):
        """Test that the home page loads correctly."""
        response = self.client.get('/')
        assert response.status_code == 200
        assert b'Voice Emotion Detector' in response.data

    @patch('app.requests.post')
    def test_upload_success(self, mock_post):
        """Test successful audio upload and analysis."""
        # Mock the response from the ML client
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "status": "success",
            "result": {
                "emotion": "HAPPY",
                "timestamp": "2025-04-13T12:00:00Z"
            }
        }
        mock_response.status_code = 200
        mock_post.return_value = mock_response
        # Create a mock audio file
        audio_file = (io.BytesIO(b"mock audio data"), 'test_audio.wav')
        # Make the request
        response = self.client.post(
            '/upload',
            data={'audio': audio_file},
            content_type='multipart/form-data'
        )
        # Check response
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['status'] == 'success'
        assert data['result']['emotion'] == 'HAPPY'
        # Verify ML client was called correctly
        mock_post.assert_called_once()
        args, kwargs = mock_post.call_args
        assert 'http://ml-client:6000/analyze' in args
        assert 'files' in kwargs
        assert 'audio' in kwargs['files']

    def test_upload_no_file(self):
        """Test error handling when no file is uploaded."""
        response = self.client.post('/upload')
        assert response.status_code == 400
        data = json.loads(response.data)
        assert 'error' in data
        assert 'No audio file uploaded' in data['error']

    def test_upload_empty_filename(self):
        """Test error handling when file has no name."""
        audio_file = (io.BytesIO(b""), '')
        response = self.client.post(
            '/upload',
            data={'audio': audio_file},
            content_type='multipart/form-data'
        )
        assert response.status_code == 400
        data = json.loads(response.data)
        assert 'error' in data
        assert 'No selected file' in data['error']

    @patch('app.requests.post')
    def test_upload_ml_client_error(self, mock_post):
        """Test handling of ML client errors."""
        # Mock a connection error
        mock_post.side_effect = requests.RequestException("ML client connection error")
        # Create a mock audio file
        audio_file = (io.BytesIO(b"mock audio data"), 'test_audio.wav')
        # Make the request
        response = self.client.post(
            '/upload',
            data={'audio': audio_file},
            content_type='multipart/form-data'
        )
        # Check response
        assert response.status_code == 500
        data = json.loads(response.data)
        assert 'error' in data
        assert 'Failed to connect to ML client' in data['error']

    @patch('app.requests.post')
    def test_upload_invalid_json_response(self, mock_post):
        """Test handling of invalid JSON from ML client."""
        # Mock invalid JSON response
        mock_response = MagicMock()
        mock_response.json.side_effect = json.JSONDecodeError("Invalid JSON", "", 0)
        mock_response.text = "Not a JSON response"
        mock_response.status_code = 200
        mock_post.return_value = mock_response
        # Create a mock audio file
        audio_file = (io.BytesIO(b"mock audio data"), 'test_audio.wav')
        # Make the request
        response = self.client.post(
            '/upload',
            data={'audio': audio_file},
            content_type='multipart/form-data'
        )
        # Check response
        assert response.status_code == 500
        data = json.loads(response.data)
        assert 'error' in data
        assert 'ML Client did not return valid JSON' in data['error']

    @patch('app.requests.get')
    @patch('app.client')
    def test_health_check_all_services_up(self, mock_mongo_client, mock_requests_get):
        """Test health check when all services are up."""
        # Mock MongoDB connection
        mock_mongo_client.server_info.return_value = True
        # Mock ML client response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_requests_get.return_value = mock_response
        # Make request
        response = self.client.get('/health')
        # Check response
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['status'] == 'ok'
        assert data['mongodb_connected'] is True
        assert data['ml_client_connected'] is True

    @patch('app.requests.get')
    @patch('app.DB', None)  # Simulate MongoDB not connected
    def test_health_check_mongo_down(self, mock_requests_get):
        """Test health check when MongoDB is down."""
        # Mock ML client response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_requests_get.return_value = mock_response
        # Make request
        response = self.client.get('/health')
        # Check response
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['status'] == 'ok'  # Overall status still ok
        assert data['mongodb_connected'] is False
        assert data['ml_client_connected'] is True

    @patch('app.requests.get')
    def test_health_check_ml_client_down(self, mock_requests_get):
        """Test health check when ML client is down."""
        # Mock ML client error
        mock_requests_get.side_effect = requests.RequestException("ML client down")
        # Make request
        response = self.client.get('/health')
        # Check response
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['status'] == 'ok'  # Overall status still ok
        assert data['ml_client_connected'] is False

@pytest.fixture
def test_client():
    """Create a test client for the app."""
    with flask_app.test_client() as client:
        with flask_app.app_context():
            yield client
