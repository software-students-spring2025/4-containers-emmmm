"""
Tests for the machine learning client component of the Voice Emotion Detection system.
"""

import io
import os
import json
import tempfile
import sys
from unittest.mock import MagicMock, patch

import pytest


class MockTorch:
    """Mocked torch module."""

    def no_grad(self):
        """Mock no_grad context manager."""

        class NoGradContext:
            def __enter__(self):
                return None

            def __exit__(self, *args):
                pass

        return NoGradContext()

    @staticmethod
    def argmax(_):
        """Mock argmax function."""
        mock_result = MagicMock()
        mock_result.item.return_value = 0
        return mock_result

    @staticmethod
    def tensor(value):  # pylint: disable=unused-argument
        """Mock tensor function."""
        return value


class MockFunctional:
    """Mocked torch.nn.functional module."""

    @staticmethod
    def softmax(_, __=0):  # pylint: disable=unused-argument
        """Mock softmax function."""
        mock_probs = MagicMock()
        mock_probs.__getitem__.return_value.item.return_value = 0.85
        return mock_probs


torch_mock = MockTorch()
torch_mock.nn = MagicMock()
torch_mock.nn.functional = MockFunctional()


class MockTorchaudio:
    """Mocked torchaudio module."""

    @staticmethod
    def load(_):  # pylint: disable=unused-argument
        """Mock audio loading."""
        return MagicMock(), MagicMock()


class MockEncoderClassifier:
    """Mocked SpeechBrain EncoderClassifier."""

    @staticmethod
    def from_hparams(_, __):  # pylint: disable=unused-argument
        """Mocked from_hparams method."""
        mock_classifier = MagicMock()
        mock_classifier.mods = MagicMock()
        mock_classifier.mods.wav2vec2 = MagicMock()
        mock_classifier.mods.avg_pool = MagicMock()
        mock_classifier.mods.output_mlp = MagicMock()

        mock_classifier.hparams = MagicMock()
        mock_classifier.hparams.label_encoder = MagicMock()
        mock_classifier.hparams.label_encoder.decode_ndim.return_value = "HAPPY"
        mock_classifier.hparams.label_encoder.expect_len = MagicMock()

        return mock_classifier


speechbrain_mock = MagicMock()
speechbrain_mock.inference = MagicMock()
speechbrain_mock.inference.EncoderClassifier = MockEncoderClassifier

sys.modules["torch"] = torch_mock
sys.modules["torch.nn"] = torch_mock.nn
sys.modules["torch.nn.functional"] = torch_mock.nn.functional
sys.modules["torchaudio"] = MockTorchaudio()
sys.modules["speechbrain"] = speechbrain_mock
sys.modules["speechbrain.inference"] = speechbrain_mock.inference

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from main import app, analyze_emotion  # pylint: disable=wrong-import-position


@pytest.fixture(autouse=True)
def mock_mongo_client(monkeypatch):
    """Automatically apply MongoDB client mock to all tests."""
    mongo_client_mock = MagicMock()
    mock_db = MagicMock()
    mock_collection = MagicMock()
    mock_collection.insert_one.return_value.inserted_id = "mock_id"
    mock_db.__getitem__.return_value = mock_collection
    mongo_client_mock.return_value.__getitem__.return_value = mock_db

    monkeypatch.setattr("pymongo.MongoClient", mongo_client_mock)
    return mongo_client_mock


@pytest.fixture
def client():
    """Create a test client for the Flask app."""
    app.config["TESTING"] = True
    with app.test_client() as test_client:
        yield test_client


DUMMY_AUDIO = b"mock audio data"


def test_analyze_no_file(client):  # pylint: disable=redefined-outer-name
    """Test error handling when no file is uploaded."""
    response = client.post("/analyze")
    assert response.status_code == 400
    result = json.loads(response.data)
    assert "error" in result
    assert "No file part" in result["error"]


@patch("main.analyze_emotion")
def test_analyze_with_error(mock_analyze, client):  # pylint: disable=redefined-outer-name
    """Test handling of errors during analysis."""
    mock_analyze.side_effect = Exception("Analysis failed")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp:
        temp.write(DUMMY_AUDIO)
        temp_name = temp.name

    try:
        with open(temp_name, "rb") as file:
            data = {
                "audio": (io.BytesIO(file.read()), "test_audio.wav", "audio/wav")
            }

        response = client.post(
            "/analyze", data=data, content_type="multipart/form-data"
        )

        assert response.status_code == 500
        result = json.loads(response.data)
        assert "error" in result
        assert "Analysis failed" in result["error"]
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


@patch("emotion_analyzer.EncoderClassifier.from_hparams")
@patch("emotion_analyzer.torchaudio.load")
@patch("emotion_analyzer.torch.argmax")
@patch("emotion_analyzer.F.softmax")
@patch("torch.no_grad")
def test_emotion_analyzer(
    mock_no_grad, mock_softmax, mock_argmax, mock_load, mock_classifier
):
    """Test the emotion analyzer functionality."""
    mock_context = MagicMock()
    mock_no_grad.return_value.__enter__.return_value = mock_context

    mock_classifier_instance = MagicMock()
    mock_classifier.return_value = mock_classifier_instance

    mock_wav2vec_out = MagicMock()
    mock_classifier_instance.mods.wav2vec2.return_value = mock_wav2vec_out

    mock_pooled = MagicMock()
    mock_classifier_instance.mods.avg_pool.return_value = mock_pooled

    mock_logits = MagicMock()
    mock_classifier_instance.mods.output_mlp.return_value = mock_logits
    mock_logits.squeeze.return_value = MagicMock()

    mock_load.return_value = (MagicMock(), MagicMock())

    mock_probs = MagicMock()
    mock_softmax.return_value = mock_probs
    mock_argmax.return_value.item.return_value = 2
    mock_probs.__getitem__.return_value.item.return_value = 0.85

    mock_classifier_instance.hparams.label_encoder.decode_ndim.return_value = "HAPPY"

    result = analyze_emotion("dummy_path.wav")
    assert result == "HAPPY"
