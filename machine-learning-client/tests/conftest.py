"""
Pytest configuration file for the ML client tests.
"""

import sys
import pytest
from unittest.mock import MagicMock

# Mock the speechbrain module
class MockEncoderClassifier:
    @staticmethod
    def from_hparams(source, savedir):
        mock_classifier = MagicMock()
        mock_classifier.mods = MagicMock()
        mock_classifier.hparams = MagicMock()
        mock_classifier.hparams.label_encoder = MagicMock()
        mock_classifier.hparams.label_encoder.decode_ndim = MagicMock(return_value="HAPPY")
        mock_classifier.hparams.label_encoder.expect_len = MagicMock()
        return mock_classifier

# Create mock for speechbrain
speechbrain_mock = MagicMock()
speechbrain_mock.inference = MagicMock()
speechbrain_mock.inference.EncoderClassifier = MockEncoderClassifier

# Create mock for torchaudio
torchaudio_mock = MagicMock()
torchaudio_mock.load = MagicMock(return_value=(MagicMock(), MagicMock()))

# Create mock for torch
torch_mock = MagicMock()
torch_mock.no_grad = MagicMock()
torch_mock.argmax = MagicMock(return_value=MagicMock(item=MagicMock(return_value=0)))
torch_mock.tensor = MagicMock()
torch_mock.nn = MagicMock()
torch_mock.nn.functional = MagicMock()
mock_probs = MagicMock()
mock_probs.__getitem__.return_value.item.return_value = 0.85
torch_mock.nn.functional.softmax = MagicMock(return_value=mock_probs)

# Add mocks to sys.modules
sys.modules['speechbrain'] = speechbrain_mock
sys.modules['speechbrain.inference'] = speechbrain_mock.inference
sys.modules['torchaudio'] = torchaudio_mock
sys.modules['torch'] = torch_mock
sys.modules['torch.nn'] = torch_mock.nn
sys.modules['torch.nn.functional'] = torch_mock.nn.functional

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
