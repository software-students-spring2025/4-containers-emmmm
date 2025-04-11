"""Analyze audio emotion by using pre-trained model"""

import torch
import torch.nn.functional as F
import torchaudio

from speechbrain.inference import EncoderClassifier


def analyze_emotion(file_path):
    """Apply the third party pre-trained model to analyze the audio."""
    print("Loading model...")
    classifier = EncoderClassifier.from_hparams(
        source="speechbrain/emotion-recognition-wav2vec2-IEMOCAP",
        savedir="pretrained_models/emotion-recognition",
    )

    classifier.hparams.label_encoder.expect_len(4)
    print("Loading audio...")
    waveform, _ = torchaudio.load(file_path)
    print("Extracting features with wav2vec2...")

    with torch.no_grad():
        wav2vec_out = classifier.mods.wav2vec2(waveform)
        pooled = classifier.mods.avg_pool(wav2vec_out)
        logits = classifier.mods.output_mlp(pooled)
        logits = logits.squeeze()
        probs = F.softmax(logits, dim=0)
    top_index = torch.argmax(probs).item()
    label = classifier.hparams.label_encoder.decode_ndim(torch.tensor(top_index))
    confidence = probs[top_index].item()
    print(f"Detected Emotion: {label} (probability: {confidence:.4f})")
    return label
