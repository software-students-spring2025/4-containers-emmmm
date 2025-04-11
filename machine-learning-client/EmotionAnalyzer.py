import torch
import torchaudio
import torch.nn.functional as F
from speechbrain.inference import EncoderClassifier

def analyze_emotion(file_path):
    print("sLoading model...")
    classifier = EncoderClassifier.from_hparams(
        source="speechbrain/emotion-recognition-wav2vec2-IEMOCAP",
        savedir="pretrained_models/emotion-recognition"
    )
    
    # Address the warning
    classifier.hparams.label_encoder.expect_len(4)

    print("Loading audio...")
    waveform, sample_rate = torchaudio.load(file_path)

    print("Extracting features with wav2vec2...")
    with torch.no_grad():
        wav2vec_out = classifier.mods.wav2vec2(waveform)
        pooled = classifier.mods.avg_pool(wav2vec_out)
        logits = classifier.mods.output_mlp(pooled)
        logits = logits.squeeze()  # Remove all dimensions of size 1
        probs = F.softmax(logits, dim=0)
        
    top_index = torch.argmax(probs).item()
    label = classifier.hparams.label_encoder.decode_ndim(torch.tensor(top_index))
    confidence = probs[top_index].item()

    print(f"Detected Emotion: {label} (probability: {confidence:.4f})")

    return label