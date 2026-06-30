import librosa
import torch
import torch.nn.functional as F
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

MODEL_NAME = "firdhokk/speech-emotion-recognition-with-facebook-wav2vec2-large-xlsr-53"

device = "cuda" if torch.cuda.is_available() else "cpu"
feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_NAME)
model = AutoModelForAudioClassification.from_pretrained(MODEL_NAME).to(device)

def detect_emotion(wav_path: str) -> dict[str, float]:
    """
    Detects the emotion expressed in a speech recording using a Wav2Vec2 model.

    Args:
        wav_path (str): Path to a 16 kHz mono WAV file.

    Returns:
        dict[str, float]: Emotion labels mapped to predicted probability scores.
    """
    audio, sr = librosa.load(wav_path, sr=16000)
    inputs = feature_extractor(audio, sampling_rate=16000, return_tensors="pt").to(device)

    with torch.no_grad():
        logits = model(**inputs).logits

    probs = F.softmax(logits, dim=1)
    labels = model.config.id2label
    result = {labels[i]: probs[0][i].item() for i in range(len(labels))}

    return result