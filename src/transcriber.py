import torch
from faster_whisper import WhisperModel

device = "cuda" if torch.cuda.is_available() else "cpu"
model = WhisperModel("small", device=device, compute_type="float16")

def transcribe(wav_path):
    """
    Transcribes a WAV file using faster-whisper.

    Args:
        wav_path (str): Path to the 16kHz mono WAV file.

    Returns:
        list: List of segments with text and timestamps.
    """
    segments, info = model.transcribe(wav_path)

    result = []
    for segment in segments:
        result.append({
            "start": segment.start,
            "end": segment.end,
            "text": segment.text.strip()
        })

    return result