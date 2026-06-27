import ffmpeg

def extract_audio(input_path, output_path):
    """
    Extracts audio from a video or audio file and saves it as a 16kHz mono WAV file.

    Args:
        input_path (str): Path to the input file (.mp4, .wav, .mp3, etc.)
        output_path (str): Path where the extracted WAV file will be saved.

    Returns:
        str: Path to the output WAV file.
    """
    try:
        ffmpeg.input(input_path).output(output_path, ar=16000, ac=1).run(overwrite_output=True,quiet=True)
        return output_path
    except Exception as e:
        print(f"Audio extraction failed: {e}")
        raise