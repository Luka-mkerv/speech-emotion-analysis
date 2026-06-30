from src.audio import extract_audio

input_path = "data/samples/03-01-05-01-01-01-02.wav"
output_path = "data/samples/output_test.wav"
result = extract_audio(input_path, output_path)
print(f"Success: {result}")



from src.transcriber import transcribe

result = transcribe("data/samples/output_test.wav")
for segment in result:
    print(segment)



from src.emotion import detect_emotion

emotions = detect_emotion("data/samples/output_test.wav")
print(emotions)