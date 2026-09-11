from transformers import pipeline

# Load SOTA model globally to avoid reloading on each call
# Using whisper-tiny for local efficiency, can be upgraded to whisper-large
print("Loading SOTA Transcription Model (Whisper)...")
transcriber = pipeline("automatic-speech-recognition", model="openai/whisper-tiny")

def process_transcription(audio_signal):
    """
    Process audio for Speech-to-Text Transcription using OpenAI Whisper.
    """
    print('Running Speech-to-Text Transcription task...')
    try:
        # Pass raw audio array with sampling rate 16000 (Whisper's expected rate)
        result = transcriber({"array": audio_signal, "sampling_rate": 16000})
        print(f"Transcription Result: {result['text']}")
        return {"status": "success", "text": result['text']}
    except Exception as e:
        print(f"Transcription failed: {e}")
        return {"status": "error", "text": ""}
