from transformers import pipeline

print("Loading SOTA Emotion Recognition Model...")
emotion_classifier = pipeline("audio-classification", model="superb/wav2vec2-base-superb-er")

def process_emotion(audio_signal):
    """
    Process audio for Emotion Identification using Wav2Vec2.
    """
    print('Running Emotion Identification task...')
    try:
        result = emotion_classifier({"array": audio_signal, "sampling_rate": 16000})
        # Result is a list of dicts with 'label' and 'score'
        top_emotion = result[0]['label']
        print(f"Emotion Result: {top_emotion}")
        return {"status": "success", "emotions": result}
    except Exception as e:
        print(f"Emotion classification failed: {e}")
        return {"status": "error", "emotions": []}
