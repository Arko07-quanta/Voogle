from transformers import pipeline

print("Loading SOTA Language Identification Model (MMS)...")
lang_classifier = pipeline("audio-classification", model="facebook/mms-lid-126")

def process_language(audio_signal):
    """
    Process audio for Language & Accent Identification using MMS.
    """
    print('Running Language & Accent Identification task...')
    try:
        result = lang_classifier({"array": audio_signal, "sampling_rate": 16000})
        top_lang = result[0]['label']
        print(f"Language Result: {top_lang}")
        return {"status": "success", "language": top_lang, "details": result}
    except Exception as e:
        print(f"Language identification failed: {e}")
        return {"status": "error", "language": "unknown"}
