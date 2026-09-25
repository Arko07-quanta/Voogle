import os
from main import VooglePipeline

def test_pipeline():
    pipeline = VooglePipeline()
    
    print("\n" + "="*50)
    print("1. Testing Emotion Identification (Sad Sample)")
    print("="*50)
    sad_file = "datasets/emotions/sad/sample_0.wav"
    if os.path.exists(sad_file):
        pipeline.process(sad_file, selected_tasks=['Emotion'])
    else:
        print("Sad sample not found.")
        
    print("\n" + "="*50)
    print("2. Testing Emotion Identification (Angry Sample)")
    print("="*50)
    angry_file = "datasets/emotions/angry/sample_0.wav"
    if os.path.exists(angry_file):
        pipeline.process(angry_file, selected_tasks=['Emotion'])
    else:
        print("Angry sample not found.")
        
    print("\n" + "="*50)
    print("3. Testing Language Identification (English Sample)")
    print("="*50)
    en_file = "datasets/languages/en/sample_0.wav"
    if os.path.exists(en_file):
        pipeline.process(en_file, selected_tasks=['Language'])
    else:
        print("English sample not found.")

    print("\n" + "="*50)
    print("4. Testing Language Identification (German Sample)")
    print("="*50)
    de_file = "datasets/languages/de/sample_0.wav"
    if os.path.exists(de_file):
        pipeline.process(de_file, selected_tasks=['Language'])
    else:
        print("German sample not found.")

    print("\n" + "="*50)
    print("5. Testing Diarization, Transcription & Music (Neutral Sample)")
    print("="*50)
    neutral_file = "datasets/emotions/neutral/sample_0.wav"
    if os.path.exists(neutral_file):
        pipeline.process(neutral_file, selected_tasks=['Diarization', 'Transcription', 'Music'])
    else:
        print("Neutral sample not found.")

    print("\n" + "="*50)
    print("6. Testing Audio Retrieval & Matching")
    print("="*50)
    if os.path.exists(en_file):
        # We test retrieval on the English sample
        pipeline.process(en_file, selected_tasks=['Retrieval'])
    
    print("\nAll Comprehensive Tests Completed!")

if __name__ == "__main__":
    test_pipeline()
