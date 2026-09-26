import os
import subprocess
import librosa
import soundfile as sf
import time
import numpy as np
from src.tasks.emotion import process_emotion
from src.tasks.language import process_language
from src.tasks.retrieval import process_retrieval

# --- Configuration ---
eval_voices_emo = [
    "en-US-SteffanNeural", "en-GB-ThomasNeural", "en-AU-CarlyNeural", 
    "en-IE-EmilyNeural", "en-CA-ClaraNeural", "en-US-RogerNeural",
    "en-US-AvaNeural", "en-US-AndrewNeural", "en-US-EmmaNeural", "en-US-BrianNeural"
]

eval_voices_lang = {
    "en": ["en-US-SteffanNeural", "en-GB-ThomasNeural", "en-AU-CarlyNeural", "en-IE-EmilyNeural", "en-CA-ClaraNeural"],
    "es": ["es-ES-AbrilNeural", "es-MX-JorgeNeural", "es-AR-ElenaNeural", "es-CO-GonzaloNeural", "es-US-PalomaNeural"],
    "zh": ["zh-CN-YunjianNeural", "zh-TW-YunJheNeural", "zh-HK-WanLungNeural", "zh-CN-XiaoyiNeural", "zh-TW-HsiaoYuNeural"],
    "de": ["de-DE-KillianNeural", "de-AT-JonasNeural", "de-CH-JanNeural", "de-DE-KlarissaNeural", "de-DE-LouisaNeural"],
    "bn": ["bn-BD-NabanitaNeural", "bn-BD-PradeepNeural", "bn-IN-BashkarNeural", "bn-IN-TanishaaNeural", "bn-BD-NabanitaNeural"]
}

phrases_emo = {
    "happy": "I am so absolutely thrilled about this, it is amazing!",
    "angry": "I am mad as hell, this is completely ridiculous!",
    "sad": "I feel so empty and depressed, everything is just awful."
}

phrases_lang = {
    "en": "The fast brown fox jumps over the lazy dog today.",
    "es": "El rápido zorro marrón salta sobre el perro perezoso hoy.",
    "zh": "敏捷的棕色狐狸今天跳过了那只懒狗。",
    "de": "Ein schnelles braunes Fuchs springt heute über den faulen Hund.",
    "bn": "দ্রুত বাদামী শিয়াল আজ অলস কুকুরের উপর দিয়ে লাফিয়ে পড়ে।"
}

def safe_tts(voice, text, out_wav):
    temp_mp3 = out_wav.replace('.wav', '.mp3')
    try:
        subprocess.run([
            "edge-tts", "--voice", voice, "--text", text, "--write-media", temp_mp3
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        y, sr = librosa.load(temp_mp3, sr=16000)
        sf.write(out_wav, y, sr)
        os.remove(temp_mp3)
    except Exception as e:
        pass
    time.sleep(1.5)

def eval_emotion():
    print("--- Evaluating Emotion Recognition ---")
    correct = 0
    total = 0
    os.makedirs("eval_temp", exist_ok=True)
    
    for emotion, text in phrases_emo.items():
        for i, voice in enumerate(eval_voices_emo):
            wav_path = f"eval_temp/emo_{emotion}_{i}.wav"
            safe_tts(voice, text, wav_path)
            if not os.path.exists(wav_path):
                continue
                
            y, sr = librosa.load(wav_path, sr=16000)
            res = process_emotion(y, sr)
            pred = res.get("emotions", "unknown")
            
            if pred == emotion:
                correct += 1
            total += 1
            
    acc = (correct / total) * 100 if total > 0 else 0
    print(f"Emotion Accuracy: {acc:.2f}% ({correct}/{total})")
    return acc

def eval_language():
    print("--- Evaluating Language ID ---")
    correct = 0
    total = 0
    
    # Map edge-tts prefixes to KD-Tree labels
    label_map = {"en": "English", "es": "Spanish (Espaol)", "zh": "Zh", "de": "German (Deutsch)", "bn": "Bn"}
    # Note: Using precise substrings to handle unicode mismatch in Spanish if needed
    
    for lang_code, voices in eval_voices_lang.items():
        for i, voice in enumerate(voices):
            wav_path = f"eval_temp/lang_{lang_code}_{i}.wav"
            safe_tts(voice, phrases_lang[lang_code], wav_path)
            if not os.path.exists(wav_path):
                continue
                
            y, sr = librosa.load(wav_path, sr=16000)
            res = process_language(y, sr)
            pred = res.get("language", "unknown")
            
            # Simple substring match to bypass exact unicode mismatch
            target = label_map[lang_code][:5]
            if target in pred:
                correct += 1
            total += 1
            
    acc = (correct / total) * 100 if total > 0 else 0
    print(f"Language Accuracy: {acc:.2f}% ({correct}/{total})")
    return acc

def eval_retrieval():
    print("--- Evaluating Song Retrieval (Fingerprinting) ---")
    correct = 0
    total = 0
    
    songs = ["Vibe_Ace_Beat", "Nutcracker_Suite", "Brahms_Hungarian_Dance"]
    
    for song in songs:
        orig_path = f"demo_audio/{song}.wav"
        if not os.path.exists(orig_path):
            continue
            
        y, sr = librosa.load(orig_path, sr=16000)
        
        # Test 1: 5-second crop from middle
        start_samp = len(y) // 2
        chunk1 = y[start_samp : start_samp + 5*sr]
        
        # Test 2: 3-second crop with added white noise
        chunk2 = y[start_samp+5*sr : start_samp+8*sr]
        noise = np.random.randn(len(chunk2)) * 0.05
        chunk2_noisy = chunk2 + noise
        
        tests = [chunk1, chunk2_noisy]
        
        for t_idx, test_audio in enumerate(tests):
            res = process_retrieval(audio_signal=test_audio, sample_rate=sr)
            pred_match = res.get("best_match")
            pred = pred_match["label"] if pred_match else "None"
            
            if pred == song:
                correct += 1
            total += 1
            
    acc = (correct / total) * 100 if total > 0 else 0
    print(f"Retrieval Accuracy: {acc:.2f}% ({correct}/{total})")
    return acc

if __name__ == "__main__":
    import sys
    
    # We will override stdout just inside the specific calls to prevent terminal spam
    
    def run_silent(func, *args):
        class DummyOutput:
            def write(self, x): pass
            def flush(self): pass
        old_stdout = sys.stdout
        sys.stdout = DummyOutput()
        res = func(*args)
        sys.stdout = old_stdout
        return res

    # Patch the modules to use run_silent?
    # Better: just redefine the evaluation calls.
    # Actually, I'll just rewrite the main block to leave stdout alone, let the user see the logs!
    
    print("\n==========================================")
    print("       STARTING LARGE-SCALE EVALUATION      ")
    print("==========================================")
    
    emo_acc = eval_emotion()
    lang_acc = eval_language()
    ret_acc = eval_retrieval()
    
    print("\n==========================================")
    print("       LARGE-SCALE EVALUATION REPORT      ")
    print("==========================================")
    print(f"Emotion Recognition Accuracy : {emo_acc:.2f}%")
    print(f"Language ID Accuracy         : {lang_acc:.2f}%")
    print(f"Song Retrieval (Shazam)      : {ret_acc:.2f}%")
    print("==========================================")
