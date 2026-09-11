import librosa

def process_music(audio_file_path):
    """
    Process audio for Music & Rhythm Search using librosa's advanced beat tracker.
    """
    print('Running Music & Rhythm Search task...')
    try:
        y, sr = librosa.load(audio_file_path)
        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
        print(f"Estimated Tempo: {tempo[0]:.2f} BPM")
        return {"status": "success", "tempo": tempo[0]}
    except Exception as e:
        print(f"Music analysis failed: {e}")
        return {"status": "error", "tempo": None}
