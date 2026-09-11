import os

def process_diarization(audio_file_path):
    """
    Process audio for Speaker Detection & Diarization using Pyannote.
    """
    print('Running Diarization task (SOTA Pyannote)...')
    
    # Grab the token from an environment variable, or use the provided one
    hf_token = os.environ.get("HF_TOKEN", "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")
    
    if not hf_token:
        print("\\n[ERROR] Pyannote requires a HuggingFace token.")
        print("Please set the HF_TOKEN environment variable and accept the terms of use.")
        return {"status": "skipped", "reason": "Missing HF_TOKEN"}
    
    # --- SOTA Implementation ---
    try:
        from pyannote.audio import Pipeline
        # Load the diarization pipeline
        pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", token=hf_token)
        
        # Run diarization on the audio file
        diarization = pipeline(audio_file_path)
        
        speakers_detected = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            print(f"start={turn.start:.1f}s stop={turn.end:.1f}s speaker_{speaker}")
            speakers_detected.append({"start": turn.start, "end": turn.end, "speaker": speaker})
            
        return {"status": "success", "speakers": speakers_detected}
    except Exception as e:
        print(f"Diarization failed: {e}")
        return {"status": "error", "reason": str(e)}
