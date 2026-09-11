import gradio as gr
from main import VooglePipeline

print("Initializing Voogle Pipeline for GUI...")
pipeline = VooglePipeline()

def analyze_audio(audio_path):
    if not audio_path:
        return "Please provide an audio file."
    
    print(f"\\n[GUI] Received audio file: {audio_path}")
    results = pipeline.process(audio_path)
    
    if not results:
        return "Error processing audio. Check the console logs."
    
    # Format results nicely
    output = "=== Voogle Analysis Results ===\\n\\n"
    
    # Transcription
    trans_text = results['transcription'].get('text', 'N/A')
    output += f"📝 Transcription: {trans_text}\\n"
    
    # Emotion (handling dict vs string depending on task output)
    emo_res = results['emotion'].get('emotions', 'N/A')
    if isinstance(emo_res, list) and len(emo_res) > 0 and 'label' in emo_res[0]:
        emo_res = emo_res[0]['label']
    output += f"🎭 Emotion: {emo_res}\\n"
    
    # Language
    output += f"🌍 Language: {results['language'].get('language', 'N/A')}\\n"
    
    # Music/Tempo
    output += f"🎵 Tempo: {results['music'].get('tempo', 'N/A')} BPM\\n"
    
    # Diarization
    speakers = results['diarization'].get('speakers', [])
    output += f"👥 Diarization: {len(speakers)} segments detected\\n"
    for s in speakers:
        output += f"   - Speaker {s.get('speaker', '?')}: {s.get('start', 0):.1f}s to {s.get('end', 0):.1f}s\\n"
        
    return output

# Create Gradio Interface
iface = gr.Interface(
    fn=analyze_audio,
    inputs=gr.Audio(type="filepath", sources=["upload", "microphone"], label="Upload Audio or Record Live"),
    outputs=gr.Textbox(label="Analysis Results", lines=15),
    title="Voogle - The Ultimate Voice Search Engine",
    description="Upload an audio file or record from your microphone to analyze it using State-of-the-Art Neural Networks!",
    theme="huggingface"
)

if __name__ == "__main__":
    print("\\nStarting GUI... Open the link below in your browser!")
    iface.launch(inbrowser=True)
