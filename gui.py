import gradio as gr
from main import VooglePipeline

print("Initializing Voogle Pipeline for GUI...")
pipeline = VooglePipeline()

def analyze_audio(audio_path, selected_tasks):
    if not audio_path:
        return "Please provide an audio file."
    
    if not selected_tasks:
        return "Please select at least one task to run."
    
    print(f"\n[GUI] Received audio file: {audio_path} for tasks: {selected_tasks}")
    results = pipeline.process(audio_path, selected_tasks=selected_tasks)
    
    if not results:
        return "Error processing audio. Check the console logs."
    
    # Format results nicely
    output = "=== Voogle Analysis Results ===\n\n"
    
    if 'Transcription' in selected_tasks:
        trans_text = results['transcription'].get('text', 'N/A')
        output += f"📝 Transcription: {trans_text}\n"
    
    if 'Emotion' in selected_tasks:
        emo_res = results['emotion'].get('emotions', 'N/A')
        if isinstance(emo_res, list) and len(emo_res) > 0 and 'label' in emo_res[0]:
            emo_res = emo_res[0]['label']
        output += f"🎭 Emotion: {emo_res}\n"
    
    if 'Language' in selected_tasks:
        output += f"🌍 Language: {results['language'].get('language', 'N/A')}\n"
    
    if 'Music' in selected_tasks:
        output += f"🎵 Tempo: {results['music'].get('tempo', 'N/A')} BPM\n"
    
    if 'Diarization' in selected_tasks:
        if results['diarization'].get('status') == 'error':
            output += f"👥 Diarization: [ERROR] {results['diarization'].get('reason', 'Unknown Error')}\n"
        else:
            speakers = results['diarization'].get('speakers', [])
            output += f"👥 Diarization: {len(speakers)} segments detected\n"
            for s in speakers:
                output += f"   - Speaker {s.get('speaker', '?')}: {s.get('start', 0):.1f}s to {s.get('end', 0):.1f}s\n"
            
    if 'Retrieval' in selected_tasks:
        output += f"🔍 Retrieval Match: {results['retrieval'].get('similarity', 'N/A')}\n"
        
    return output

# Create Gradio Interface
iface = gr.Interface(
    fn=analyze_audio,
    inputs=[
        gr.Audio(type="filepath", sources=["upload", "microphone"], label="Upload Audio or Record Live"),
        gr.CheckboxGroup(["Diarization", "Emotion", "Retrieval", "Transcription", "Language", "Music"], label="Select Tasks to Run (fewer tasks = much faster!)", value=["Transcription"])
    ],
    outputs=gr.Textbox(label="Analysis Results", lines=15),
    title="Voogle - The Ultimate Voice Search Engine",
    description="Upload an audio file or record from your microphone, choose your tasks, and analyze it using State-of-the-Art Signal Processing Algorithms!",
    theme="huggingface"
)

if __name__ == "__main__":
    print("\nStarting GUI... Open the link below in your browser!")
    iface.launch(inbrowser=True)
