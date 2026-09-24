import gradio as gr
from main import VooglePipeline
from src.tasks.retrieval import add_song_sample, add_voice_sample, process_retrieval, get_song_tree, get_voice_tree
from src.tasks.language import add_language_sample, get_language_tree
import librosa

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
        lang_res = results['language']
        output += f"🌍 Top Language Match: {lang_res.get('language', 'N/A')} (Similarity: {lang_res.get('similarity', 0):.4f})\n"
        matches = lang_res.get('matches', [])
        if matches:
            output += "   KD-Tree Language Candidates:\n"
            for m in matches:
                output += f"     • {m['label']}: Sim {m['similarity']:.4f}\n"
    
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
        ret_res = results['retrieval']
        best_match = ret_res.get('best_match')
        if best_match:
            output += f"🔍 KD-Tree Song Match: {best_match['label']} (Similarity: {best_match['similarity']:.4f})\n"
            for m in ret_res.get('matches', []):
                output += f"     • {m['label']} ({m['path']}): Sim {m['similarity']:.4f}\n"
        else:
            output += f"🔍 Retrieval: {ret_res.get('message', 'No indexed items')}\n"
        
    return output

def add_to_tree_dataset(audio_path, label, dataset_type):
    if not audio_path or not label:
        return "❌ Error: Please provide both an audio file and a label/name."
    
    try:
        if dataset_type == "Song":
            res = add_song_sample(label.strip(), audio_path)
            return f"✅ Successfully added '{label}' to Songs KD-Tree! (Total songs indexed: {res['total_samples']})"
        elif dataset_type == "Voice":
            res = add_voice_sample(label.strip(), audio_path)
            return f"✅ Successfully added '{label}' to Voices KD-Tree! (Total voices indexed: {res['total_samples']})"
        elif dataset_type == "Language":
            res = add_language_sample(label.strip(), audio_path)
            return f"✅ Successfully added '{label}' to Languages KD-Tree! (Total languages indexed: {res['total_samples']})"
        else:
            return "❌ Unknown dataset type."
    except Exception as e:
        return f"❌ Failed to index sample: {e}"

def search_voice_tree(audio_path):
    if not audio_path:
        return "Please upload an audio sample to match."
    try:
        y, sr = librosa.load(audio_path, sr=16000)
        res = process_retrieval(audio_signal=y, sample_rate=sr, match_type="voice", top_k=5)
        if not res.get("matches"):
            return res.get("message", "No matches found. Make sure to add samples to the Voice KD-Tree first.")
        out = "=== Voice Match KD-Tree Results ===\n\n"
        for i, m in enumerate(res["matches"], 1):
            out += f"{i}. {m['label']} - Cosine Similarity: {m['similarity']:.4f}\n   File: {m['path']}\n"
        return out
    except Exception as e:
        return f"Error matching voice: {e}"

def list_indexed_trees():
    song_tree = get_song_tree()
    voice_tree = get_voice_tree()
    lang_tree = get_language_tree()
    
    out = "=== Locally Indexed Metric Trees ===\n\n"
    out += f"🎵 Songs Tree ({len(song_tree.items)} indexed):\n"
    for item in song_tree.items:
        out += f"  - [{item['id']}] {item['label']} ({item['path']})\n"
    
    out += f"\n🗣️ Voices Tree ({len(voice_tree.items)} indexed):\n"
    for item in voice_tree.items:
        out += f"  - [{item['id']}] {item['label']} ({item['path']})\n"

    out += f"\n🌍 Languages Tree ({len(lang_tree.items)} indexed):\n"
    for item in lang_tree.items:
        out += f"  - [{item['id']}] {item['label']}\n"
        
    return out

# Gradio Blocks UI
with gr.Blocks(title="Voogle - Voice Search & Signal Tree Matcher") as demo:
    gr.Markdown("# 🚀 Voogle Audio Search Engine\n### Classical Signal-Processing & Local Metric KD-Tree Matching")

    with gr.Tab("🎙️ Multi-Task Audio Analyzer"):
        with gr.Row():
            with gr.Column():
                input_audio = gr.Audio(type="filepath", sources=["upload", "microphone"], label="Input Audio")
                task_selector = gr.CheckboxGroup(
                    ["Diarization", "Emotion", "Retrieval", "Transcription", "Language", "Music"],
                    label="Select Tasks to Run",
                    value=["Transcription", "Language", "Retrieval"]
                )
                analyze_btn = gr.Button("Analyze Audio", variant="primary")
            with gr.Column():
                analysis_output = gr.Textbox(label="Analysis Results", lines=16)
        
        analyze_btn.click(fn=analyze_audio, inputs=[input_audio, task_selector], outputs=analysis_output)

    with gr.Tab("🌲 Manage & Expand Tree Datasets"):
        gr.Markdown("Add your own reference audio samples into local KD-Trees for **Voice**, **Song**, and **Language** identification.")
        with gr.Row():
            with gr.Column():
                sample_audio = gr.Audio(type="filepath", sources=["upload", "microphone"], label="Sample Audio File")
                sample_label = gr.Textbox(label="Label / Name (e.g. 'Alice', 'Beethoven Symphony 5', 'French')", placeholder="Enter identifier")
                dataset_type = gr.Radio(["Song", "Voice", "Language"], label="Tree Index Target", value="Song")
                add_btn = gr.Button("Index Sample into KD-Tree", variant="primary")
                add_status = gr.Textbox(label="Status / Confirmation", lines=3)
            with gr.Column():
                refresh_btn = gr.Button("Inspect Indexed Trees")
                tree_contents = gr.Textbox(label="Indexed Samples across Trees", lines=15)
        
        add_btn.click(fn=add_to_tree_dataset, inputs=[sample_audio, sample_label, dataset_type], outputs=add_status)
        refresh_btn.click(fn=list_indexed_trees, inputs=[], outputs=tree_contents)

    with gr.Tab("🗣️ Dedicated Voice Matcher"):
        gr.Markdown("Query an unknown voice sample against your local **Voices KD-Tree**.")
        with gr.Row():
            with gr.Column():
                voice_query_audio = gr.Audio(type="filepath", sources=["upload", "microphone"], label="Query Voice Audio")
                voice_search_btn = gr.Button("Search Voice Tree", variant="primary")
            with gr.Column():
                voice_search_output = gr.Textbox(label="Matching Speakers", lines=10)
        
        voice_search_btn.click(fn=search_voice_tree, inputs=[voice_query_audio], outputs=voice_search_output)

if __name__ == "__main__":
    print("\nStarting Voogle GUI with Tree-based search...")
    demo.launch(inbrowser=True)
