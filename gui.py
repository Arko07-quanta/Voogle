import gradio as gr
import librosa
from main import VooglePipeline
from src.tasks.retrieval import (
    add_song_sample, 
    add_voice_sample, 
    process_retrieval, 
    get_song_tree, 
    get_voice_tree
)
from src.tasks.language import add_language_sample, get_language_tree
from src.tasks.emotion import add_emotion_sample, get_emotion_tree

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
        emo_res = results['emotion']
        output += f"🎭 Emotion (KD-Tree Match): {emo_res.get('emotions', 'N/A')} (Similarity: {emo_res.get('similarity', 0):.4f})\n"
        matches = emo_res.get('matches', [])
        if matches:
            output += "   KD-Tree Emotion Candidates:\n"
            for m in matches:
                output += f"     • {m['label']}: Sim {m['similarity']:.4f}\n"
    
    if 'Language' in selected_tasks:
        lang_res = results['language']
        output += f"🌍 Language (KD-Tree Match): {lang_res.get('language', 'N/A')} (Similarity: {lang_res.get('similarity', 0):.4f})\n"
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
            output += f"🔍 Song Match: {best_match['label']} (Confidence Score: {best_match['similarity']:.4f})\n"
            for m in ret_res.get('matches', []):
                output += f"     • {m['label']}: Best Sim {m['similarity']:.4f} (Avg: {m.get('avg_similarity', m['similarity']):.4f})\n"
        else:
            output += f"🔍 Retrieval: {ret_res.get('message', 'No indexed items found')}\n"
        
    return output

def add_to_tree_dataset(audio_path, label, dataset_type):
    if not audio_path:
        return "❌ Error: Please record or upload an audio file first."
    if not label or not label.strip():
        return "❌ Error: Please provide a label or person's name (e.g. 'Alice', 'John', 'happy')."
    
    clean_label = label.strip()
    try:
        if dataset_type == "Voice":
            res = add_voice_sample(clean_label, audio_path)
            return (
                f"✅ Added voice sample to 'datasets/voices/': '{clean_label}'!\n"
                f"• Samples for this person: {res['sample_count_for_label']}\n"
                f"• Total voice samples in KD-Tree: {res['total_samples']}\n"
                f"• Saved at: {res['saved_path']}"
            )
        elif dataset_type == "Song":
            res = add_song_sample(clean_label, audio_path)
            return (
                f"✅ Added song sample to 'datasets/songs/': '{clean_label}'!\n"
                f"• Samples for this track: {res['sample_count_for_label']}\n"
                f"• Total song samples in KD-Tree: {res['total_samples']}\n"
                f"• Saved at: {res['saved_path']}"
            )
        elif dataset_type == "Language":
            res = add_language_sample(clean_label, audio_path)
            return (
                f"✅ Added language sample to 'datasets/languages/': '{clean_label}'!\n"
                f"• Samples for this language: {res['sample_count_for_label']}\n"
                f"• Total language samples in KD-Tree: {res['total_samples']}\n"
                f"• Saved at: {res['saved_path']}"
            )
        elif dataset_type == "Emotion":
            res = add_emotion_sample(clean_label, audio_path)
            return (
                f"✅ Added emotion sample to 'datasets/emotions/': '{clean_label}'!\n"
                f"• Samples for this emotion: {res['sample_count_for_label']}\n"
                f"• Total emotion samples in KD-Tree: {res['total_samples']}\n"
                f"• Saved at: {res['saved_path']}"
            )
        else:
            return "❌ Unknown dataset target."
    except Exception as e:
        return f"❌ Failed to index sample: {e}"

def search_voice_tree(audio_path):
    if not audio_path:
        return "Please upload or record an audio sample to match."
    try:
        y, sr = librosa.load(audio_path, sr=16000)
        res = process_retrieval(audio_signal=y, sample_rate=sr, match_type="voice", top_k=5)
        if not res.get("matches"):
            return res.get("message", "No matches found. Make sure to record or add samples to the Voice KD-Tree first.")
        out = "=== Voice Match KD-Tree Results ===\n\n"
        for i, m in enumerate(res["matches"], 1):
            out += (
                f"{i}. Person: {m['label']}\n"
                f"   • Best Cosine Similarity: {m['similarity']:.4f}\n"
                f"   • Avg Similarity across recordings: {m.get('avg_similarity', m['similarity']):.4f}\n"
                f"   • Number of reference recordings in DB: {m.get('sample_count', 1)}\n\n"
            )
        return out
    except Exception as e:
        return f"Error matching voice: {e}"

def list_indexed_trees():
    song_tree = get_song_tree()
    voice_tree = get_voice_tree()
    lang_tree = get_language_tree()
    emo_tree = get_emotion_tree()
    
    out = "=== Locally Indexed Metric Trees ===\n\n"
    
    emo_stats = emo_tree.get_label_stats()
    out += f"🎭 Emotions Tree ({len(emo_tree.items)} total samples across {len(emo_stats)} emotions):\n"
    for emo, count in emo_stats.items():
        out += f"  - Emotion '{emo}': {count} sample(s)\n"
    if not emo_stats:
        out += "  (No emotions indexed yet)\n"

    out += "\n" + "-"*40 + "\n"
    voice_stats = voice_tree.get_label_stats()
    out += f"🗣️ Voices Tree ({len(voice_tree.items)} total recordings across {len(voice_stats)} people):\n"
    for person, count in voice_stats.items():
        out += f"  - Person '{person}': {count} recording(s)\n"
    if not voice_stats:
        out += "  (No voices recorded yet)\n"

    out += "\n" + "-"*40 + "\n"
    song_stats = song_tree.get_label_stats()
    out += f"🎵 Songs Tree ({len(song_tree.items)} total recordings across {len(song_stats)} titles):\n"
    for song, count in song_stats.items():
        out += f"  - Track '{song}': {count} sample(s)\n"
    if not song_stats:
        out += "  (No songs recorded yet)\n"

    out += "\n" + "-"*40 + "\n"
    lang_stats = lang_tree.get_label_stats()
    out += f"🌍 Languages Tree ({len(lang_tree.items)} total samples across {len(lang_stats)} languages):\n"
    for lang, count in lang_stats.items():
        out += f"  - '{lang}': {count} sample(s)\n"
        
    return out

# Gradio Blocks UI
with gr.Blocks(title="Voogle - Voice Search & Signal Tree Matcher") as demo:
    gr.Markdown("# 🚀 Voogle Audio Search Engine\n### Classical Signal-Processing & Local Metric KD-Tree Matching")

    with gr.Tab("🎙️ Multi-Task Audio Analyzer"):
        with gr.Row():
            with gr.Column():
                input_audio = gr.Audio(type="filepath", sources=["upload", "microphone"], label="Input Audio (Record Live or Upload)")
                task_selector = gr.CheckboxGroup(
                    ["Diarization", "Emotion", "Retrieval", "Transcription", "Language", "Music"],
                    label="Select Tasks to Run",
                    value=["Emotion", "Language", "Retrieval"]
                )
                analyze_btn = gr.Button("Analyze Audio", variant="primary")
            with gr.Column():
                analysis_output = gr.Textbox(label="Analysis Results", lines=16)
        
        analyze_btn.click(fn=analyze_audio, inputs=[input_audio, task_selector], outputs=analysis_output)

    with gr.Tab("🌲 Record & Add to Tree Datasets"):
        gr.Markdown(
            "### Add new reference audio to your local KD-Tree datasets.\n"
            "- Files are neatly organized into separate folders: `datasets/emotions/`, `datasets/voices/`, `datasets/languages/`, and `datasets/songs/`.\n"
            "- You can record **multiple audio clips for the same person or class**! The KD-Tree groups them together and matches whoever has the closest acoustic signature."
        )
        with gr.Row():
            with gr.Column():
                sample_audio = gr.Audio(type="filepath", sources=["microphone", "upload"], label="Record from Mic or Upload Sample")
                sample_label = gr.Textbox(label="Label / Person / Song / Language / Emotion", placeholder="e.g. 'Alice', 'happy', 'Spanish', 'Track A'")
                dataset_type = gr.Radio(["Voice", "Song", "Language", "Emotion"], label="Tree Index Target", value="Voice")
                add_btn = gr.Button("Save & Index into KD-Tree", variant="primary")
                add_status = gr.Textbox(label="Status / Confirmation", lines=5)
            with gr.Column():
                refresh_btn = gr.Button("Inspect Indexed Trees & Counts")
                tree_contents = gr.Textbox(label="Indexed People & Datasets", lines=15)
        
        add_btn.click(fn=add_to_tree_dataset, inputs=[sample_audio, sample_label, dataset_type], outputs=add_status)
        refresh_btn.click(fn=list_indexed_trees, inputs=[], outputs=tree_contents)

    with gr.Tab("🗣️ Dedicated Voice Matcher"):
        gr.Markdown(
            "### Test Voice Identification\n"
            "Record or upload an unknown voice to find out who it matches among your saved people in the **Voices KD-Tree**."
        )
        with gr.Row():
            with gr.Column():
                voice_query_audio = gr.Audio(type="filepath", sources=["microphone", "upload"], label="Record Voice to Identify")
                voice_search_btn = gr.Button("Identify Speaker", variant="primary")
            with gr.Column():
                voice_search_output = gr.Textbox(label="Identified Speaker(s)", lines=12)
        
        voice_search_btn.click(fn=search_voice_tree, inputs=[voice_query_audio], outputs=voice_search_output)

if __name__ == "__main__":
    print("\nStarting Voogle GUI with Tree-based search...")
    demo.launch(inbrowser=True)
