# @title Beat Identifier (Soft Theme)

import gradio as gr
import librosa
import numpy as np
import tempfile
import os
import requests
import shutil

notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

def download_from_gdrive(gdrive_url):
    """
    Converts a Google Drive share URL to a direct download link and downloads the file.
    """
    if "drive.google.com" not in gdrive_url:
        raise ValueError("Not a valid Google Drive URL.")
    
    file_id = None
    if "id=" in gdrive_url:
        file_id = gdrive_url.split("id=")[1].split("&")[0]
    elif "/d/" in gdrive_url:
        file_id = gdrive_url.split("/d/")[1].split("/")[0]
    
    if not file_id:
        raise ValueError("Couldn't extract file ID from the URL.")
    
    download_url = f"https://drive.google.com/uc?export=download&id={file_id}"
    response = requests.get(download_url, stream=True)

    if response.status_code != 200:
        raise ValueError("Failed to download the file from Google Drive.")
    
    temp_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3").name
    with open(temp_path, 'wb') as f:
        shutil.copyfileobj(response.raw, f)

    return temp_path

def analyze_audio(audio_path, gdrive_url):
    # Determine input source
    if audio_path:
        tmp_path = audio_path
    elif gdrive_url.strip():
        tmp_path = download_from_gdrive(gdrive_url.strip())
    else:
        return "⚠️ No input provided.", None

    # Load audio
    try:
        y, sr = librosa.load(tmp_path, sr=None)
    except Exception as e:
        return f"❌ Error loading audio: {e}", None

    # Duration
    duration = librosa.get_duration(y=y, sr=sr)
    minutes, seconds = divmod(int(duration), 60)
    duration_str = f"{minutes} min {seconds} sec ({duration:.2f}s)"

    # BPM detection
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    tempo, _ = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
    rounded_bpm = int(round(float(tempo)))

    # Main key note
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    chroma_mean = np.mean(chroma, axis=1)
    main_note = notes[int(chroma_mean.argmax())]

    # Tuning offset
    tuning = librosa.estimate_tuning(y=y, sr=sr)

    # Summary
    summary = (
        f"  **Audio Analysis**\n\n"
        f"- **Duration:** {duration_str}\n"
        f"- **Song BPM:** {rounded_bpm} BPM\n"
        f"- **Main Key Note:** {main_note}\n"
        f"- **Tuning Offset:** {tuning:.3f} semitones"
    )
    return summary, tmp_path

# === Gradio UI with Soft theme ===
with gr.Blocks(title=" Beat Identifier", theme=gr.themes.Soft()) as demo:
    gr.Markdown("#  Beat Identifier")
    gr.Markdown("Upload an audio file **or** paste a Google Drive share URL.")

    with gr.Row():
        with gr.Column():
            audio_in     = gr.Audio(type="filepath", label="Upload Audio File")
            gdrive_in    = gr.Textbox(label="Google Drive URL", placeholder="https://drive.google.com/...")
            analyze_btn  = gr.Button("Analyze", variant="primary")
        with gr.Column():
            result_out   = gr.Markdown("", label="Analysis Results")
            player_out   = gr.Audio(label="Play Audio")

    analyze_btn.click(
        fn=analyze_audio,
        inputs=[audio_in, gdrive_in],
        outputs=[result_out, player_out]
    )

if __name__ == "__main__":
    demo.launch()
