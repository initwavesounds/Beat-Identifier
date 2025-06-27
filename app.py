# @title Beat Identifier

import os
import gradio as gr
import librosa
import numpy as np
import tempfile
import requests
import shutil
import torch
from concurrent.futures import ThreadPoolExecutor

# Detect device
use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')
print(f"Using device: {device}")  # logged on startup

notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

def download_from_gdrive(gdrive_url):
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

def analyze_single(path):
    # load audio
    y, sr = librosa.load(path, sr=None)
    # optionally move data to GPU tensor (for future GPU-based ops)
    if use_cuda:
        y_tensor = torch.from_numpy(y).to(device)
        y = y_tensor.cpu().numpy()  # librosa still expects numpy
    # compute duration
    duration = librosa.get_duration(y=y, sr=sr)
    m, s = divmod(int(duration), 60)
    duration_str = f"{m} min {s} sec ({duration:.2f}s)"
    # BPM
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    tempo, _ = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
    bpm = int(round(float(tempo)))
    # key detection
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    main_note = notes[int(np.mean(chroma, axis=1).argmax())]
    # tuning
    tuning = librosa.estimate_tuning(y=y, sr=sr)
    summary = (
        f"**File:** {os.path.basename(path)}\n\n"
        f"- **Duration:** {duration_str}\n"
        f"- **Estimated BPM:** {bpm} BPM\n"
        f"- **Main Key Note:** {main_note}\n"
        f"- **Tuning Offset:** {tuning:.3f} semitones"
    )
    return summary, path

def analyze_batch(files, gdrive_url):
    # gather paths
    if files:
        paths = files
    elif gdrive_url.strip():
        paths = [download_from_gdrive(gdrive_url.strip())]
    else:
        return "⚠️ No input provided.", []
    # parallel analysis using all CPU cores
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as exe:
        futures = [exe.submit(analyze_single, p) for p in paths]
        results = [f.result() for f in futures]
    summaries, out_paths = zip(*results)
    return "\n\n---\n\n".join(summaries), list(out_paths)

# Gradio UI
with gr.Blocks(title="Beat Identifier", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Beat Identifier")
    gr.Markdown("Upload one or more audio files **or** paste a Google Drive share URL.")
    with gr.Row():
        with gr.Column():
            file_in    = gr.File(label="Upload Audio File(s)", file_count="multiple", file_types=["audio"])
            gdrive_in  = gr.Textbox(label="Google Drive URL", placeholder="https://drive.google.com/...")
            run_btn    = gr.Button("Analyze", variant="primary")
        with gr.Column():
            output_md  = gr.Markdown("", label="Analysis Results")
            select_btn = gr.Dropdown(choices=[], label="Select File to Play")
            player     = gr.Audio(label="Play Audio")
    run_btn.click(analyze_batch, inputs=[file_in, gdrive_in], outputs=[output_md, select_btn])
    select_btn.change(lambda x: x, inputs=select_btn, outputs=player)

# queue prevents timeouts; concurrency matches CPU or GPU availability
demo.queue(max_size=16, concurrency_count=os.cpu_count())

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
