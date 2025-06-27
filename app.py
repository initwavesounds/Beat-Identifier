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

# --- Patch SciPy so librosa.beat_track can find hann() ---
import scipy.signal
from scipy.signal.windows import hann as _hann
scipy.signal.hann = _hann

# Detect device
use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')
print(f"Using device: {device}")

notes = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']

def download_from_gdrive(url):
    if "drive.google.com" not in url:
        raise ValueError("Not a valid Google Drive URL.")
    if "id=" in url:
        file_id = url.split("id=")[1].split("&")[0]
    else:
        file_id = url.split("/d/")[1].split("/")[0]
    download_url = f"https://drive.google.com/uc?export=download&id={file_id}"
    resp = requests.get(download_url, stream=True)
    if resp.status_code != 200:
        raise ValueError("Failed to download.")
    path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3").name
    with open(path, "wb") as f:
        shutil.copyfileobj(resp.raw, f)
    return path

def analyze_single(path):
    # load audio
    y, sr = librosa.load(path, sr=None)
    # optionally use GPU tensor
    if use_cuda:
        y = torch.from_numpy(y).to(device).cpu().numpy()
    # compute duration
    duration = librosa.get_duration(y=y, sr=sr)
    m, s = divmod(int(duration), 60)
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
        f"**{os.path.basename(path)}**\n\n"
        f"- Duration: {m}m{s}s ({duration:.2f}s)\n"
        f"- BPM: {bpm}\n"
        f"- Key: {main_note}\n"
        f"- Tuning offset: {tuning:.3f} semitones"
    )
    return summary, path

def analyze_batch(files, gdrive_url):
    if files:
        paths = files
    elif gdrive_url.strip():
        paths = [download_from_gdrive(gdrive_url.strip())]
    else:
        return "⚠️ No input provided.", []
    # parallel analysis using all CPU cores
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as exe:
        results = list(exe.map(analyze_single, paths))
    summaries, out_paths = zip(*results)
    return "\n\n---\n\n".join(summaries), list(out_paths)

# Gradio UI
with gr.Blocks(title="Beat Identifier", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Beat Identifier")
    gr.Markdown("Upload audio file(s) or paste a Google Drive share URL.")
    with gr.Row():
        with gr.Column():
            upload = gr.File(label="Upload Audio File(s)", file_count="multiple", file_types=["audio"])
            gdrive = gr.Textbox(label="Google Drive URL", placeholder="https://drive.google.com/...")
            btn    = gr.Button("Analyze", variant="primary")
        with gr.Column():
            out_md    = gr.Markdown("", label="Analysis Results")
            out_files = gr.File(label="Download / Play Files", file_count="multiple")
    btn.click(analyze_batch, inputs=[upload, gdrive], outputs=[out_md, out_files])

# enable queue to avoid timeouts
demo.queue(max_size=16)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
