# @title Beat Identifier

import os
import tempfile
import socket
import requests
import gdown
import gradio as gr
import librosa
import numpy as np
import torch
from concurrent.futures import ThreadPoolExecutor
import scipy.signal
from scipy.signal.windows import hann as _hann

# Patch SciPy so librosa.beat_track can find hann()
scipy.signal.hann = _hann

# Detect device
use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')
print(f"Using device: {device}")

notes = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']

def download_from_url(url):
    """
    Download a single audio file from any direct URL.
    Uses Content-Type header (or URL path) to pick .mp3/.wav extension.
    Returns a list with the local filepath.
    """
    temp_dir = tempfile.mkdtemp(prefix="audio_dl_")
    resp = requests.get(url, stream=True)
    resp.raise_for_status()

    # Determine extension from Content-Type
    ct = resp.headers.get('content-type', '').lower()
    if 'mpeg' in ct or 'mp3' in ct:
        ext = '.mp3'
    elif 'wav' in ct:
        ext = '.wav'
    else:
        # fallback: strip query params and take URL path extension
        path = url.split('?',1)[0]
        ext = os.path.splitext(path)[1].lower() or '.mp3'

    local_path = os.path.join(temp_dir, f"download{ext}")
    with open(local_path, 'wb') as f:
        for chunk in resp.iter_content(chunk_size=1024*1024):
            f.write(chunk)

    return [local_path]

def download_from_gdrive(url):
    """
    Download a single MP3/WAV file or all MP3/WAV files in a Google Drive folder URL.
    Returns a list of local file paths.
    """
    temp_dir = tempfile.mkdtemp(prefix="gdrive_dl_")
    # Folder URL?
    if "/folders/" in url or ("google.com/drive" in url and "id=" in url):
        gdown.download_folder(url=url, output=temp_dir, quiet=False, use_cookies=False)
        files = []
        for root, _, fs in os.walk(temp_dir):
            for f in fs:
                if f.lower().endswith(('.mp3', '.wav')):
                    files.append(os.path.join(root, f))
        return files

    # File URL?
    if "/file/d/" in url or "open?id=" in url:
        if "/file/d/" in url:
            file_id = url.split("/file/d/")[1].split("/")[0]
        else:
            file_id = url.split("open?id=")[1].split("&")[0]
        out_path = os.path.join(temp_dir, file_id)
        gdown.download(id=file_id, output=out_path, quiet=False, use_cookies=False)
        downloaded = []
        for ext in ('.mp3', '.wav'):
            candidate = out_path + ext
            if os.path.exists(candidate):
                downloaded.append(candidate)
        return downloaded or [out_path]

    raise ValueError("Invalid Google Drive URL. Provide a file or folder link.")

def analyze_single(path):
    """Analyze a single audio file and return (summary, path)."""
    try:
        filename = os.path.basename(path)
        print(f"Analyzing: {filename}")
        y, sr = librosa.load(path, sr=None)
        if use_cuda:
            import torch as _torch
            y = _torch.from_numpy(y).to(device).cpu().numpy()
        duration = librosa.get_duration(y=y, sr=sr)
        m, s = divmod(int(duration), 60)
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        tempo, _ = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
        bpm = int(round(float(tempo)))
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
        main_note = notes[int(np.mean(chroma, axis=1).argmax())]
        tuning = librosa.estimate_tuning(y=y, sr=sr)
        summary = (
            f"**{filename}**\n\n"
            f"- Duration: {m}m {s}s ({duration:.2f} s)\n"
            f"- BPM: {bpm}\n"
            f"- Key: {main_note}\n"
            f"- Tuning offset: {tuning:.3f} semitones"
        )
        return summary, path
    except Exception as e:
        return f"Error analyzing {os.path.basename(path)}: {e}", None

def analyze_batch(paths):
    """Analyze multiple audio files given their local paths."""
    with ThreadPoolExecutor(max_workers=min(4, os.cpu_count())) as exe:
        results = list(exe.map(analyze_single, paths))
    summaries, out_paths, options = [], [], []
    for summary, path in results:
        summaries.append(summary)
        if path:
            out_paths.append(path)
            filename = os.path.basename(path)
            options.append((filename, path))
    return "\n\n---\n\n".join(summaries), out_paths, options

def find_available_port(start=7860, end=7900):
    for port in range(start, end + 1):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("", port))
                return port
            except OSError:
                continue
    return start

def process_inputs(files, url):
    """
    1) Use uploaded files if any.
    2) Else if URL points to a direct audio file → download_from_url.
    3) Else treat URL as Google Drive link → download_from_gdrive.
    4) Else prompt user.
    """
    if files:
        paths = files
    elif url and url.strip():
        url = url.strip()
        try:
            if url.lower().startswith(('http://','https://')) and any(ext in url.lower() for ext in ('.mp3','.wav')):
                paths = download_from_url(url)
            else:
                paths = download_from_gdrive(url)
            if not paths:
                return "No MP3/WAV files found at that URL.", [], []
        except Exception as e:
            return f"Download error: {e}", [], []
    else:
        return "Please upload audio files or provide a direct or Drive URL.", [], []

    # No more extension-based filtering: analyze whatever got downloaded
    summary, out_paths, options = analyze_batch(paths)
    default = out_paths[0] if out_paths else None
    return (
        summary,
        out_paths,
        gr.update(choices=options, value=default)
    )

# Build Gradio interface
with gr.Blocks(title="Beat Identifier", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Beat Identifier")
    gr.Markdown("Analyze BPM, key, and tuning of audio files")
    with gr.Row():
        with gr.Column():
            upload = gr.File(
                label="Upload Audio Files (MP3/WAV)",
                file_count="multiple",
                file_types=[".mp3", ".wav"],
                type="filepath"
            )
            url = gr.Textbox(
                label="Direct Audio URL or Google Drive URL",
                placeholder="https://..."
            )
            btn = gr.Button("Analyze Audio", variant="primary")
        with gr.Column():
            out_md = gr.Markdown("", label="Analysis Summary")
            selector = gr.Dropdown(
                label="Select File to Play",
                choices=[],
                interactive=True,
                type="value"
            )
            player = gr.Audio(label="Preview", interactive=True)
            out_files = gr.File(
                label="Download Analyzed Files",
                file_count="multiple"
            )

    selector.change(lambda p: p or None, inputs=selector, outputs=player)
    btn.click(
        fn=process_inputs,
        inputs=[upload, url],
        outputs=[out_md, out_files, selector]
    )

# Launch server
port = find_available_port()
print(f"Starting server on port {port}")
demo.launch(server_name="0.0.0.0", server_port=port)
