# Beat Identifier

Try it out: [Beat Identifier GUI](https://officialcyber88-beat-identifier.hf.space)

A simple Gradio-based web app that analyzes an audio file (or Google Drive link) and reports:
- **Duration**  
- **Estimated BPM (tempo)**  
- **Main key note**  
- **Tuning offset**  

It also lets you play back the uploaded or downloaded file in-browser.

---

## Features

- **Upload or Link**: Accepts local audio files (wav, mp3, etc.) or a Google Drive share URL.  
- **Tempo Detection**: Uses `librosa` to estimate BPM.  
- **Key Detection**: Computes chroma features to find the main musical note.  
- **Tuning Estimate**: Reports tuning deviation in semitones.  
- **Web UI**: Interactive interface built with Gradio’s “Soft” theme.  
- **In-browser Playback**: Play the processed audio directly in the browser.

---

## Dependencies

Make sure you have Python ≥ 3.7 installed. Then install the following:

```bash
pip install gradio==3.44.0 librosa==0.10.0 numpy==1.24.3 soundfile==0.12.1 requests==2.31.0 matplotlib==3.7.1
```

**Note:**
- `ffmpeg` must be installed on your system and available in your `PATH` for some audio formats.
- If you plan to process very large files or many files, ensure you have sufficient RAM.

---

## Installation

1. Clone this repository (or copy the script).
2. Create and activate a virtual environment (optional, but recommended):

```bash
python3 -m venv venv
source venv/bin/activate
```

3. Install the required packages:

```bash
pip install -r requirements.txt
```

---

## Usage

Run the app:

```bash
python app.py
```

You’ll see output such as:

```
Running on local URL:  http://127.0.0.1:7860
```

1. Open that URL in your browser.  
2. Upload an audio file: Drag-and-drop or browse for a local `.wav`, `.mp3`, etc.  
3. Or paste a Google Drive share link.  
4. Click **Analyze**.  
5. View the **Analysis Results** (duration, BPM, key, tuning) and play back your audio.

---

## requirements.txt

```
gradio==3.44.0
librosa==0.10.0
numpy==1.24.3
soundfile==0.12.1
requests==2.31.0
matplotlib==3.7.1
```
