# Beat Identifier

Try it out: [Beat Identifier GUI](https://officialcyber88-beat-identifier.hf.space)

# Create the README.md file with the provided content and structure

readme_content = """# Beat Identifier GUI

A Gradio-based web application to analyze audio files (MP3/WAV) and extract:
- **Duration**  
- **Estimated BPM (tempo)**  
- **Key**  
- **Tuning offset**  
- **In-browser playback**  

---

## Features

- **Batch Analysis**: Drag & drop multiple audio files for simultaneous processing.  
- **Tempo Detection**: Uses `librosa` for beat tracking and BPM estimation.  
- **Key Detection**: Leverages chroma features to identify the main musical note.  
- **Tuning Offset**: Estimates deviation from standard tuning in semitones.  
- **Playback**: Play any processed file directly in your browser.  
- **CUDA Support**: Utilizes GPU acceleration when available.

---

## Installation

### System Dependencies

On Linux (including Google Colab), install the system library required by `soundfile`:

```bash
sudo apt-get update && sudo apt-get install -y libsndfile1
```

> ⚠️ This step is **not** included in `requirements.txt`.

### Python Dependencies

Create a `requirements.txt` with the following content:

```txt
soundfile
gradio==3.44.0
librosa==0.10.0
numpy==1.24.3
requests==2.31.0
gdown==4.7.1
torch==2.1.0
```

Then install with:

```bash
pip install -r requirements.txt
```

Or install directly:

```bash
pip install \
  soundfile \
  gradio==3.44.0 \
  librosa==0.10.0 \
  numpy==1.24.3 \
  requests==2.31.0 \
  gdown==4.7.1 \
  torch==2.1.0
```

---

## Usage

1. **Clone the repository** (or download the script file).  
2. **Run the app**:

   ```bash
   python app.py
   ```

   The script will automatically find an available port between 7860–7900 and launch the Gradio interface.  
   - To share in Google Colab, it enables `share=True`.  
   - In local/production, access the UI at `http://0.0.0.0:<port>`.

3. **Analyze Audio**:
   - Drag & drop or select your MP3/WAV files.  
   - Click **"Analyze Audio"**.  
   - View the summary, download processed files, or playback.

---

## Optional Setup Scripts

- **setup.sh**: Combine system and Python installs:

  ```bash
  #!/usr/bin/env bash
  sudo apt-get update && sudo apt-get install -y libsndfile1
  pip install -r requirements.txt
  ```

- **Dockerfile**: For containerized deployment, include both apt-get and pip steps.

---

## License

This project is released under the **Unlicense**. See [https://unlicense.org](https://unlicense.org) for details.
"""

# Write the content to README.md
path = "/mnt/data/README.md"
with open(path, "w") as f:
    f.write(readme_content)

path
