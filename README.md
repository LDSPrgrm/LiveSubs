# LiveSubs

A specific-purpose tool for real-time Japanese audio transcription and English translation on Windows.

It captures system audio via WASAPI Loopback, processes it using `faster-whisper` (CTranslate2 implementation of OpenAI's Whisper), and displays subtitles in a transparent overlay.

## Technical Overview

*   **Audio Capture**: `pyaudiowpatch` (PortAudio with WASAPI Loopback). **Windows Only**.
*   **Inference**: `faster-whisper` running heavily quantized (int8/float16) on CUDA.
*   **Pipeline**:
    1.  **VAD**: Energy-based segmentation to detect speech.
    2.  **Transcription**: Large-v3 or Kotoba-Bilingual Whisper models.
    3.  **Romaji**: `pykakasi` conversion.
    4.  **Translation**: DeepL or Google Translate API (via `deep-translator`).
*   **Latency**: Intrinsic latency of 1-3 seconds due to chunk-based processing (required for context). Scaling to "Real-time" involves incomplete partial results.

## Prerequisites

1.  **OS**: Windows 10/11 (Strict requirement for WASAPI loopback).
2.  **GPU**: NVIDIA GPU with CUDA support.
    *   **VRAM**: Minimum 4GB for `large-v3` (Int8). 2GB may suffice for `small`/`medium`.
    *   *Note: CPU inference is technically possible but will result in significant desync (processing time > audio duration).*
3.  **Environment**: Conda (recommended for CUDA toolkit management).

## Installation

The following specific combination of packages is required to avoid CUDA version mismatches and Windows DLL errors.

```powershell
# 1. Create Environment (Python 3.11 tested)
conda create -n LiveSubs python=3.11.14 -y
conda activate LiveSubs

# 2. Install PyTorch & CUDA 12.1
# Must be installed via Conda to link system-level CUDA binaries correctly.
conda install pytorch==2.5.1 torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y

# 3. Install Python Dependencies
pip install -r requirements.txt

# 4. Windows CTranslate2 Fix
# Required to prevent "DLL load failed" errors for zlib on Windows.
conda install -c conda-forge zlib-wapi -y
```

## Usage

```powershell
conda activate LiveSubs
python live_subtitles.py
```

*   **Positioning**: Drag the window to move.
*   **Menu**: Right-click to change segmentation presets or exit.
*   **Audio Source**: The script automatically attempts to find the default WASAPI output device. Check stdout if it fails.

## Configuration

Edit `live_subtitles.py` directly to change:

*   **`WHISPER_MODEL_PATH`**: Default is `kotoba-tech/kotoba-whisper-bilingual-v1.0-faster`. Change to `large-v3` if you have specific needs.
*   **`COMPUTE_TYPE`**: Default `float16` (if GPU). Change to `int8_float16` if OOM occurs.
*   **thresholds**: `VOLUME_THRESHOLD_DB` controls VAD sensitivity.

## Known Limitations

*   **Hallucinations**: Whisper models are prone to repeating phrases ("Thank you for watching") during silence. A regex-based filter is implemented in `live_subtitles.py` to mitigate this, but valid speech may essentially be filtered if it matches common hallucination patterns.
*   **Single Stream**: Currently strictly mono-stream processing. It cannot differentiate between speakers (diarization is not implemented).
*   **App Focus**: This is a standalone script, not a packaged application. It requires an active terminal window.
