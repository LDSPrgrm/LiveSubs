<p align="center">
  <h1 align="center">🎙️ LiveSubs</h1>
</p>

<p align="center">
  <strong>Real-Time AI Subtitles for Windows</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Platform-Windows%2010%2F11-blue?style=for-the-badge&logo=windows&logoColor=white" alt="Windows Only" />
  <img src="https://img.shields.io/badge/AI-Faster--Whisper-green?style=for-the-badge&logo=nvidia&logoColor=white" alt="Faster Whisper" />
  <img src="https://img.shields.io/badge/Acceleration-CUDA%2012.1-76b900?style=for-the-badge&logo=nvidia&logoColor=white" alt="CUDA" />
  <img src="https://img.shields.io/badge/Python-3.11-3776ab?style=for-the-badge&logo=python&logoColor=white" alt="Python 3.11" />
</p>

<br>

## ✨ Overview

**LiveSubs** is a high-performance tool developed to bridge language barriers in real-time. Whether you are watching a raw Japanese stream, gaming, or attending a meeting, LiveSubs captures your **System Audio** directly and overlays cinema-quality subtitles on top of your screen.

It leverages the power of **Faster-Whisper** and **CTranslate2** to deliver transcription speeds up to **4x faster** than real-time on consumer GPUs, ensuring you never miss a beat.

---

## ⚡ Core Features

| Feature | Description |
|:---|:---|
| **🎧 Loopback Capture** | Captures audio directly from your speakers (WASAPI) — no virtual cables required. |
| **🚀 Hyper-Fast AI** | Powered by `faster-whisper` (Large-v3 / Kotoba) running on CUDA Int8 quantization. |
| **🈯 Dual-Display** | Shows the original **Japanese**, **Romaji** pronunciation, and **English** translation simultaneously. |
| **🧠 Smart Context** | Uses dynamic audio chunking with overlap to prevent cutting off words mid-sentence. |
| **👻 Anti-Hallucination** | Built-in filters to detect and remove common Whisper phantom phrases (e.g., "Thank you for watching"). |
| **🖼️ Glass Overlay** | A click-through, transparent UI that floats unobtrusively over your content. |

---

## 🛠️ System Requirements

Before you begin, ensure your "Command Center" is ready.

*   **Operating System**: Windows 10 or 11 (Strict requirement for WASAPI Loopback).
*   **GPU**: NVIDIA GeForce GTX 10-Series or newer.
    *   *Minimum*: 4GB VRAM (for Int8 quantization).
    *   *Recommended*: 6GB+ VRAM (for Large-v3 models).
*   **Software**: Conda (Miniconda/Anaconda) for environment management.

---

## 📦 Installation Guide

We have optimized the setup to strictly control library versions, as CUDA and Windows DLLs can be temperamental.

### 1. Initialize Environment
Create a clean room for the AI to breathe.
```powershell
conda create -n LiveSubs python=3.11.14 -y
conda activate LiveSubs
```

### 2. Install The Engine (Critical)
We must install PyTorch and CUDA 12.1 together to ensure they link correctly.
```powershell
conda install pytorch==2.5.1 torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
```

### 3. Install Dependencies
Install the audio wrappers and translation logic.
```powershell
pip install -r requirements.txt
```

### 4. Apply Windows Fix
This allows CTranslate2 to load the necessary zlib binaries on Windows systems.
```powershell
conda install -c conda-forge zlib-wapi -y
```

---

## 🎮 Controls & Usage

### Start the Engine
```powershell
conda activate LiveSubs
python live_subtitles.py
```

### Interaction Map
| Action | Effect |
|:---:|:---|
| **Drag Window** | Click and hold anywhere on the black background to reposition the subtitles. |
| **Right-Click** | Opens the **Context Menu** to change modes or exit. |
| **Speech Modes** | Switch between **Lyric** (High quality/cohesion) and **Stream** (Low latency). |

---

## ⚙️ Configuration

You can fine-tune the engine by editing variables at the top of `live_subtitles.py`.

| Variable | Default (Example) | Purpose |
|:---|:---|:---|
| `WHISPER_MODEL_PATH` | `kotoba-tech/...` | The specific AI model to load. Try `large-v3` for max accuracy. |
| `VOLUME_THRESHOLD_DB` | `-45.0` | Sensitivity gate. Lower this if the audio is very quiet. |
| `COMPUTE_TYPE` | `float16` | Change to `int8_float16` if you are running out of VRAM. |

---

## ❓ Troubleshooting

**Q: "The window opens but no text appears."**
> **A:** Check the console. It prints which audio device it hooked into (e.g., `[AUDIO] Source: Speakers`). If silence persists, play louder audio to trigger the gate.

**Q: "DLL load failed while importing ctranslate2"**
> **A:** You likely skipped Step 4. Run `conda install -c conda-forge zlib-wapi -y`.

**Q: "It's lagging behind the audio."**
> **A:** This is intrinsic generic latency (1-3s) required to gather enough context for a translation. For faster (but less accurate) results, right-click and select **"Fast (Gaming)"** mode.

---

<p align="center">
  <strong>Break The Language Barrier.</strong>
</p>
