import numpy as np
import queue
import threading
import time
import tkinter as tk
import sys
import torch
import ctypes
try:
    ctypes.windll.shcore.SetProcessDpiAwareness(1) # DPI Aware
except:
    try: ctypes.windll.user32.SetProcessDPIAware()
    except: pass
from faster_whisper import WhisperModel
import scipy.signal
import pyaudiowpatch as pyaudio
import difflib
import re
import pykakasi
from deep_translator import GoogleTranslator

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------
# AI Settings
WHISPER_MODEL_PATH = "kotoba-tech/kotoba-whisper-bilingual-v1.0-faster"

COMPUTE_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
COMPUTE_TYPE = "float16" if torch.cuda.is_available() else "int8"

# Audio Settings
SAMPLE_RATE = 16000

# VAD / Segmentation (Dynamic)
VOLUME_THRESHOLD_DB = -45.0
MIN_SEGMENT_DURATION_MS = 200  
MAX_SEGMENT_DURATION_MS = 13000 # Default: Lyric mode
PAUSE_THRESHOLD_MS = 1000       # Default: Lyric mode

# Hybrid Decoding Strategy
LIVE_CHUNK_DURATION_MS = 2500   # Fast Partial Mode: 2.5s chunks
BEAM_SIZE_FINAL = 10            # Default: Lyric mode quality
BEAM_SIZE_LIVE = 1              # Greedy for live (Interim Results)
CONDITION_ON_PREV_TEXT = True   # Only for final

# Overlap Optimization
CHUNK_OVERLAP_MS = 1000          # 1.0s base overlap (Silent break)
ADAPTIVE_OVERLAP_MS = 2000      # 2.0s adaptive overlap (Max duration cut)

# Display Settings (Readability)
MIN_DISPLAY_DURATION_MS = 1500  # Minimum time a subtitle stays visible
MAX_DISPLAY_DURATION_MS = 2500  # Maximum time to hold a subtitle
CHAR_DISPLAY_RATE_MS = 40       # Extra time per character of English text
MAX_CATCHUP_BACKLOG = 3         # If queue > this, skip the wait to catch up


# Presets
SPEECH_PRESETS = {
    "Lyric/Song (Cohesion)": {
        "MAX_SEG": 13000, "PAUSE": 2000, "ADAPTIVE": 2000, "BEAM": 10,
        "TIP": "Prioritizes lyric cohesion. High context and beam search for beauty."
    },
    "Slow (Chill/Documentary)": {
        "MAX_SEG": 10000, "PAUSE": 600, "ADAPTIVE": 1000, "BEAM": 5,
        "TIP": "Longer context (10s), lenient pauses (0.6s). Best for clear, slow talk."
    },
    "Medium (Natural/Stream)": {
        "MAX_SEG": 7000,  "PAUSE": 300, "ADAPTIVE": 2000, "BEAM": 4,
        "TIP": "Balanced (7s context, 0.3s pause). Standard for most streams."
    },
    "Fast (Gaming/Heated)": {
        "MAX_SEG": 4000,  "PAUSE": 200, "ADAPTIVE": 3000, "BEAM": 3,
        "TIP": "Aggressive (4s context, 0.2s pause), high overlap. Best for rapid speech."
    },
}

# Noise / Hallucination Filter
HALLUCINATION_PHRASES = [
    "ご視聴ありがとうございました",
    "Thank you for watching",
    "I'm not sure what you mean by that",
    "It seems like the temperature will rise steadily from morning in central Tokyo.",
    "It seems like the temperature will rise steadily from the morning in central Tokyo.",
    "It seems like the temperature will steadily rise from the morning in central Tokyo.",
    "It seems like the temperature will rise steadily from the morning in the heart of Tokyo.",
    "It seems like the temperature will steadily rise from the morning in the heart of Tokyo.",
    "I'm sorry, but it seems like the temperature will steadily rise from the morning in central Tokyo.",
    "I'm sorry, but it seems like the temperature will rise steadily from the morning in central Tokyo.",
    "Lately, it seems like the temperature will rise steadily from the morning in central Tokyo.",
    "The temperature is expected to rise steadily from the morning in central Tokyo.",
    "In central Tokyo, the temperature is likely to rise rapidly from the morning."
]

def is_hallucination_text(text):
    """
    Advanced filter for hallucinations using fuzzy matching and keyword density.
    """
    text = text.strip()
    if not text: return True
    
    # Clean text for matching
    clean_text = re.sub(r'[^\w\s]', '', text.lower())
    words = set(clean_text.split())
    
    # 1. Exact or Substring match
    for p in HALLUCINATION_PHRASES:
        p_low = p.lower()
        if p_low in text.lower():
            return True
            
    # 2. Fuzzy similarity match (catches variations)
    for p in HALLUCINATION_PHRASES:
        if len(text) > 15 and len(p) > 15:
            # Check similarity ratio
            ratio = difflib.SequenceMatcher(None, text.lower(), p.lower()).ratio()
            if ratio > 0.75: # 75% similarity is a high signal for hallucination
                return True
                
    # 3. Keyword Density (The 'Tokyo Weather' Sink)
    # Most hallucinations in kotoba-whisper/whisper mention these words in combination
    tokyo_keywords = {"temperature", "tokyo", "rise", "steadily", "morning", "central", "heart"}
    match_count = len(tokyo_keywords.intersection(words))
    if match_count >= 3: # If 3 or more keywords from the set appear, it's likely a hallucination
        return True
        
    return False
# -----------------------------------------------------------------------------
# LOGGING (UTF-8)
# -----------------------------------------------------------------------------
class Logger(object):
    def __init__(self):
        self.terminal = sys.stdout
        self.log = open("debug.log", "w", encoding="utf-8")
        if sys.platform == "win32":
            sys.stdout.reconfigure(encoding='utf-8')

    def write(self, message):
        try:
            self.terminal.write(message)
        except Exception: pass
        self.log.write(message)
        self.log.flush()

    def flush(self):
        try:
            self.terminal.flush()
        except: pass
        self.log.flush()

sys.stdout = Logger()
sys.stderr = sys.stdout

# -----------------------------------------------------------------------------
# GLOBAL STATE
# -----------------------------------------------------------------------------
raw_queue = queue.Queue()
live_queue = queue.Queue()  # For fast partial updates
final_queue = queue.Queue() # For high-quality final updates

current_live_text = ""      # Live (Growing)
current_ja_text = ""        # Final (Stable Japanese)
current_romaji_text = ""    # Final (Stable Romaji)
current_trans_text = ""     # Final (Stable English)
current_speech_duration_ms = 0  # Total duration of current chunk
current_silence_duration_ms = 0 # Current silence trail
last_final_update_time = 0      # Timestamp of last final text commitment
running = True
CURRENT_MODE = "Lyric/Song (Cohesion)"

# Initialize Romaji Converter
kks = pykakasi.kakasi()


# Initialize Translator
translator = GoogleTranslator(source='ja', target='en')

# -----------------------------------------------------------------------------
# SEGMENTATION & PREPROCESSING LAYER
# -----------------------------------------------------------------------------
def segmentation_thread():
    global current_speech_duration_ms, current_silence_duration_ms
    print(f"[SEGMENTER] Started. Dynamic Chunking Active.\n")
    segment_buffer = np.array([], dtype=np.float32)
    is_speech_active = False
    silence_duration_ms = 0
    speech_duration_ms = 0
    current_speech_duration_ms = 0
    last_live_update = 0
    
    while running:
        try:
            chunk, rate = raw_queue.get(timeout=1.0)
        except queue.Empty:
            continue

        if rate == 48000: chunk_16k = chunk[::3]
        else:
            samps = int(len(chunk) * 16000 / rate)
            chunk_16k = scipy.signal.resample(chunk, samps).astype(np.float32)

        rms = np.sqrt(np.mean(chunk_16k**2) + 1e-9)
        db = 20 * np.log10(rms)
        chunk_ms = (len(chunk_16k) / 16000) * 1000
        now = time.time()
        
        if db > VOLUME_THRESHOLD_DB:
             if not is_speech_active:
                 is_speech_active = True
             segment_buffer = np.concatenate((segment_buffer, chunk_16k))
             speech_duration_ms += chunk_ms
             silence_duration_ms = 0
        else:
             if is_speech_active:
                 segment_buffer = np.concatenate((segment_buffer, chunk_16k))
                 silence_duration_ms += chunk_ms
                 speech_duration_ms += chunk_ms
                 if silence_duration_ms > PAUSE_THRESHOLD_MS:
                     if speech_duration_ms > MIN_SEGMENT_DURATION_MS:
                         final_queue.put(segment_buffer.copy())
                         live_queue.put("RESET") 
                         
                         # OVERLAP: Keep small 0.5s overlap (Normal Pause)
                         overlap_samples = int((CHUNK_OVERLAP_MS / 1000.0) * SAMPLE_RATE)
                         if len(segment_buffer) > overlap_samples:
                             segment_buffer = segment_buffer[-overlap_samples:]
                             speech_duration_ms = (len(segment_buffer) / 16000) * 1000
                         
                         is_speech_active = False
                         silence_duration_ms = 0
                     elif speech_duration_ms < 200: 
                         segment_buffer = np.array([], dtype=np.float32)
                         speech_duration_ms = 0
                         is_speech_active = False
                         silence_duration_ms = 0
             else:
                 silence_duration_ms = 0 # Not in speech, no silence timer
                 speech_duration_ms = 0 # Not in speech, no speech duration

        if speech_duration_ms > MAX_SEGMENT_DURATION_MS:
             final_queue.put(segment_buffer.copy())
             live_queue.put("RESET")
             
             # ADAPTIVE OVERLAP: Keep 2.0s overlap (Max Duration cut assumes fast speech)
             overlap_samples = int((ADAPTIVE_OVERLAP_MS / 1000.0) * SAMPLE_RATE)
             if len(segment_buffer) > overlap_samples:
                 segment_buffer = segment_buffer[-overlap_samples:]
                 speech_duration_ms = (len(segment_buffer) / 16000) * 1000
             
             silence_duration_ms = 0
             is_speech_active = False
             current_speech_duration_ms = speech_duration_ms
             current_silence_duration_ms = 0

        if is_speech_active and speech_duration_ms > 300:
            if (now - last_live_update) * 1000 > 300: # Throttle to 300ms to save GPU for Final
                 # Fast Partial Mode: 2.5s window for live inference
                 live_audio = segment_buffer.copy()
                 max_samples = int((LIVE_CHUNK_DURATION_MS / 1000.0) * SAMPLE_RATE)
                 if len(live_audio) > max_samples:
                     live_audio = live_audio[-max_samples:]
                 
                 live_queue.put(live_audio)
                 last_live_update = now
        
        current_speech_duration_ms = speech_duration_ms
        current_silence_duration_ms = silence_duration_ms

# -----------------------------------------------------------------------------
# WHISPER TRANSCRIPTION LAYER (MULTI-THREADED)
# -----------------------------------------------------------------------------
def live_inference_thread():
    """Greedy, fast updates for 'live' text."""
    global current_live_text
    while running:
        try:
            task = live_queue.get(timeout=1.0)
            if isinstance(task, str) and task == "RESET":
                current_live_text = ""
                continue
            
            # Greedy decoding for Japanese (Matches final language)
            segments, _ = model.transcribe(
                task, language="ja", task="transcribe", beam_size=BEAM_SIZE_LIVE, 
                condition_on_previous_text=False, vad_filter=False
            )
            
            text_list = []
            for s in segments:
                if s.avg_logprob < -4.0: continue
                if is_hallucination_text(s.text): continue
                text_list.append(s.text)
                
            raw_text = "".join(text_list).strip()
            
            if raw_text:
                current_live_text = raw_text
                
        except queue.Empty: continue
        except Exception as e: print(f"[Live Error] {e}")

def final_inference_thread():
    """Beam search, high-quality updates for finalized text."""
    global current_trans_text, current_ja_text, current_romaji_text
    last_committed_text = ""
    
    while running:
        try:
            audio = final_queue.get(timeout=1.0)
            
            # SINGLE PASS: Transcribe to Japanese only
            segments_ja, _ = model.transcribe(
                audio, language="ja", task="transcribe", beam_size=BEAM_SIZE_FINAL, 
                condition_on_previous_text=CONDITION_ON_PREV_TEXT, vad_filter=False
            )
            
            ja_text_list = []
            for s in segments_ja:
                if s.avg_logprob < -4.0: continue
                if is_hallucination_text(s.text): continue
                ja_text_list.append(s.text)
            
            raw_ja = "".join(ja_text_list).strip()
            if not raw_ja: continue

            # Step 2: Convert to Romaji (Derived from JA)
            result = kks.convert(raw_ja)
            romaji = " ".join([item['hepburn'] for item in result])

            # Step 3: Translate to English (Derived from JA - Zero discrepancy)
            try:
                raw_en = translator.translate(raw_ja)
            except Exception as te:
                print(f"[Translation Error] {te}")
                raw_en = ""

            if not raw_en: continue
            if raw_en == last_committed_text: continue
            
            # --- Readability Hold Logic ---
            global last_final_update_time
            now = time.time()
            elapsed_ms = (now - last_final_update_time) * 1000
            
            # Calculate dynamic duration based on English text length (Capped at Max)
            required_wait = min(MAX_DISPLAY_DURATION_MS, MIN_DISPLAY_DURATION_MS + (len(raw_en) * CHAR_DISPLAY_RATE_MS))
            
            # Catch-up mechanism: If we are falling behind, reduce wait
            backlog = final_queue.qsize()
            if backlog >= MAX_CATCHUP_BACKLOG:
                required_wait = 0 # Forced catch-up
            elif backlog > 0:
                required_wait *= 0.5 # Fast-tracked display
                
            if elapsed_ms < required_wait:
                time.sleep((required_wait - elapsed_ms) / 1000.0)

            # Update globals
            last_final_update_time = time.time()
            current_ja_text = raw_ja
            current_romaji_text = romaji
            current_trans_text = raw_en
            
            print(f"JA: {raw_ja}")
            print(f"RO: {romaji}")
            print(f"EN: {raw_en}")
            print()
            last_committed_text = raw_en
            
        except queue.Empty: continue
        except Exception as e: print(f"[Final Error] {e}")





# -----------------------------------------------------------------------------
# GUI
# -----------------------------------------------------------------------------
class SubtitleApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Deep Translate")
        self.root.attributes("-topmost", True)
        self.root.overrideredirect(True)
        self.root.geometry(f"{root.winfo_screenwidth()}x250+0+{root.winfo_screenheight()-230}")
        self.root.configure(bg="#000001")
        self.root.attributes("-transparentcolor", "#000001")
        self.root.attributes("-alpha", 0.85) 
        
        # Status Message Logic
        self.status_text = ""
        self.status_color = "#FFFFFF"
        self.status_until = 0
        
        # 1. Top Layer: Live / Final JA Text (Gray)
        self.lbl_jp = tk.Label(root, text="Waiting...", font=("Georgia", 16), fg="#DDDDDD", bg="black", wraplength=1200, justify="center")
        self.lbl_jp.pack(anchor="s", pady=(10, 0))
        
        # 2. Middle Layer: Romaji (Yellow, Italic)
        self.lbl_ro = tk.Label(root, text="", font=("Georgia", 18, "italic"), fg="#FFFF00", bg="black", wraplength=1200, justify="center")
        self.lbl_ro.pack(anchor="s", pady=(2, 0))

        # 3. Bottom Layer: Final Translation (Green, Large Serif)
        self.lbl_en = tk.Label(root, text="", font=("Georgia", 26, "bold"), fg="#00FF00", bg="black", wraplength=1200, justify="center")
        self.lbl_en.pack(anchor="s", pady=(5, 30))
        
        # 4. Context Timer (Progress Bar - Centered & Subtle)
        self.canvas_progress = tk.Canvas(root, height=2, bg="black", highlightthickness=0)
        self.canvas_progress.pack(pady=(0, 10)) # Centered and thin
        self.bg_progress = self.canvas_progress.create_rectangle(0, 0, 300, 2, fill="#222222", outline="")
        self.rect_progress = self.canvas_progress.create_rectangle(0, 0, 0, 2, fill="#00FF00", outline="")
        self.rect_pause = self.canvas_progress.create_rectangle(0, 0, 0, 2, fill="#FFFF00", outline="")
        
        # Events
        self.root.bind("<Button-1>", self.move_start)
        self.root.bind("<B1-Motion>", self.move_active)
        self.root.bind("<Button-3>", self.show_context_menu) # Right Click
        
        # Context Menu
        self.menu = tk.Menu(root, tearoff=0)
        
        # 1. Speech Mode Cascade
        self.mode_menu = tk.Menu(self.menu, tearoff=0)
        self.selected_mode = tk.StringVar(value=CURRENT_MODE)
        self.menu.add_cascade(label="Speech Mode", menu=self.mode_menu)
        for mode in SPEECH_PRESETS.keys():
            self.mode_menu.add_radiobutton(label=mode, variable=self.selected_mode, value=mode, command=lambda m=mode: self.apply_preset(m))
        

        self.menu.add_separator()
        self.menu.add_command(label="Exit", command=self.close_app)
        
        self.update_loop()

    def apply_preset(self, mode):
        global MAX_SEGMENT_DURATION_MS, PAUSE_THRESHOLD_MS, ADAPTIVE_OVERLAP_MS, BEAM_SIZE_FINAL, CURRENT_MODE
        p = SPEECH_PRESETS[mode]
        CURRENT_MODE = mode
        self.selected_mode.set(mode)
        MAX_SEGMENT_DURATION_MS = p["MAX_SEG"]
        PAUSE_THRESHOLD_MS = p["PAUSE"]
        ADAPTIVE_OVERLAP_MS = p["ADAPTIVE"]
        BEAM_SIZE_FINAL = p["BEAM"]
        
        # Visual Tooltip / Feedback
        self.show_status(f"Mode: {mode}", "#FFFF00", duration=3)
        print(f"[CONFIG] Applied {mode}: {p}\n")


    def show_status(self, text, color, duration=3):
        self.status_text = text
        self.status_color = color
        self.status_until = time.time() + duration

        
    def show_context_menu(self, e):
        self.menu.post(e.x_root, e.y_root)

    def close_app(self):
        global running
        running = False
        self.root.destroy()
        
    def move_start(self, e):
        self.x, self.y = e.x, e.y
        
    def move_active(self, e):
        new_x = self.root.winfo_x() + (e.x - self.x)
        new_y = self.root.winfo_y() + (e.y - self.y)
        self.root.geometry(f"+{new_x}+{new_y}")
        
    def update_loop(self):
        global current_trans_text, current_live_text, current_romaji_text, current_ja_text, CURRENT_MODE
        
        # 1. Determine Main Text
        if CURRENT_MODE == "Lyric/Song (Cohesion)":
            top_display = current_ja_text
        else:
            top_display = current_ja_text if current_ja_text else current_live_text
        
        # 2. Status Override Logic
        now = time.time()
        if now < self.status_until:
            display_text = f"[{self.status_text}] {top_display}" if top_display else f"[{self.status_text}]"
            self.lbl_jp.config(text=display_text, fg=self.status_color)
        else:
            self.lbl_jp.config(text=top_display, fg="#DDDDDD")

        self.lbl_ro.config(text=current_romaji_text)
        self.lbl_en.config(text=current_trans_text)
        
        # Update Progress Bar (Centered status bar)
        BAR_WIDTH = 300
        self.canvas_progress.config(width=BAR_WIDTH)
        
        # 1. Main Speech Duration Progress (Green)
        # base_speech_ms stays stable while pausing so the green bar doesn't "jitter" or grow
        base_speech_ms = max(0, current_speech_duration_ms - current_silence_duration_ms)
        speech_progress = min(base_speech_ms / MAX_SEGMENT_DURATION_MS, 1.0)
        base_x = int(speech_progress * BAR_WIDTH)
        self.canvas_progress.coords(self.rect_progress, 0, 0, base_x, 2)
        
        # 2. Pause Timer Progress (Wall closing in from the RIGHT)
        if current_silence_duration_ms > 0:
            pause_ratio = min(current_silence_duration_ms / PAUSE_THRESHOLD_MS, 1.0)
            remaining_gap = BAR_WIDTH - base_x
            # Calculate width starting from the right edge
            pause_visual_width = int(pause_ratio * remaining_gap)
            
            # Draw pause indicator growing from BAR_WIDTH LEFTWARDS
            self.canvas_progress.coords(self.rect_pause, BAR_WIDTH - pause_visual_width, 0, BAR_WIDTH, 2)
            
            # Color shift for pause timer
            if pause_ratio > 0.8:
                self.canvas_progress.itemconfig(self.rect_pause, fill="#FF4444") # Red (Closing in!)
            else:
                self.canvas_progress.itemconfig(self.rect_pause, fill="#FFFF00") # Yellow (Silence)
        else:
            self.canvas_progress.coords(self.rect_pause, 0, 0, 0, 2)

        # Main bar color logic: Only turn green bar red if we are actually recording speech near the limit
        if current_silence_duration_ms == 0 and speech_progress > 0.8:
            self.canvas_progress.itemconfig(self.rect_progress, fill="#FF4444")
        else:
            self.canvas_progress.itemconfig(self.rect_progress, fill="#00FF00")
            
        self.root.after(100, self.update_loop)



# -----------------------------------------------------------------------------
# AUDIO CAPTURE
# -----------------------------------------------------------------------------
def start_capture():
    p = pyaudio.PyAudio()
    wasapi_info = None
    for i in range(p.get_host_api_count()):
        if p.get_host_api_info_by_index(i)["type"] == pyaudio.paWASAPI:
            wasapi_info = p.get_host_api_info_by_index(i)
            break
            
    if not wasapi_info: return None, None
    def_dev = p.get_device_info_by_index(wasapi_info["defaultOutputDevice"])
    print(f"[AUDIO] Source: {def_dev['name']}\n")
    
    # Try loopback find
    loopback_dev = None
    if def_dev["isLoopbackDevice"]: loopback_dev = def_dev
    else:
        for info in p.get_loopback_device_info_generator():
            if info["name"] == f"{def_dev['name']} [Loopback]":
                loopback_dev = info
                break
        if not loopback_dev: loopback_dev = next(p.get_loopback_device_info_generator())
    
    # Negotiation
    target_rate = SAMPLE_RATE # 16000
    native_rate = int(loopback_dev["defaultSampleRate"])
    actual_rate = [target_rate] 

    def callback(in_data, frame_count, time_info, status):
        if running:
            audio_data = np.frombuffer(in_data, dtype=np.float32)
            if len(audio_data) % 2 == 0: audio_data = audio_data.reshape(-1, 2).mean(axis=1) # Mono
            raw_queue.put((audio_data, actual_rate[0]))
        return (None, pyaudio.paContinue)
        
    stream = None
    try:
        # 1. Try FORCE 16k
        stream = p.open(format=pyaudio.paFloat32, channels=2,
                        rate=target_rate, input=True,
                        input_device_index=loopback_dev["index"], stream_callback=callback)
        print(f"[AUDIO] Success forcing {target_rate}Hz capture.")
        actual_rate[0] = target_rate
    except Exception as e:
        # 2. Fallback to Native
        print(f"[AUDIO] 16kHz rejected ({e}). Falling back to native {native_rate}Hz.\n")
        stream = p.open(format=pyaudio.paFloat32, channels=2,
                        rate=native_rate, input=True,
                        input_device_index=loopback_dev["index"], stream_callback=callback)
        actual_rate[0] = native_rate
        
    return stream, p

# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    
    # Load Model Globally
    print(f"[AI] Loading Model: {WHISPER_MODEL_PATH}")
    try:
        model = WhisperModel(WHISPER_MODEL_PATH, device=COMPUTE_DEVICE, compute_type=COMPUTE_TYPE)
        print("[AI] Whisper Ready.\n")
    except Exception as e:
        print(f"[AI FATAL] {e}")
        sys.exit(1)

    t_seg = threading.Thread(target=segmentation_thread, daemon=True)
    t_live = threading.Thread(target=live_inference_thread, daemon=True)
    t_final = threading.Thread(target=final_inference_thread, daemon=True)
    
    t_seg.start()
    t_live.start()
    t_final.start()
    
    
    stream, p = start_capture()
    if stream: stream.start_stream()
    
    root = tk.Tk()
    app = SubtitleApp(root)

    try:
        root.mainloop()
    except KeyboardInterrupt:
        pass
    finally:
        running = False
        if stream: stream.stop_stream(); stream.close()
        p.terminate()
