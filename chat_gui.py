import tkinter as tk
from tkinter import ttk
from tkinter.scrolledtext import ScrolledText
import threading
import bright_area_detector

# Define styles to match main_gui
BG_COLOR = "#2E3F5B"
FG_COLOR = "#EAEAEA"
SELECT_BG = "#4A6A9B"
FRAME_BG = "#263347"
ENTRY_BG = "#3A4D6B"

class ChatTab(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        self.configure(style="TFrame")
        self._setup_ui()

    def _setup_ui(self):
        # Configure local styles if needed, though main_gui sets most T* styles.
        # We need specific colors for ScrolledText and Entry as they are tk widgets or need specific config.
        
        # Chat history area
        self.history_frame = ttk.Frame(self)
        self.history_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.history_area = ScrolledText(
            self.history_frame, 
            state='disabled', 
            wrap=tk.WORD, 
            font=("Segoe UI", 10),
            bg=FRAME_BG,
            fg=FG_COLOR,
            insertbackground=FG_COLOR, # caret color
            bd=0,
            highlightthickness=1,
            highlightbackground=SELECT_BG
        )
        self.history_area.pack(fill=tk.BOTH, expand=True)

        # Tag configuration for colors
        self.history_area.tag_config("user", foreground="#87CEEB", font=("Segoe UI", 10, "bold"))
        self.history_area.tag_config("ai", foreground="#98FB98", font=("Segoe UI", 10, "bold"))
        self.history_area.tag_config("system", foreground="#AAAAAA", font=("Segoe UI", 9, "italic"))
        self.history_area.tag_config("error", foreground="#FF6B6B")
        
        # Input area container
        input_container = ttk.Frame(self)
        input_container.pack(fill=tk.X, padx=10, pady=(0, 10))
        
        # Input field
        self.input_field = ttk.Entry(input_container) # Style is handled by TEntry in main_gui
        self.input_field.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.input_field.bind("<Return>", self.send_message)
        
        # Send button
        self.send_btn = ttk.Button(input_container, text="送信", command=self.send_message)
        self.send_btn.pack(side=tk.RIGHT, padx=(5, 0))

        # Initial message
        self.append_message("System", "Qwen3-VL 4B Chat Ready. You can ask me about how to use this app or other questions.", "system")

        # Application Knowledge Base for the AI
        self.system_prompt = """You are an expert technical assistant for the "Meteor Detector" (Automated Meteor Detection and Analysis Pipeline) software. 
Your goal is to provide precise, helpful, and detailed information about the application's features, settings, and underlying algorithms.

【Overview】
This application detects and analyzes meteors from video files (MP4, AVI, MOV) and RTSP streams (live cameras like Atom Cam 2). It uses a two-stage detection process:
1. Coarse Detection: Frame differencing and Probabilistic Hough Transform to find linear motion.
2. Fine Classification: A deep learning model (ComplexCNN, a custom ResNet) checks candidates to reject false positives (planes, satellites, insects).

【Core tabs & Features】
- Usage (使い方): Basic setup guide.
- Source Selection (ソース選択): 
    - Drag and drop files/folders. 
    - RTSP URL setup (supports hardware acceleration if available).
    - 'Periodic Scan': Monitors a directory (e.g., Atom Cam network share) for new files every N seconds.
- Settings (保存設定 / 各種設定):
    - Save paths (Default: 'meteor' and 'not_meteor' folders).
    - Save Options: Choose to save video clips, cutouts, full-size diffs, or 'Summary Video'.
    - Astrometry: API Key for 'Astrometry.net' (Local solve via WSL /usr/share/astrometry/data is also supported).
- Analysis (解析):
    - Meteor Path Visualization: Drag '.txt' info files to see trajectories.
    - Long Exposure Map: Combine multiple images into one.
    - Lighten Blend: Create composite images/videos. Supports 'AI Mode' to only blend sections where meteors were detected.
- ⚙️ (Advanced Settings):
    - Min Line Length: Minimum length of a streak to be detected (Default: 25px).
    - Meteor Probability: Threshold for the CNN (Default: 0.5).
    - Airplane Detection: Thresholds for duration and distance to filter out aircraft.

【Technical Details】
- Algorithm: 'video_processing.py' uses Canny edge detection and Hough lines.
- Astrometry: Converts pixel coordinates to RA/Dec (equatorial system). Requires 1080p images.
- RTSP optimization: Uses TCP for stability and NVIDIA HWAccel if configured. Has presets for 'Clear' and 'Cloudy' skies to adjust sensitivity.
- Time Sync: 'auto_time_updater.py' keeps the system clock accurate for scientific timing.

【Common Q&A】
- Q: Where are results? A: In the 'meteor' folder in the app directory.
- Q: How to reduce false positives? A: Increase 'Min Line Length' or 'Meteor Probability Threshold' in Advanced Settings.
- Q: Plate solving fails. A: Ensure the API key is correct and you have an internet connection, or that the local WSL silver is set up.

Answer clearly. Use bullet points for readability. If you don't know something, suggest checking the technical PDF or README."""

    def send_message(self, event=None):
        msg = self.input_field.get()
        if not msg.strip():
            return
            
        self.append_message("User", msg, "user")
        self.input_field.delete(0, tk.END)
        self.input_field.config(state='disabled')
        self.send_btn.config(state='disabled')
        
        # Thread generation
        threading.Thread(target=self._get_response, args=(msg,), daemon=True).start()

    def append_message(self, sender, message, tag=None):
        self.history_area.configure(state='normal')
        self.history_area.insert(tk.END, f"\n[{sender}]:\n", tag)
        self.history_area.insert(tk.END, f"{message}\n")
        self.history_area.configure(state='disabled')
        self.history_area.see(tk.END)

    def _get_response(self, user_msg):
        try:
            response = bright_area_detector.generate_response(
                user_prompt=user_msg,
                system_prompt=self.system_prompt
            )
        except Exception as e:
            response = f"Error: {e}"
        
        # Update UI in main thread
        self.after(0, self._on_response_ready, response)

    def _on_response_ready(self, response):
        self.append_message("AI", response, "ai")
        self.input_field.config(state='normal')
        self.send_btn.config(state='normal')
        self.input_field.focus()

def create_tab(parent):
    return ChatTab(parent)
