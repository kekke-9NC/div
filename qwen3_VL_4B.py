import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk, ImageDraw, ImageFont
import sys
import os
import base64
import re
import threading
import time

import json

# Try to import openai
try:
    from openai import OpenAI
except ImportError:
    messagebox.showerror("Error", "openai-python is not installed. Please run: pip install openai")
    sys.exit(1)

# Configuration
LM_STUDIO_URL = "http://localhost:1234/v1"
MODEL_ID = "qwen/qwen3-vl-4b"
AVAILABLE_MODEL_IDS = ("qwen/qwen3-vl-4b", "qwen3.5-2b")

class QwenVLApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Qwen3-VL Object Detector")
        self.root.geometry("1200x800")
        
        self.model = None
        self.image_path = None
        self.cv_image = None  # Original PIL Image
        self.tk_image = None  # PhotoImage for display
        self.is_loading = False
        self.model_id_var = tk.StringVar(value=MODEL_ID)
        
        # Detection state
        self.detected_shapes = []  # List of (label, points)
        self.visible_categories = {}  # category -> BooleanVar
        self.category_colors = {}  # category -> color

        self.setup_ui()
        
        # Setup OpenAI client for LM Studio
        self.client = OpenAI(base_url=LM_STUDIO_URL, api_key="lm-studio")
        
        # Check connection in a separate thread
        self.status_var.set("Connecting to LM Studio... Make sure local server is running.")
        threading.Thread(target=self.check_connection, daemon=True).start()

    def setup_ui(self):
        # Configure Grid
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)

        # Control Panel
        control_frame = ttk.Frame(self.root, padding=10)
        control_frame.grid(row=0, column=0, sticky="ew")

        ttk.Label(control_frame, text="AI Model:").pack(side=tk.LEFT, padx=5)
        self.model_combo = ttk.Combobox(
            control_frame,
            textvariable=self.model_id_var,
            values=AVAILABLE_MODEL_IDS,
            width=24,
        )
        self.model_combo.pack(side=tk.LEFT, padx=5)

        ttk.Label(control_frame, text="Prompt:").pack(side=tk.LEFT, padx=5)
        self.prompt_var = tk.StringVar(value="Detect the cat")
        self.prompt_entry = ttk.Entry(control_frame, textvariable=self.prompt_var, width=50)
        self.prompt_entry.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)

        self.select_btn = ttk.Button(control_frame, text="Select Image", command=self.select_image)
        self.select_btn.pack(side=tk.LEFT, padx=5)

        self.run_btn = ttk.Button(control_frame, text="Generate Box", command=self.run_inference, state=tk.DISABLED)
        self.run_btn.pack(side=tk.LEFT, padx=5)

        self.auto_btn = ttk.Button(control_frame, text="Auto Detect", command=self.run_auto_detect, state=tk.DISABLED)
        self.auto_btn.pack(side=tk.LEFT, padx=5)

        self.quick_btn = ttk.Button(control_frame, text="Quick Detect", command=self.run_quick_detect, state=tk.DISABLED)
        self.quick_btn.pack(side=tk.LEFT, padx=5)

        # Main Content Area (PanedWindow for Canvas + Legend)
        self.main_paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        self.main_paned.grid(row=1, column=0, sticky="nsew", padx=10, pady=10)

        # Image Display Area (Left)
        self.canvas_frame = ttk.Frame(self.main_paned)
        self.main_paned.add(self.canvas_frame, weight=4)

        self.canvas = tk.Canvas(self.canvas_frame, bg="#333333")
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # Legend Panel (Right)
        self.legend_frame = ttk.LabelFrame(self.main_paned, text="Legend", padding=10)
        self.main_paned.add(self.legend_frame, weight=1)

        # Scrollable legend content
        self.legend_canvas = tk.Canvas(self.legend_frame, width=180)
        self.legend_scrollbar = ttk.Scrollbar(self.legend_frame, orient="vertical", command=self.legend_canvas.yview)
        self.legend_inner = ttk.Frame(self.legend_canvas)

        self.legend_inner.bind("<Configure>", lambda e: self.legend_canvas.configure(scrollregion=self.legend_canvas.bbox("all")))
        self.legend_canvas.create_window((0, 0), window=self.legend_inner, anchor="nw")
        self.legend_canvas.configure(yscrollcommand=self.legend_scrollbar.set)

        self.legend_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.legend_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Status Bar
        self.status_var = tk.StringVar(value="Initializing...")
        self.status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.grid(row=2, column=0, sticky="ew")

    def check_connection(self):
        try:
            self.client.models.list()
            self.root.after(0, lambda: self.status_var.set("Connected to LM Studio. Select an image."))
            self.root.after(0, lambda: self.run_btn.config(state=tk.DISABLED))
            self.root.after(0, lambda: self.auto_btn.config(state=tk.DISABLED))
        except Exception as e:
            err_msg = str(e)
            self.root.after(0, lambda: self.status_var.set("Connection to LM Studio failed."))
            self.root.after(0, lambda: messagebox.showwarning("Connection Warning", 
                f"Failed to connect to LM Studio at {LM_STUDIO_URL}.\n"
                "Please start the Local Server in LM Studio with the model loaded.\n\n"
                f"Error: {err_msg}"))

    def select_image(self):
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.webp")]
        )
        if file_path:
            self.image_path = file_path
            self.load_image()
            self.run_btn.config(state=tk.NORMAL)
            self.auto_btn.config(state=tk.NORMAL)
            self.quick_btn.config(state=tk.NORMAL)
            # Clear previous detections
            self.detected_shapes = []
            self.visible_categories = {}
            self.category_colors = {}
            self._rebuild_legend()
            self.status_var.set(f"Loaded: {os.path.basename(file_path)}")

    def load_image(self):
        try:
            image = Image.open(self.image_path)
            self.cv_image = image
            self._redraw_image()
        except Exception as e:
            messagebox.showerror("Image Error", f"Failed to load image: {e}")

    def run_inference(self):
        if not self.image_path:
            return
        prompt = self.prompt_var.get()
        if not prompt:
            messagebox.showwarning("Warning", "Please enter a prompt.")
            return
        self._disable_controls()
        self.status_var.set("Running inference... (This may take a while on CPU)")
        threading.Thread(target=self._inference_worker, args=(prompt,), daemon=True).start()

    def run_auto_detect(self):
        if not self.image_path:
            return
        self._disable_controls()
        # Clear previous detections
        self.detected_shapes = []
        self.visible_categories = {}
        self.category_colors = {}
        self.root.after(0, self._rebuild_legend)
        self.status_var.set("Starting Auto Detection...")
        threading.Thread(target=self._auto_detect_worker, daemon=True).start()

    def run_quick_detect(self):
        """Single-prompt detection that identifies and labels all objects at once."""
        if not self.image_path:
            return
        self._disable_controls()
        # Clear previous detections
        self.detected_shapes = []
        self.visible_categories = {}
        self.category_colors = {}
        self.root.after(0, self._rebuild_legend)
        self.status_var.set("Starting Quick Detection (single prompt)...")
        threading.Thread(target=self._quick_detect_worker, daemon=True).start()

    def _quick_detect_worker(self):
        """Worker for single-prompt object detection with labels."""
        try:
            data_uri = self._image_to_base64_data_uri(self.image_path)
            
            system_prompt = """You are an advanced object detector. Detect ALL objects in the image and return their bounding boxes with labels.

For each object, return in this format:
label: (xmin,ymin),(xmax,ymax)

IMPORTANT: For any person detected, include estimated age and gender in the label.
Format for people: "person (age, gender)": (xmin,ymin),(xmax,ymax)
Example: "person (XXs, male or female)": (100,200),(300,500)

Multiple objects should be separated by semicolons (;).
Coordinates should be 0-1000 (normalized).

Example output:
person (XXs, male or female): (100,200),(300,500); person (XXs, male or female): (350,180),(480,520); car: (400,300),(600,450); tree: (700,100),(850,400)

Be thorough and detect as many distinct objects as possible. Include people, animals, vehicles, buildings, furniture, plants, and any other visible objects."""

            user_prompt = "Detect and label ALL visible objects in this image. For each person, estimate their age and gender. Return each object with its label and bounding box coordinates."
            
            self.root.after(0, lambda: self.status_var.set("Quick Detect: Analyzing image..."))
            
            content = self._call_vlm(system_prompt, user_prompt, data_uri)
            print(f"Quick Detect Raw Output: {content}")
            
            # Parse the labeled response
            shapes = self._parse_labeled_boxes(content)
            
            for label, shape in shapes:
                self.detected_shapes.append((label, shape))
            
            self.root.after(0, self._update_display)
            self.root.after(0, lambda: self.status_var.set(f"Quick Detect Complete. Found {len(self.detected_shapes)} objects."))

        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Quick Detect Error", f"Error: {e}"))
            self.root.after(0, lambda: self.status_var.set("Quick detection failed."))
        finally:
            self.root.after(0, self._enable_controls)

    def _parse_labeled_boxes(self, text):
        """Parse labeled bounding boxes from VLM response."""
        results = []
        
        # Split by semicolon for multiple objects
        if ';' in text:
            parts = text.split(';')
        else:
            parts = [text]
        
        for part in parts:
            part = part.strip()
            if not part:
                continue
            
            # Try to extract label and coordinates
            # Format: "label: (x1,y1),(x2,y2)" or "person (age, gender): (x1,y1),(x2,y2)"
            # Find the last colon before coordinates pattern
            coord_match = re.search(r':\s*\((\d+),\s*(\d+)\)', part)
            if coord_match:
                # Split at the colon that precedes the coordinates
                colon_pos = coord_match.start()
                label = part[:colon_pos].strip()
                coords_text = part[colon_pos+1:]
            else:
                label = "object"
                coords_text = part
            
            # Parse coordinates
            point_pattern = r"\((-?\d+),\s*(-?\d+)\)"
            points = re.findall(point_pattern, coords_text)
            parsed_points = []
            for p in points:
                x, y = map(int, p)
                if 0 <= x <= 1000 and 0 <= y <= 1000:
                    parsed_points.append((x, y))
            
            if 2 <= len(parsed_points) <= 20:
                results.append((label, parsed_points))
        
        return results

    def _disable_controls(self):
        self.run_btn.config(state=tk.DISABLED)
        self.auto_btn.config(state=tk.DISABLED)
        self.quick_btn.config(state=tk.DISABLED)
        self.select_btn.config(state=tk.DISABLED)

    def _enable_controls(self):
        self.run_btn.config(state=tk.NORMAL)
        self.auto_btn.config(state=tk.NORMAL)
        self.quick_btn.config(state=tk.NORMAL)
        self.select_btn.config(state=tk.NORMAL)

    def _call_vlm(self, system_prompt, user_prompt, data_uri):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": [
                {"type": "text", "text": user_prompt},
                {"type": "image_url", "image_url": {"url": data_uri}}
            ]}
        ]
        response = self.client.chat.completions.create(
            model=self.model_id_var.get().strip() or MODEL_ID,
            messages=messages,
            temperature=0.1,
            top_p=0.9,
            max_tokens=512
        )
        return response.choices[0].message.content

    def _inference_worker(self, prompt):
        try:
            data_uri = self._image_to_base64_data_uri(self.image_path)
            detect_keywords = ["detect", "find", "locate", "point out", "検出", "探して"]
            if not any(x in prompt.lower() for x in detect_keywords):
                final_prompt = f"Detect all {prompt} instances"
            else:
                final_prompt = prompt

            system_prompt = "You are an object detector. Find all instances of the object specified by the user in the image. Return bounding box coordinates (xmin,ymin),(xmax,ymax) for each instance. Separate multiple objects with semicolon ';'."

            content = self._call_vlm(system_prompt, final_prompt, data_uri)
            print(f"Raw Output: {content}")
            
            shapes = self._parse_boxes(content)
            labeled_shapes = [(prompt, shape) for shape in shapes]
            
            self.detected_shapes = labeled_shapes
            self.root.after(0, self._update_display)

        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Inference Error", f"Error during inference: {e}"))
            self.root.after(0, lambda: self.status_var.set("Inference failed."))
        finally:
            self.root.after(0, self._enable_controls)

    def _auto_detect_worker(self):
        try:
            data_uri = self._image_to_base64_data_uri(self.image_path)
            
            # Step 1: List objects
            self.root.after(0, lambda: self.status_var.set("Auto Detect: Identifying object types..."))
            
            list_system_prompt = "You are an AI assistant. Analyze the image and list all distinct object categories visible in the image. Return ONLY a comma-separated list of object names (e.g., 'person, car, tree'). Do not output anything else."
            list_prompt = "List all object categories in this image."
            
            list_response = self._call_vlm(list_system_prompt, list_prompt, data_uri)
            print(f"Auto List Response: {list_response}")
            
            clean_list_text = list_response.replace("Object categories:", "").strip()
            # Filter out "star" or "星" as requested
            exclude_keywords = ["star", "星"]
            raw_object_types = [x.strip() for x in clean_list_text.split(',') if x.strip()]
            object_types = [ot for ot in raw_object_types if not any(k in ot.lower() for k in exclude_keywords)]
            
            if not object_types:
                self.root.after(0, lambda: messagebox.showinfo("Auto Detect", "No objects identified (or all were excluded) in the first step."))
                return

            system_prompt = "You are an object detector. Find all instances of the object specified by the user in the image. Return bounding box coordinates (xmin,ymin),(xmax,ymax) for each instance. Separate multiple objects with semicolon ';'."

            for obj_type in object_types:
                obj_type_lower = obj_type.lower()
                self.root.after(0, lambda ot=obj_type: self.status_var.set(f"Auto Detect: detecting '{ot}'..."))
                
                # Request individual detection for all object types
                detect_prompt = f"Detect each individual {obj_type} separately. Return a bounding box for EACH distinct {obj_type}, not one box for all. Find as many as possible."
                content = self._call_vlm(system_prompt, detect_prompt, data_uri)
                print(f"Detect '{obj_type}' Output: {content}")
                
                shapes = self._parse_boxes(content)
                
                # For person, get rich labels
                if 'person' in obj_type_lower or 'people' in obj_type_lower or '人' in obj_type:
                    for i, shape in enumerate(shapes):
                        # Get details for each person
                        self.root.after(0, lambda: self.status_var.set(f"Auto Detect: analyzing person {i+1}..."))
                        detail_prompt = f"Describe the person at approximately position ({shape[0][0]},{shape[0][1]}) to ({shape[1][0] if len(shape)>1 else shape[0][0]},{shape[1][1] if len(shape)>1 else shape[0][1]}). Return ONLY: age estimate, gender, and what they are doing. Format: 'Age: X, Gender: Y, Situation: Z'"
                        detail_system = "You are an image analyst. Describe the person at the specified location concisely. Return only: Age: X, Gender: Y, Situation: Z"
                        try:
                            detail_response = self._call_vlm(detail_system, detail_prompt, data_uri)
                            print(f"Person {i+1} detail: {detail_response}")
                            # Create rich label
                            rich_label = f"person ({detail_response.strip()[:50]})"
                        except:
                            rich_label = "person"
                        self.detected_shapes.append((rich_label, shape))
                else:
                    for shape in shapes:
                        self.detected_shapes.append((obj_type, shape))
                
                # Progressive update after each category
                self.root.after(0, self._update_display)
            
            self.root.after(0, lambda: self.status_var.set(f"Auto Detect Complete. Found {len(self.detected_shapes)} objects."))

        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Auto Detect Error", f"Error: {e}"))
            self.root.after(0, lambda: self.status_var.set("Auto detection failed."))
        finally:
            self.root.after(0, self._enable_controls)

    def _update_display(self):
        # Rebuild legend and redraw image
        self._rebuild_legend()
        self._redraw_image()

    def _rebuild_legend(self):
        # Clear existing legend items
        for widget in self.legend_inner.winfo_children():
            widget.destroy()
        
        # Get unique categories from detected_shapes
        categories = set()
        for label, _ in self.detected_shapes:
            # Use base category (before parentheses) for grouping checkboxes
            base_cat = label.split('(')[0].strip() if '(' in label else label
            categories.add(base_cat)
        
        for cat in sorted(categories):
            if cat not in self.visible_categories:
                self.visible_categories[cat] = tk.BooleanVar(value=True)
            if cat not in self.category_colors:
                self.category_colors[cat] = self._get_color(cat)
            
            color = self.category_colors[cat]
            
            frame = ttk.Frame(self.legend_inner)
            frame.pack(fill=tk.X, pady=2)
            
            # Color swatch
            swatch = tk.Canvas(frame, width=20, height=20, highlightthickness=0)
            swatch.create_rectangle(0, 0, 20, 20, fill=color, outline=color)
            swatch.pack(side=tk.LEFT, padx=5)
            
            # Checkbox
            cb = ttk.Checkbutton(frame, text=cat, variable=self.visible_categories[cat],
                                 command=self._redraw_image)
            cb.pack(side=tk.LEFT)

    def _image_to_base64_data_uri(self, path):
        # Resize image to 1080px height while maintaining aspect ratio
        img = Image.open(path)
        target_height = 1080
        w, h = img.size
        if h != target_height:
            scale = target_height / h
            new_w = int(w * scale)
            img = img.resize((new_w, target_height), Image.Resampling.LANCZOS)
        
        # Convert to JPEG bytes
        import io
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=85)
        encoded_string = base64.b64encode(buffer.getvalue()).decode('utf-8')
        return f"data:image/jpeg;base64,{encoded_string}" 

    def _parse_boxes(self, text):
        shapes = []
        if ';' in text:
            parts = text.split(';')
        else:
            parts = [text]
        for part in parts:
            if not part.strip():
                continue
            # Also match negative numbers to filter them out
            point_pattern = r"\((-?\d+),\s*(-?\d+)\)"
            points = re.findall(point_pattern, part)
            parsed_points = []
            for p in points:
                x, y = map(int, p)
                # Validate: coordinates should be 0-1000
                if 0 <= x <= 1000 and 0 <= y <= 1000:
                    parsed_points.append((x, y))
            # Only accept shapes with 2-20 valid points
            if 2 <= len(parsed_points) <= 20:
                shapes.append(parsed_points)
        return shapes

    def _get_color(self, label):
        import hashlib
        hash_object = hashlib.md5(label.encode())
        hex_dig = hash_object.hexdigest()
        colors = [
            "#FF0000", "#00FF00", "#0000FF", "#FFFF00", "#FF00FF", "#00FFFF", 
            "#FFA500", "#800080", "#008000", "#FFC0CB", "#A52A2A", "#808080"
        ]
        index = int(hex_dig, 16) % len(colors)
        return colors[index]

    def _redraw_image(self):
        if self.cv_image is None:
            return
            
        draw_image = self.cv_image.copy()
        draw = ImageDraw.Draw(draw_image)
        
        width, height = draw_image.size
        
        # Dynamic font size based on image dimensions (roughly 2% of shorter dimension)
        font_size = max(16, int(min(width, height) * 0.025))
        
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except IOError:
            font = ImageFont.load_default()
        
        width, height = draw_image.size
        
        for label, points in self.detected_shapes:
            # Check visibility by base category
            base_cat = label.split('(')[0].strip() if '(' in label else label
            if base_cat in self.visible_categories:
                if not self.visible_categories[base_cat].get():
                    continue  # Skip hidden categories
            
            color = self.category_colors.get(base_cat, self._get_color(base_cat))
            
            pixel_points = []
            for (xn, yn) in points:
                x = xn / 1000 * width
                y = yn / 1000 * height
                pixel_points.append((x, y))
            
            if len(pixel_points) == 2:
                x1, y1 = pixel_points[0]
                x2, y2 = pixel_points[1]
                draw.rectangle([x1, y1, x2, y2], outline=color, width=5)
                text_x, text_y = x1, max(0, y1 - 25)
            elif len(pixel_points) > 2:
                draw.polygon(pixel_points, outline=color, width=5)
                text_x, text_y = pixel_points[0][0], max(0, pixel_points[0][1] - 25)
            else:
                continue
            
            # Draw label
            try:
                if hasattr(draw, "textbbox"):
                    left, top, right, bottom = draw.textbbox((text_x, text_y), label, font=font)
                    draw.rectangle((left-3, top-3, right+3, bottom+3), fill=color)
                else:
                    w, h = draw.textsize(label, font=font)
                    draw.rectangle((text_x-3, text_y-3, text_x+w+3, text_y+h+3), fill=color)
                draw.text((text_x, text_y), label, fill="white", font=font)
            except Exception as e:
                print(f"Drawing text failed: {e}")
                draw.text((text_x, text_y), label, fill=color)
        
        # Resize for display
        display_max_width = 800
        display_max_height = 600
        w, h = draw_image.size
        scale = min(display_max_width/w, display_max_height/h)
        
        if scale < 1:
            new_w, new_h = int(w * scale), int(h * scale)
            self.display_image = draw_image.resize((new_w, new_h), Image.Resampling.LANCZOS)
        else:
            self.display_image = draw_image.copy()
            
        self.tk_image = ImageTk.PhotoImage(self.display_image)
        self.canvas.delete("all")
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        self.canvas.create_image(cw//2, ch//2, anchor=tk.CENTER, image=self.tk_image)

if __name__ == "__main__":
    root = tk.Tk()
    app = QwenVLApp(root)
    root.mainloop()
