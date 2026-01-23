
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
import gc
import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info

# Configuration
MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"

class QwenVLApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Qwen3-VL Object Detector (Local)")
        self.root.geometry("1200x800")
        
        self.model = None
        self.processor = None
        self.image_path = None
        self.cv_image = None  # Original PIL Image
        self.tk_image = None  # PhotoImage for display
        self.is_loading = False
        
        self.detected_shapes = []  # List of (label, points)
        self.visible_categories = {}  # category -> BooleanVar
        self.category_colors = {}  # category -> color

        self.setup_ui()
        
        # Check connection in a separate thread
        self.status_var.set("Initializing model... Please wait.")
        threading.Thread(target=self.initialize_model, daemon=True).start()

    def setup_ui(self):
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)

        control_frame = ttk.Frame(self.root, padding=10)
        control_frame.grid(row=0, column=0, sticky="ew")

        ttk.Label(control_frame, text="Prompt:").pack(side=tk.LEFT, padx=5)
        self.prompt_var = tk.StringVar(value="Detect the meteor")
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

        self.main_paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        self.main_paned.grid(row=1, column=0, sticky="nsew", padx=10, pady=10)

        self.canvas_frame = ttk.Frame(self.main_paned)
        self.main_paned.add(self.canvas_frame, weight=4)

        self.canvas = tk.Canvas(self.canvas_frame, bg="#333333")
        self.canvas.pack(fill=tk.BOTH, expand=True)

        self.legend_frame = ttk.LabelFrame(self.main_paned, text="Legend", padding=10)
        self.main_paned.add(self.legend_frame, weight=1)

        self.legend_canvas = tk.Canvas(self.legend_frame, width=180)
        self.legend_scrollbar = ttk.Scrollbar(self.legend_frame, orient="vertical", command=self.legend_canvas.yview)
        self.legend_inner = ttk.Frame(self.legend_canvas)

        self.legend_inner.bind("<Configure>", lambda e: self.legend_canvas.configure(scrollregion=self.legend_canvas.bbox("all")))
        self.legend_canvas.create_window((0, 0), window=self.legend_inner, anchor="nw")
        self.legend_canvas.configure(yscrollcommand=self.legend_scrollbar.set)

        self.legend_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.legend_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.status_var = tk.StringVar(value="Initializing...")
        self.status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.grid(row=2, column=0, sticky="ew")

    def initialize_model(self):
        try:
            self.root.after(0, lambda: self.status_var.set("Loading model... (This may take a while)"))
            

            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16
            )
            
            local_model_dir = "./quantized_model"
            
            loaded = False
            # Try loading from local directory first
            if os.path.exists(local_model_dir):
                print(f"Loading local model from {local_model_dir}...")
                try:
                    self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                        local_model_dir,
                        device_map="cuda",
                        trust_remote_code=True,
                        low_cpu_mem_usage=True
                    )
                    self.processor = AutoProcessor.from_pretrained(local_model_dir, trust_remote_code=True, fix_mistral_regex=True)
                    loaded = True
                    print("Loaded from local storage.")
                except Exception as e:
                    print(f"Failed to load local model: {e}")
            
            # If not loaded, download and load
            if not loaded:
                print(f"Downloading/Loading model {MODEL_ID}...")
                self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                    MODEL_ID,
                    quantization_config=bnb_config,
                    device_map="cuda",
                    low_cpu_mem_usage=True,
                    trust_remote_code=True
                )
                self.processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True, fix_mistral_regex=True)
                
                # Try to save locally
                try:
                    print("Saving model locally...")
                    self.model.save_pretrained(local_model_dir)
                    self.processor.save_pretrained(local_model_dir)
                    print("Model saved locally.")
                except Exception as e:
                    print(f"Failed to save model: {e}")

            self.root.after(0, lambda: self.status_var.set("Model loaded successfully. Select an image."))
            # We don't disable buttons here because they are enabled on image selection
            
        except Exception as e:
            err_msg = str(e)
            print(f"Model initialization error: {e}")
            self.root.after(0, lambda: self.status_var.set("Model loading failed."))
            self.root.after(0, lambda: messagebox.showerror("Error", f"Failed to load model:\n{err_msg}"))

    def _cleanup(self):
        """Explicitly release memory."""
        try:
            if hasattr(self, 'inputs'):
                del self.inputs
            gc.collect()
            torch.cuda.empty_cache()
        except:
            pass

    def select_image(self):
        self._cleanup() # Clean up before loading new image
        
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.webp")]
        )
        if file_path:
            self.image_path = file_path
            self.load_image()
            if self.model is not None:
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
        self.status_var.set("Running inference...")
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
        self.status_var.set("Starting Quick Detection...")
        threading.Thread(target=self._quick_detect_worker, daemon=True).start()

    def _quick_detect_worker(self):
        try:
            system_prompt = """You are an advanced object detector. Detect ALL objects in the image and return their bounding boxes with labels.

For each object, return in this format:
label: (xmin,ymin),(xmax,ymax)

IMPORTANT: For any person detected, include estimated age and gender in the label.
Format for people: "person (age, gender)": (xmin,ymin),(xmax,ymax)
Example: "person (XXs, male or female)": (100,200),(300,500)

Multiple objects should be separated by semicolons (;).
Coordinates should be 0-1000 (normalized).

Be thorough and detect as many distinct objects as possible."""

            user_prompt = "Detect and label ALL visible objects in this image. For each person, estimate their age and gender. Return each object with its label and bounding box coordinates."
            
            self.root.after(0, lambda: self.status_var.set("Quick Detect: Analyzing image..."))
            
            content = self._call_vlm(system_prompt, user_prompt, self.image_path)
            print(f"Quick Detect Raw Output: {content}")
            
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
        results = []
        if ';' in text:
            parts = text.split(';')
        else:
            parts = [text]
        
        for part in parts:
            part = part.strip()
            if not part:
                continue
            
            coord_match = re.search(r':\s*\((\d+),\s*(\d+)\)', part)
            if coord_match:
                colon_pos = coord_match.start()
                label = part[:colon_pos].strip()
                coords_text = part[colon_pos+1:]
            else:
                label = "object"
                coords_text = part
            
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

    def _call_vlm(self, system_prompt, user_prompt, image_path):
        if self.model is None or self.processor is None:
            raise Exception("Model not loaded yet.")

        # Prepare image
        try:
            image_obj = Image.open(image_path)
            # Resize logic similar to llm_test/app.py
            if image_obj.height != 1080:
                aspect_ratio = image_obj.width / image_obj.height
                new_height = 1080
                new_width = int(new_height * aspect_ratio)
                image_obj = image_obj.resize((new_width, new_height))
        except Exception as e:
            print(f"Image processing error: {e}")
            raise e

        # Construct messages
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": [
                {"type": "image", "image": image_obj},
                {"type": "text", "text": user_prompt}
            ]}
        ]

        # Process inputs
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to("cuda")

        # Generate
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.7,
                top_p=0.8,
                top_k=20,
                do_sample=True,
            )
        
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        
        # Explicit cleanup of large tensors
        del inputs
        del generated_ids
        del image_inputs
        del video_inputs
        torch.cuda.empty_cache()
        gc.collect()

        return output_text

    def _inference_worker(self, prompt):
        try:
            detect_keywords = ["detect", "find", "locate", "point out", "検出", "探して"]
            if not any(x in prompt.lower() for x in detect_keywords):
                final_prompt = f"Detect all {prompt} instances"
            else:
                final_prompt = prompt

            system_prompt = "You are an object detector. Find all instances of the object specified by the user in the image. Return bounding box coordinates (xmin,ymin),(xmax,ymax) for each instance. Separate multiple objects with semicolon ';'."

            content = self._call_vlm(system_prompt, final_prompt, self.image_path)
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
            # Step 1: List objects
            self.root.after(0, lambda: self.status_var.set("Auto Detect: Identifying object types..."))
            
            list_system_prompt = "You are an AI assistant. Analyze the image and list all distinct object categories visible in the image. Return ONLY a comma-separated list of object names (e.g., 'person, car, tree'). Do not output anything else."
            list_prompt = "List all object categories in this image."
            
            list_response = self._call_vlm(list_system_prompt, list_prompt, self.image_path)
            print(f"Auto List Response: {list_response}")
            
            clean_list_text = list_response.replace("Object categories:", "").strip()
            exclude_keywords = ["star", "星"]
            raw_object_types = [x.strip() for x in clean_list_text.split(',') if x.strip()]
            object_types = [ot for ot in raw_object_types if not any(k in ot.lower() for k in exclude_keywords)]
            
            if not object_types:
                self.root.after(0, lambda: messagebox.showinfo("Auto Detect", "No objects identified."))
                return

            system_prompt = "You are an object detector. Find all instances of the object specified by the user in the image. Return bounding box coordinates (xmin,ymin),(xmax,ymax) for each instance. Separate multiple objects with semicolon ';'."

            for obj_type in object_types:
                obj_type_lower = obj_type.lower()
                self.root.after(0, lambda ot=obj_type: self.status_var.set(f"Auto Detect: detecting '{ot}'..."))
                
                detect_prompt = f"Detect each individual {obj_type} separately. Return a bounding box for EACH distinct {obj_type}, not one box for all. Find as many as possible."
                content = self._call_vlm(system_prompt, detect_prompt, self.image_path)
                print(f"Detect '{obj_type}' Output: {content}")
                
                shapes = self._parse_boxes(content)
                
                if 'person' in obj_type_lower or 'people' in obj_type_lower or '人' in obj_type:
                    for i, shape in enumerate(shapes):
                        self.root.after(0, lambda: self.status_var.set(f"Auto Detect: analyzing person {i+1}..."))
                        detail_prompt = f"Describe the person at approximately position ({shape[0][0]},{shape[0][1]}). Return ONLY: age estimate, gender, and what they are doing. Format: 'Age: X, Gender: Y, Situation: Z'"
                        detail_system = "You are an image analyst. Describe the person at the specified location concisely."
                        try:
                            # We need to crop or point to the person? The original code sent the full image.
                            # The original code's detail prompt was: 
                            # "Describe the person at approximately position ({shape[0][0]},{shape[0][1]}) to ..."
                            # The model takes the full image and the prompt restricts attention.
                            detail_response = self._call_vlm(detail_system, detail_prompt, self.image_path)
                            print(f"Person {i+1} detail: {detail_response}")
                            rich_label = f"person ({detail_response.strip()[:50]})"
                        except:
                            rich_label = "person"
                        self.detected_shapes.append((rich_label, shape))
                else:
                    for shape in shapes:
                        self.detected_shapes.append((obj_type, shape))
                
                self.root.after(0, self._update_display)
            
            self.root.after(0, lambda: self.status_var.set(f"Auto Detect Complete. Found {len(self.detected_shapes)} objects."))

        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Auto Detect Error", f"Error: {e}"))
            self.root.after(0, lambda: self.status_var.set("Auto detection failed."))
        finally:
            self.root.after(0, self._enable_controls)

    def _update_display(self):
        self._rebuild_legend()
        self._redraw_image()

    def _rebuild_legend(self):
        for widget in self.legend_inner.winfo_children():
            widget.destroy()
        
        categories = set()
        for label, _ in self.detected_shapes:
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
            
            swatch = tk.Canvas(frame, width=20, height=20, highlightthickness=0)
            swatch.create_rectangle(0, 0, 20, 20, fill=color, outline=color)
            swatch.pack(side=tk.LEFT, padx=5)
            
            cb = ttk.Checkbutton(frame, text=cat, variable=self.visible_categories[cat],
                                 command=self._redraw_image)
            cb.pack(side=tk.LEFT)

    def _parse_boxes(self, text):
        shapes = []
        if ';' in text:
            parts = text.split(';')
        else:
            parts = [text]
        for part in parts:
            if not part.strip():
                continue
            point_pattern = r"\((-?\d+),\s*(-?\d+)\)"
            points = re.findall(point_pattern, part)
            parsed_points = []
            for p in points:
                x, y = map(int, p)
                if 0 <= x <= 1000 and 0 <= y <= 1000:
                    parsed_points.append((x, y))
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
        
        font_size = max(16, int(min(width, height) * 0.025))
        
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except IOError:
            font = ImageFont.load_default()
        
        for label, points in self.detected_shapes:
            base_cat = label.split('(')[0].strip() if '(' in label else label
            if base_cat in self.visible_categories:
                if not self.visible_categories[base_cat].get():
                    continue
            
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
