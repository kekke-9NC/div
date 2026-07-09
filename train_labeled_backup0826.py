import customtkinter as ctk
from tkinter import filedialog, ttk
import threading
import queue
import os
from pathlib import Path
from datetime import datetime
from PIL import Image, ImageTk
import copy

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np
import time
import random

import matplotlib
matplotlib.use('Agg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import model_catalog

# Diff画像は256x256を学習対象とする（ファイル名ではなく実解像度で判定）
IMAGE_SIZE = 256
TARGET_DIFF_SIZE = (256, 256)  # (width, height)
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

# =========================
# Added: Training Stabilization Utilities
# =========================

def set_global_seed(seed: int = 42):
    """Set seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def iter_image_files(root_dir):
    if not root_dir or not os.path.isdir(root_dir):
        return
    for path in Path(root_dir).rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield str(path)


def read_image_size(path):
    try:
        with Image.open(path) as img:
            return tuple(img.size)  # (width, height)
    except Exception:
        return None


def collect_diff_images_by_target_size(data_paths, target_size=TARGET_DIFF_SIZE):
    """
    Collect training images by exact target resolution for each class.
    Filename is ignored; only actual image size is used.
    """
    target_size = tuple(target_size)
    selected_paths = {}
    counts_by_size = {}
    for cls_name, class_dir in data_paths.items():
        size_to_paths = {}
        for fp in iter_image_files(class_dir):
            size = read_image_size(fp)
            if size is None:
                continue
            size_to_paths.setdefault(size, []).append(fp)
        selected_paths[cls_name] = list(size_to_paths.get(target_size, []))
        counts_by_size[cls_name] = {str(k): len(v) for k, v in size_to_paths.items()}
    return target_size, selected_paths, counts_by_size


def estimate_mean_std(dataset, max_samples=2048):
    """Estimate mean/std from the dataset (for each RGB channel)"""
    if len(dataset.image_paths) == 0:
        # Fallback (ImageNet statistics)
        return [0.027, 0.027, 0.027], [0.046, 0.046, 0.046]
    tmp = transforms.ToTensor()
    n = min(max_samples, len(dataset.image_paths))
    idxs = np.random.choice(len(dataset.image_paths), size=n, replace=False)
    m = torch.zeros(3)
    s = torch.zeros(3)
    count = 0
    for i in idxs:
        try:
            img = Image.open(dataset.image_paths[i]).convert('RGB')
            t = tmp(img)
            m += t.mean(dim=[1, 2])
            s += t.std(dim=[1, 2])
            count += 1
        except Exception:
            continue
    if count == 0:
        return [0.027, 0.027, 0.027], [0.046, 0.046, 0.046]
    m /= count
    s /= count
    return m.tolist(), s.tolist()


def mixup_data(x, y, alpha=0.4):
    if alpha <= 0:
        return x, y, y, 1.0
    lam = np.random.beta(alpha, alpha)
    index = torch.randperm(x.size(0), device=x.device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


class EMA:
    """Exponential Moving Average for model parameters (evaluation smoothing)."""
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {n: p.clone().detach() for n, p in model.named_parameters() if p.requires_grad}
        self.backup = {}

    @torch.no_grad()
    def update(self, model):
        for (n, p) in model.named_parameters():
            if n in self.shadow:
                self.shadow[n].mul_(self.decay).add_(p.data, alpha=1 - self.decay)

    def apply_shadow(self, model):
        self.backup = {}
        for (n, p) in model.named_parameters():
            if n in self.shadow:
                self.backup[n] = p.data.clone()
                p.data.copy_(self.shadow[n])

    def restore(self, model):
        for (n, p) in model.named_parameters():
            if n in self.backup:
                p.data.copy_(self.backup[n])
        self.backup = {}

# =======================================================
# --- より複雑なCNNモデルの定義 (ResNet風アーキテクチャ) ---
# =======================================================

class ResidualBlock(nn.Module):
    """
    Residual Block（残差ブロック）。
    ResNetの基本構成要素。スキップコネクションを持つ。
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.relu(out)
        return out

class ComplexCNN(nn.Module):
    """
    Residual Blockを組み合わせた、より複雑なCNNモデル。
    """
    def __init__(self, num_classes=2):
        super(ComplexCNN, self).__init__()
        self.in_channels = 64
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(512, num_classes)
        )

    def _make_layer(self, out_channels, num_blocks, stride):
        """Residual Blockを複数持つステージを作成するヘルパー関数"""
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(ResidualBlock(self.in_channels, out_channels, stride=s))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


# ===== TTA (Test-Time Augmentation with flips) =====

def tta_predict(model, pil_image, device, mean=(0.027, 0.027, 0.027), std=(0.046, 0.046, 0.046), input_resize=None):
    ops = []
    if input_resize is not None:
        try:
            h, w = int(input_resize[0]), int(input_resize[1])
            if h > 0 and w > 0:
                ops.append(transforms.Resize((h, w)))
        except Exception:
            pass
    ops.extend([transforms.ToTensor(), transforms.Normalize(mean, std)])
    tf = transforms.Compose(ops)
    flips = [
        lambda x: x,
        lambda x: x.transpose(Image.FLIP_LEFT_RIGHT),
        lambda x: x.transpose(Image.FLIP_TOP_BOTTOM)
    ]
    probs = []
    model.eval()
    with torch.no_grad():
        for f in flips:
            t = tf(f(pil_image)).unsqueeze(0).to(device)
            p = torch.softmax(model(t), dim=1)
            probs.append(p)
    return torch.stack(probs).mean(dim=0).squeeze(0)

# FIX: Moved worker_init_fn to the top level to make it picklable for multiprocessing.
def worker_init_fn(worker_id):
    """Ensures that each dataloader worker has a different random seed."""
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


class ImageClassifierApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Shooting Star Classifier AI - High Performance & Compatibility Mode")
        self.geometry("1200x900")
        ctk.set_appearance_mode("System")
        ctk.set_default_color_theme("blue")
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)
        self.train_thread = None
        self.data_queue = queue.Queue()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"=== Device Info ===")
        print(f"Using device: {self.device}")
        if self.device.type == 'cuda':
            print(f"GPU Name: {torch.cuda.get_device_name(0)}")
            print(f"CUDA Capability: {torch.cuda.get_device_capability(0)}")
        print("===================")
        self.model = None
        self.class_names = None
        self.norm_stats = None
        self.input_resize = None
        self.epoch_models_cache = []
        self.plot_data = {
            'epochs': [], 'train_loss': [], 'val_loss': [],
            'train_acc': [], 'val_acc': []
        }
        self.create_widgets()
        self.process_queue()

    def create_widgets(self):
        self.control_frame = ctk.CTkFrame(self, corner_radius=10)
        self.control_frame.grid(row=0, column=0, padx=10, pady=10, sticky="ns")
        self.control_frame.grid_rowconfigure(0, weight=1)
        self.control_frame.grid_columnconfigure(0, weight=1)

        self.tab_view = ctk.CTkTabview(self.control_frame, width=360)
        self.tab_view.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        self.tab_view.add("Basic Settings")
        self.tab_view.add("Data Augmentation")

        tab_1 = self.tab_view.tab("Basic Settings")
        ctk.CTkLabel(tab_1, text="Shooting Star Folder", font=ctk.CTkFont(weight="bold")).pack(pady=(10, 2), padx=10, anchor="w")
        self.meteor_dir_entry = ctk.CTkEntry(tab_1, placeholder_text="Select shooting star image folder")
        self.meteor_dir_entry.pack(pady=2, padx=10, fill="x")
        self.meteor_dir_entry.insert(0, "D:/training/test_sample/meteor")
        self.meteor_dir_button = ctk.CTkButton(tab_1, text="Select Folder", command=lambda: self.select_specific_directory(self.meteor_dir_entry))
        self.meteor_dir_button.pack(pady=(2, 10), padx=10, fill="x")

        ctk.CTkLabel(tab_1, text="Not Shooting Star Folder", font=ctk.CTkFont(weight="bold")).pack(pady=(10, 2), padx=10, anchor="w")
        self.not_meteor_dir_entry = ctk.CTkEntry(tab_1, placeholder_text="Select non-shooting star image folder")
        self.not_meteor_dir_entry.pack(pady=2, padx=10, fill="x")
        self.not_meteor_dir_entry.insert(0, "D:/training/test_sample/not_meteor")
        self.not_meteor_dir_button = ctk.CTkButton(tab_1, text="Select Folder", command=lambda: self.select_specific_directory(self.not_meteor_dir_entry))
        self.not_meteor_dir_button.pack(pady=(2, 15), padx=10, fill="x")

        ctk.CTkLabel(tab_1, text="Number of Epochs:", font=ctk.CTkFont(weight="bold")).pack(pady=(10, 2), padx=10, anchor="w")
        self.epochs_entry = ctk.CTkEntry(tab_1)
        self.epochs_entry.insert(0, "30")
        self.epochs_entry.pack(pady=2, padx=10, fill="x")

        ctk.CTkLabel(tab_1, text="Batch Size:", font=ctk.CTkFont(weight="bold")).pack(pady=(10, 2), padx=10, anchor="w")
        self.batch_size_entry = ctk.CTkEntry(tab_1)
        self.batch_size_entry.insert(0, "8")
        self.batch_size_entry.pack(pady=2, padx=10, fill="x")

        ctk.CTkLabel(tab_1, text="Learning Rate:", font=ctk.CTkFont(weight="bold")).pack(pady=(10, 2), padx=10, anchor="w")
        self.lr_entry = ctk.CTkEntry(tab_1)
        self.lr_entry.insert(0, "1e-4")
        self.lr_entry.pack(pady=2, padx=10, fill="x")

        ctk.CTkLabel(tab_1, text="Validation Data Ratio:", font=ctk.CTkFont(weight="bold")).pack(pady=(10, 2), padx=10, anchor="w")
        self.val_split_slider = ctk.CTkSlider(tab_1, from_=0.1, to=0.4, number_of_steps=30, command=lambda v: self.val_split_label.configure(text=f"{int(v*100)}%"))
        self.val_split_slider.set(0.2)
        self.val_split_slider.pack(pady=(0, 2), padx=10, fill="x")
        self.val_split_label = ctk.CTkLabel(tab_1, text="20%")
        self.val_split_label.pack(pady=(0, 10), padx=10, anchor="e")

        ctk.CTkLabel(tab_1, text="Early Stopping", font=ctk.CTkFont(weight="bold")).pack(pady=(15, 2), padx=10, anchor="w")
        self.early_stopping_check = ctk.CTkCheckBox(tab_1, text="Enable")
        self.early_stopping_check.pack(pady=2, padx=10, anchor="w")
        self.early_stopping_check.select()

        tab_2 = self.tab_view.tab("Data Augmentation")
        self.aug_frame = ctk.CTkScrollableFrame(tab_2, label_text="Augmentation Options", label_font=ctk.CTkFont(weight="bold"))
        self.aug_frame.pack(expand=True, fill="both", padx=5, pady=5)
        self.rotate_check = ctk.CTkCheckBox(self.aug_frame, text="Random Rotation"); self.rotate_check.pack(pady=5, padx=10, anchor="w")
        self.rotate_slider = ctk.CTkSlider(self.aug_frame, from_=0, to=45, number_of_steps=45, command=lambda v: self.rotate_label.configure(text=f"{int(v)}°"))
        self.rotate_slider.set(15); self.rotate_slider.pack(pady=(0, 5), padx=10, fill="x")
        self.rotate_label = ctk.CTkLabel(self.aug_frame, text="15°"); self.rotate_label.pack(pady=(0, 10), padx=10, anchor="e")
        self.hflip_check = ctk.CTkCheckBox(self.aug_frame, text="Random Horizontal Flip"); self.hflip_check.pack(pady=5, padx=10, anchor="w")
        self.vflip_check = ctk.CTkCheckBox(self.aug_frame, text="Random Vertical Flip"); self.vflip_check.pack(pady=5, padx=10, anchor="w")
        self.jitter_check = ctk.CTkCheckBox(self.aug_frame, text="Random Color Jitter"); self.jitter_check.pack(pady=(15, 5), padx=10, anchor="w")
        ctk.CTkLabel(self.aug_frame, text="Brightness").pack(pady=(5, 0), padx=15, anchor="w")
        self.brightness_slider = ctk.CTkSlider(self.aug_frame, from_=0, to=1); self.brightness_slider.set(0.2); self.brightness_slider.pack(pady=(0, 10), padx=15, fill="x")
        ctk.CTkLabel(self.aug_frame, text="Contrast").pack(pady=(5, 0), padx=15, anchor="w")
        self.contrast_slider = ctk.CTkSlider(self.aug_frame, from_=0, to=1); self.contrast_slider.set(0.2); self.contrast_slider.pack(pady=(0, 10), padx=15, fill="x")
        ctk.CTkLabel(self.aug_frame, text="Saturation").pack(pady=(5, 0), padx=15, anchor="w")
        self.saturation_slider = ctk.CTkSlider(self.aug_frame, from_=0, to=1); self.saturation_slider.set(0.2); self.saturation_slider.pack(pady=(0, 10), padx=15, fill="x")
        ctk.CTkLabel(self.aug_frame, text="Data Augmentation Multiplier (x)", font=ctk.CTkFont(weight="bold")).pack(pady=(15, 2), padx=10, anchor="w")
        self.augment_multiplier_slider = ctk.CTkSlider(self.aug_frame, from_=1, to=10, number_of_steps=9, command=lambda v: self.augment_multiplier_value.configure(text=f"x{int(v)}"))
        self.augment_multiplier_slider.set(1)
        self.augment_multiplier_slider.pack(pady=(0, 2), padx=10, fill="x")
        self.augment_multiplier_value = ctk.CTkLabel(self.aug_frame, text="x1")
        self.augment_multiplier_value.pack(pady=(0, 10), padx=10, anchor="e")

        self.start_button = ctk.CTkButton(self.control_frame, text="Start Training", command=self.start_training)
        self.start_button.grid(row=1, column=0, padx=10, pady=10, sticky="ew")

        self.progress_bar = ctk.CTkProgressBar(self.control_frame)
        self.progress_bar.grid(row=2, column=0, padx=10, pady=(0, 5), sticky="ew")
        self.progress_bar.set(0)

        self.status_label = ctk.CTkLabel(self.control_frame, text="Idle", text_color="white")
        self.status_label.grid(row=3, column=0, padx=10, pady=10, sticky="ew")

        self.main_frame = ctk.CTkFrame(self, corner_radius=10)
        self.main_frame.grid(row=0, column=1, padx=(0, 10), pady=10, sticky="nswe")
        self.main_frame.grid_rowconfigure(1, weight=1)
        self.main_frame.grid_columnconfigure(0, weight=1)

        self.main_tab_view = ctk.CTkTabview(self.main_frame)
        self.main_tab_view.pack(expand=True, fill="both", padx=5, pady=5)
        self.main_tab_view.add("Training Summary"); self.main_tab_view.add("Image Classification Results"); self.main_tab_view.add("Training Graph"); self.main_tab_view.add("Detailed Training Data")

        summary_tab = self.main_tab_view.tab("Training Summary")
        self.summary_text = ctk.CTkTextbox(summary_tab, height=140)
        self.summary_text.pack(expand=False, fill="x", padx=10, pady=10)
        self.save_model_button = ctk.CTkButton(summary_tab, text="Save Trained Model", command=self.open_model_save_dialog, state="disabled")
        self.save_model_button.pack(pady=10, padx=10)
        self.load_model_button = ctk.CTkButton(summary_tab, text="Load Model", command=self.load_model)
        self.load_model_button.pack(pady=0, padx=10)

        classification_tab = self.main_tab_view.tab("Image Classification Results")
        self.result_scroll_frame = ctk.CTkScrollableFrame(classification_tab, label_text="Classification Results")
        self.result_scroll_frame.pack(expand=True, fill="both", padx=5, pady=5)

        graph_tab = self.main_tab_view.tab("Training Graph")
        self.fig = plt.Figure(figsize=(5, 4), dpi=100); self.ax1 = self.fig.add_subplot(111); self.ax2 = self.ax1.twinx(); self.fig.tight_layout()
        self.canvas = FigureCanvasTkAgg(self.fig, master=graph_tab); self.canvas.draw(); self.canvas.get_tk_widget().pack(side=ctk.TOP, fill=ctk.BOTH, expand=1)

        metrics_tab = self.main_tab_view.tab("Detailed Training Data")
        style = ttk.Style(); style.theme_use("default"); style.configure("Treeview", background="#2a2d2e", foreground="white", fieldbackground="#343638", bordercolor="#343638", borderwidth=0); style.map('Treeview', background=[('selected', '#22559b')]); style.configure("Treeview.Heading", background="#565b5e", foreground="white", relief="flat"); style.map("Treeview.Heading", background=[('active', '#3484F0')])
        columns = ("Epoch", "Train Loss", "Train Acc", "Val Loss", "Val Acc", "Val F1", "Time")
        self.metrics_table = ttk.Treeview(metrics_tab, columns=columns, show="headings")
        for col in columns: self.metrics_table.heading(col, text=col); self.metrics_table.column(col, width=90, anchor="center")
        self.metrics_table.pack(expand=True, fill="both", padx=10, pady=10)

        self.predict_button = ctk.CTkButton(self.main_frame, text="Classify Images", command=self.classify_images, state="disabled")
        self.predict_button.pack(pady=(0, 10), padx=10)

    def process_queue(self):
        try:
            while True:
                data = self.data_queue.get_nowait()
                if data['type'] == 'status':
                    self.status_label.configure(text=data['text'], text_color=data.get('color', 'white'))
                elif data['type'] == 'class_names':
                    self.class_names = data['data']
                elif data['type'] == 'norm_stats':
                    self.norm_stats = tuple(data['data'])
                    m, s = self.norm_stats
                    self.summary_text.insert("end", f"Normalization: mean={np.round(m,3)} std={np.round(s,3)}\n")
                elif data['type'] == 'input_resize':
                    self.input_resize = data.get('data')
                    self.summary_text.insert("end", f"Input resize policy: {self.input_resize}\n")
                elif data['type'] == 'progress':
                    d = data['data']
                    self.plot_data['epochs'].append(d['epoch'])
                    self.plot_data['train_loss'].append(d['train_loss'])
                    self.plot_data['val_loss'].append(d['val_loss'])
                    self.plot_data['train_acc'].append(d['train_acc'])
                    self.plot_data['val_acc'].append(d['val_acc'])
                    
                    if len(self.plot_data['epochs']) > 0:
                        try:
                            total_epochs = int(self.epochs_entry.get())
                        except Exception:
                            total_epochs = 1
                        self.progress_bar.set(len(self.plot_data['epochs']) / max(1, total_epochs))
                    
                    self.ax1.clear()
                    self.ax2.clear()
                    self.ax1.set_xlabel('Epoch')
                    self.ax1.set_ylabel('Loss')
                    self.ax2.set_ylabel('Accuracy')

                    epochs = self.plot_data['epochs']
                    train_loss = self.plot_data['train_loss']
                    val_loss = self.plot_data['val_loss']
                    train_acc = self.plot_data['train_acc']
                    val_acc = self.plot_data['val_acc']

                    # 各曲線の色を定義
                    color_train_loss = 'blue'
                    color_val_loss = 'red'
                    color_train_acc = 'green'
                    color_val_acc = 'purple'

                    # 損失と精度をプロット
                    self.ax1.plot(epochs, train_loss, label='Train Loss', color=color_train_loss)
                    self.ax1.plot(epochs, val_loss, label='Val Loss', color=color_val_loss)
                    self.ax2.plot(epochs, train_acc, label='Train Acc', color=color_train_acc)
                    self.ax2.plot(epochs, val_acc, label='Val Acc', color=color_val_acc)

                    # 5回移動平均線を計算してプロット
                    window_size = 5
                    if len(epochs) >= window_size:
                        # 移動平均のx軸を計算
                        ma_epochs = epochs[window_size - 1:]
                        
                        # 移動平均を計算
                        train_loss_ma = np.convolve(train_loss, np.ones(window_size)/window_size, mode='valid')
                        val_loss_ma = np.convolve(val_loss, np.ones(window_size)/window_size, mode='valid')
                        train_acc_ma = np.convolve(train_acc, np.ones(window_size)/window_size, mode='valid')
                        val_acc_ma = np.convolve(val_acc, np.ones(window_size)/window_size, mode='valid')
                        
                        # 移動平均線を半透明でプロット
                        self.ax1.plot(ma_epochs, train_loss_ma, color=color_train_loss, alpha=0.4)
                        self.ax1.plot(ma_epochs, val_loss_ma, color=color_val_loss, alpha=0.4)
                        self.ax2.plot(ma_epochs, train_acc_ma, color=color_train_acc, alpha=0.4)
                        self.ax2.plot(ma_epochs, val_acc_ma, color=color_val_acc, alpha=0.4)

                    self.ax1.legend(loc='upper left')
                    self.ax2.legend(loc='upper right')
                    self.fig.tight_layout()
                    self.canvas.draw()
                    self.update_metrics_table(d)
                elif data['type'] == 'complete':
                    self.epoch_models_cache = data.get('models', [])
                    self.status_label.configure(text=data['text'])
                    self.start_button.configure(state="normal", text="Start Training")
                    self.save_model_button.configure(state="normal" if self.epoch_models_cache else "disabled")
                    best_state = data.get('best_state_dict'); best_epoch = data.get('best_epoch')
                    if best_state is not None and self.class_names is not None:
                        try:
                            self.model = ComplexCNN(num_classes=len(self.class_names))
                            self.model.load_state_dict(best_state)
                            self.model.to(self.device); self.model.eval()
                            self.predict_button.configure(state="normal")
                            self.summary_text.insert("end", f"Best epoch (Epoch {best_epoch}) model automatically loaded.\n")
                        except Exception as e:
                            self.status_label.configure(text=f"Failed to auto-load best model: {e}", text_color="red")
        except queue.Empty:
            pass
        self.after(200, self.process_queue)

    def start_training(self):
        self.plot_data = {'epochs': [], 'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
        for i in self.metrics_table.get_children(): self.metrics_table.delete(i)
        meteor_dir = self.meteor_dir_entry.get(); not_meteor_dir = self.not_meteor_dir_entry.get()
        if not os.path.isdir(meteor_dir) or not os.path.isdir(not_meteor_dir):
            self.status_label.configure(text="Error: Please specify both folders correctly.", text_color="red"); return
        try:
            params = {
                "data_paths": {"meteor": meteor_dir, "not_meteor": not_meteor_dir},
                "epochs": int(self.epochs_entry.get()), "batch_size": int(self.batch_size_entry.get()),
                "lr": float(self.lr_entry.get()), "device": self.device,
                "augmentations": self.get_augmentation_config(), "validation_split": float(self.val_split_slider.get()),
                "augment_multiplier": int(self.augment_multiplier_slider.get()),
                "early_stopping": self.early_stopping_check.get(),
            }
        except ValueError:
            self.status_label.configure(text="Error: Parameters must be numeric.", text_color="red"); return
        self.progress_bar.set(0); self.start_button.configure(state="disabled", text="Training...")
        self.status_label.configure(text=f"Starting training... (Data augmentation x{params['augment_multiplier']})", text_color="white")
        self.train_thread = threading.Thread(target=training_worker, args=(params, self.data_queue)); self.train_thread.daemon = True; self.train_thread.start()

    def classify_images(self):
        if self.model is None:
            self.status_label.configure(text="Error: Please load a model first.", text_color="red"); return
        file_paths = filedialog.askopenfilenames(title="Select images to classify", filetypes=(("Image Files", "*.jpg *.jpeg *.png *.bmp"), ("All files", "*.*")))
        if not file_paths: return
        for widget in self.result_scroll_frame.winfo_children(): widget.destroy()
        self.status_label.configure(text=f"Classifying {len(file_paths)} images...", text_color="white"); self.update()

        if self.norm_stats is not None:
            mean, std = self.norm_stats
        else:
            mean, std = (0.027, 0.027, 0.027), (0.046, 0.046, 0.046)

        for i, file_path in enumerate(file_paths):
            try:
                pil_image = Image.open(file_path).convert("RGB")
                probabilities = tta_predict(
                    self.model,
                    pil_image,
                    self.device,
                    mean=mean,
                    std=std,
                    input_resize=self.input_resize,
                )
                confidence, predicted_idx = torch.max(probabilities, 0)

                card = ctk.CTkFrame(self.result_scroll_frame); card.pack(padx=10, pady=5, fill="x")
                img_disp = pil_image.resize((128, 128)); tk_img = ImageTk.PhotoImage(img_disp)
                img_label = ctk.CTkLabel(card, image=tk_img, text=""); img_label.image = tk_img; img_label.pack(side="left", padx=10, pady=10)
                info = ctk.CTkTextbox(card, height=100); info.pack(side="left", fill="both", expand=True, padx=10, pady=10)
                pred_class = self.class_names[predicted_idx.item()] if self.class_names else str(predicted_idx.item())
                info.insert("1.0", f"File: {os.path.basename(file_path)}\nPredicted Class: {pred_class}\nConfidence: {confidence.item():.3f}\nProbability Distribution: {probabilities.squeeze().cpu().numpy()}")
                info.configure(state="disabled"); self.progress_bar.set((i + 1) / len(file_paths)); self.update()
            except Exception as e:
                err = ctk.CTkLabel(self.result_scroll_frame, text=f"Error: Could not process {os.path.basename(file_path)} ({e})", text_color="red")
                err.pack(padx=10, pady=5, fill="x")
        self.status_label.configure(text="Classification complete.", text_color="white")

    def select_specific_directory(self, entry_widget):
        dir_path = filedialog.askdirectory(title="Select a folder")
        if dir_path: entry_widget.delete(0, 'end'); entry_widget.insert(0, dir_path)

    def get_augmentation_config(self):
        config = []
        if self.rotate_check.get(): config.append(transforms.RandomRotation(degrees=int(self.rotate_slider.get())))
        if self.hflip_check.get(): config.append(transforms.RandomHorizontalFlip())
        if self.vflip_check.get(): config.append(transforms.RandomVerticalFlip())
        if self.jitter_check.get(): config.append(transforms.ColorJitter(brightness=self.brightness_slider.get(), contrast=self.contrast_slider.get(), saturation=self.saturation_slider.get()))
        return config

    def update_metrics_table(self, data):
        values = (data['epoch'], f"{data['train_loss']:.4f}", f"{data['train_acc']:.4f}", f"{data['val_loss']:.4f}", f"{data['val_acc']:.4f}", f"{data['val_f1']:.4f}", f"{data['time']:.2f}")
        self.metrics_table.insert("", "end", values=values); self.metrics_table.yview_moveto(1)

    def open_model_save_dialog(self):
        dialog = ctk.CTkToplevel(self); dialog.title("Select Model to Save"); dialog.geometry("350x400"); dialog.transient(self); dialog.grab_set()
        ctk.CTkLabel(dialog, text="Check the models from the epochs you want to save.", font=ctk.CTkFont(size=14)).pack(pady=10)
        scroll_frame = ctk.CTkScrollableFrame(dialog); scroll_frame.pack(expand=True, fill="both", padx=10)
        vars_list = []
        for i, _ in enumerate(self.epoch_models_cache, start=1):
            var = ctk.IntVar(value=0); vars_list.append(var)
            ctk.CTkCheckBox(scroll_frame, text=f"Epoch {i}", variable=var).pack(anchor="w", padx=10, pady=5)

        def save_selected():
            try:
                selected = [i for i, v in enumerate(vars_list, start=1) if v.get() == 1]
                if not selected: self.status_label.configure(text="Error: Select at least one.", text_color="red"); return
                save_dir = filedialog.askdirectory(title="Select save destination folder")
                if not save_dir: return

                def unique_path(path):
                    if not os.path.exists(path):
                        return path
                    stem, suffix = os.path.splitext(path)
                    idx = 1
                    while True:
                        candidate = f"{stem}_{idx}{suffix}"
                        if not os.path.exists(candidate):
                            return candidate
                        idx += 1

                if self.norm_stats is not None:
                    mean, std = self.norm_stats
                else:
                    mean, std = (0.027, 0.027, 0.027), (0.046, 0.046, 0.046)

                saved_paths = []
                for i in selected:
                    fname = unique_path(os.path.join(save_dir, f"model_epoch_{i}.pth"))
                    torch.save(self.epoch_models_cache[i - 1], fname)
                    class_names = self.class_names if self.class_names else ["meteor", "not_meteor"]
                    model_catalog.save_model_metadata(
                        fname,
                        mean=mean,
                        std=std,
                        class_names=class_names,
                        input_resize=self.input_resize,
                        extra={
                            "source": "train_labeled_backup0826.py",
                            "saved_epoch": i,
                            "saved_at": datetime.now().isoformat(),
                        },
                    )
                    saved_paths.append(fname)
                if self.class_names:
                    with open(os.path.join(save_dir, "class_names.txt"), 'w') as f:
                        for name in self.class_names: f.write(name + "\n")
                self.status_label.configure(
                    text=f"Saved {len(selected)} models with metadata.",
                    text_color="cyan",
                )
                self.summary_text.insert("end", "Saved models:\n")
                for p in saved_paths:
                    self.summary_text.insert("end", f" - {p}\n")
            except Exception as e:
                self.status_label.configure(text=f"Model save error: {e}", text_color="red")
            finally:
                dialog.destroy()

        save_button = ctk.CTkButton(dialog, text="Save Selected Models", command=save_selected); save_button.pack(pady=10, padx=10, fill="x")

    def load_model(self):
        model_path = filedialog.askopenfilename(title="Select model file", filetypes=(("PyTorch Model", "*.pth"), ("All files", "*.*")))
        if not model_path: return
        try:
            state_dict = torch.load(model_path, map_location=self.device)
            class_names_path = os.path.join(os.path.dirname(model_path), "class_names.txt")
            meta = model_catalog.load_model_metadata(model_path)
            self.class_names = list(meta.get("class_names", ["meteor", "not_meteor"]))
            # Compatibility fallback for legacy models with class_names.txt only.
            if os.path.exists(class_names_path) and not os.path.exists(meta.get("metadata_path", "")):
                with open(class_names_path, 'r') as f:
                    names = [line.strip() for line in f if line.strip()]
                if names:
                    self.class_names = names

            self.norm_stats = (tuple(meta.get("mean", model_catalog.DEFAULT_MEAN)), tuple(meta.get("std", model_catalog.DEFAULT_STD)))
            self.input_resize = meta.get("input_resize")
            
            self.model = ComplexCNN(num_classes=len(self.class_names))
            
            self.model.load_state_dict(state_dict)
            self.model.to(self.device); self.model.eval()
            self.predict_button.configure(state="normal")
            self.status_label.configure(text=f"Model '{os.path.basename(model_path)}' loaded.", text_color="cyan")
            self.summary_text.insert(
                "end",
                f"Loaded model metadata: mean={np.round(self.norm_stats[0],3)} std={np.round(self.norm_stats[1],3)} resize={self.input_resize}\n",
            )
        except Exception as e:
            self.status_label.configure(text=f"Model load error: {e}", text_color="red")


class CustomImageDataset(Dataset):
    def __init__(self, image_paths_by_class, transform=None, class_names=None, target_size=None):
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.class_names = list(class_names) if class_names else sorted(image_paths_by_class.keys())
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.class_names)}
        self.target_size = tuple(target_size) if target_size is not None else None

        for cls_name in self.class_names:
            label = self.class_to_idx[cls_name]
            for img_path in image_paths_by_class.get(cls_name, []):
                self.image_paths.append(img_path)
                self.labels.append(label)

    def __len__(self): return len(self.image_paths)
    def __getitem__(self, idx):
        try:
            image = Image.open(self.image_paths[idx]).convert('RGB'); label = self.labels[idx]
            if self.transform: image = self.transform(image)
            return image, label
        except Exception as e:
            print(f"Warning: Could not process image {self.image_paths[idx]}. Error: {e}")
            if self.target_size is not None:
                w, h = self.target_size
                return torch.randn(3, int(h), int(w)), -1
            return torch.randn(3, IMAGE_SIZE, IMAGE_SIZE), -1


class TransformedSubset(Dataset):
    def __init__(self, base_dataset, indices, transform=None):
        self.base = base_dataset; self.indices = list(indices); self.transform = transform
    def __len__(self): return len(self.indices)
    def __getitem__(self, i):
        idx = self.indices[i]; img, label = self.base[idx]
        if label == -1: return img, label
        if self.transform: img = self.transform(img)
        return img, label


class RepeatDataset(Dataset):
    def __init__(self, base: Dataset, repeat: int = 1):
        self.base = base; self.repeat = max(1, int(repeat))
    def __len__(self): return len(self.base) * self.repeat
    def __getitem__(self, idx): return self.base[idx % len(self.base)]


def training_worker(params, q):
    try:
        data_paths = params["data_paths"]; epochs = params["epochs"]; batch_size = params["batch_size"]; lr = params["lr"]; device = params["device"]; augmentations = params["augmentations"]; val_split = params["validation_split"]; augment_multiplier = int(params.get("augment_multiplier", 1))
        early_stopping_enabled = params.get("early_stopping", True)
        
        # Performance optimization: Enable AMP for faster training on GPU
        use_amp = (device.type == 'cuda')
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
        set_global_seed(42)
        q.put({'type': 'status', 'text': 'Preparing dataset...'})

        selected_size, selected_paths, counts_by_size = collect_diff_images_by_target_size(
            data_paths,
            target_size=TARGET_DIFF_SIZE,
        )
        if not selected_paths or selected_size is None:
            q.put({'type': 'status', 'text': 'Error: Could not collect target-size diff images.', 'color': 'red'})
            q.put({'type': 'complete', 'text': 'Training aborted.', 'models': []})
            return

        missing_classes = [cls_name for cls_name, paths in selected_paths.items() if len(paths) == 0]
        if missing_classes:
            q.put({'type': 'status', 'text': f"Error: {selected_size[0]}x{selected_size[1]} images were not found in: {', '.join(missing_classes)}", 'color': 'red'})
            q.put({'type': 'status', 'text': f"Resolution distribution: {counts_by_size}", 'color': 'red'})
            q.put({'type': 'complete', 'text': 'Training aborted.', 'models': []})
            return

        class_names = [name for name in ("meteor", "not_meteor") if name in selected_paths]
        if not class_names:
            class_names = sorted(selected_paths.keys())

        base_dataset = CustomImageDataset(
            image_paths_by_class=selected_paths,
            transform=None,
            class_names=class_names,
            target_size=selected_size,
        )
        if len(base_dataset) == 0:
            q.put({'type': 'status', 'text': 'Error: No valid images found after resolution filtering.', 'color': 'red'})
            q.put({'type': 'complete', 'text': 'Training aborted.', 'models': []})
            return

        q.put({'type': 'status', 'text': f"Selected diff resolution (fixed): {selected_size[0]}x{selected_size[1]}"})
        q.put({'type': 'status', 'text': f"Resolution distribution: {counts_by_size}"})

        mean, std = estimate_mean_std(base_dataset)
        q.put({'type': 'norm_stats', 'data': (mean, std)})
        q.put({'type': 'input_resize', 'data': [int(selected_size[1]), int(selected_size[0])]})

        train_aug_transform = transforms.Compose(augmentations + [
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        val_aug_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

        class_names = base_dataset.class_names; q.put({'type': 'class_names', 'data': class_names})
        labels = np.array(base_dataset.labels); sss = StratifiedShuffleSplit(n_splits=1, test_size=val_split, random_state=42); train_idx, val_idx = next(sss.split(np.zeros(len(labels)), labels))
        train_dataset_base = TransformedSubset(base_dataset, train_idx, transform=train_aug_transform)
        val_dataset = TransformedSubset(base_dataset, val_idx, transform=val_aug_transform)
        train_dataset = RepeatDataset(train_dataset_base, repeat=augment_multiplier)
        q.put({'type': 'status', 'text': f'Data split: Train={len(train_dataset_base)} (x{augment_multiplier} -> {len(train_dataset)}), Validation={len(val_dataset)}'})

        train_labels_for_sampler = labels[train_idx]
        class_counts = np.bincount(train_labels_for_sampler, minlength=len(class_names))
        class_weights = 1.0 / torch.tensor(class_counts, dtype=torch.float)
        sample_weights = class_weights[train_labels_for_sampler]
        repeated_sample_weights = sample_weights.repeat(augment_multiplier)
        sampler = torch.utils.data.WeightedRandomSampler(weights=repeated_sample_weights, num_samples=len(repeated_sample_weights), replacement=True)

        workers = max(2, (os.cpu_count() or 2) // 2)
        pin = (device.type == 'cuda')
        
        g = torch.Generator(); g.manual_seed(42)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler,
                                  num_workers=workers, pin_memory=pin, persistent_workers=True,
                                  generator=g, worker_init_fn=worker_init_fn)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                                num_workers=workers, pin_memory=pin, persistent_workers=True,
                                generator=g, worker_init_fn=worker_init_fn)

        q.put({'type': 'status', 'text': 'Preparing model (ComplexCNN)...'})
        model = ComplexCNN(num_classes=len(class_names)); model = model.to(device)

        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

        from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
        warmup_epochs = max(1, epochs // 10)
        warmup = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs)
        cosine = CosineAnnealingLR(optimizer, T_max=max(1, epochs - warmup_epochs))
        scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs])

        ema = EMA(model, decay=0.999)

        best_val_loss = float('inf'); best_state_dict = None; best_epoch = 0; patience, patience_counter = 7, 0; saved_models = []

        for epoch in range(1, epochs + 1):
            epoch_start_time = time.time()
            model.train(); running_loss = 0.0; train_preds, train_truth = [], []

            for inputs, labels_batch in train_loader:
                if -1 in labels_batch: continue
                inputs, labels_batch = inputs.to(device, non_blocking=True), labels_batch.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)

                with torch.cuda.amp.autocast(enabled=use_amp):
                    if np.random.rand() < 0.5:
                        inputs_mixed, y_a, y_b, lam = mixup_data(inputs, labels_batch, alpha=0.4)
                        outputs = model(inputs_mixed)
                        loss = mixup_criterion(criterion, outputs, y_a, y_b, lam)
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels_batch)

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                ema.update(model)

                running_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs.detach(), 1)
                train_preds.extend(preds.cpu().numpy()); train_truth.extend(labels_batch.detach().cpu().numpy())

            train_loss = running_loss / len(train_dataset) if len(train_dataset) > 0 else 0
            train_acc = accuracy_score(train_truth, train_preds) if len(train_truth) > 0 else 0.0

            ema.apply_shadow(model)
            model.eval(); val_running_loss = 0.0; val_preds, val_truth = [], []
            with torch.no_grad(), torch.cuda.amp.autocast(enabled=use_amp):
                for inputs, labels_batch in val_loader:
                    if -1 in labels_batch: continue
                    inputs, labels_batch = inputs.to(device, non_blocking=True), labels_batch.to(device, non_blocking=True)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels_batch)
                    val_running_loss += loss.item() * inputs.size(0)
                    _, preds = torch.max(outputs, 1); val_preds.extend(preds.cpu().numpy()); val_truth.extend(labels_batch.cpu().numpy())
            ema.restore(model)

            val_loss = val_running_loss / len(val_dataset) if len(val_dataset) > 0 else 0.0
            val_acc = accuracy_score(val_truth, val_preds) if len(val_dataset) > 0 else 0.0
            val_f1 = f1_score(val_truth, val_preds, average='macro', zero_division=0) if len(val_dataset) > 0 else 0.0

            scheduler.step()

            saved_models.append(copy.deepcopy(model.state_dict()))
            q.put({'type': 'progress', 'data': {'epoch': epoch, 'time': time.time() - epoch_start_time, 'train_loss': train_loss, 'train_acc': train_acc, 'val_loss': val_loss, 'val_acc': val_acc, 'val_f1': val_f1}})
            q.put({'type': 'status', 'text': f'Epoch {epoch}/{epochs} complete...'})

            if val_loss < best_val_loss - 1e-6:
                best_val_loss = val_loss; best_state_dict = copy.deepcopy(model.state_dict()); best_epoch = epoch; patience_counter = 0
            else:
                if early_stopping_enabled:
                    patience_counter += 1
                    if patience_counter >= patience:
                        q.put({'type': 'status', 'text': f'Early stopping: Validation loss did not improve, stopping at epoch {epoch}.'})
                        break


        q.put({'type': 'complete', 'text': 'Training complete! Select a model to save.', 'models': saved_models, 'best_state_dict': best_state_dict, 'best_epoch': best_epoch})
    except Exception as e:
        import traceback
        error_msg = f"An error occurred during training: {e}\n{traceback.format_exc()}"
        q.put({'type': 'status', 'text': error_msg, 'color': 'red'})
        q.put({'type': 'complete', 'text': 'Training aborted.', 'models': []})

if __name__ == "__main__":
    app = ImageClassifierApp()
    app.mainloop()
