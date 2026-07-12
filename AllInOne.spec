# -*- mode: python ; coding: utf-8 -*-

import os
from pathlib import Path
from glob import glob

block_cipher = None

root = Path.cwd()
pathex = [str(root)]

# Utility to include files/folders recursively

def include(patterns, dest="."):
    out = []
    for pat in patterns:
        for p in glob(str(root / pat), recursive=True):
            out.append((p, dest))
    return out

# Data assets to fully bundle (models, masks, quantized LLM, configs)
data_files = []
data_files += include([
    "app_masks.npz",
    "distortion_map_x.npy",
    "distortion_map_y.npy",
    "icon.ico",
    "app_settings.json",
    "custom_coordinates.json",
    "config.py",
    "*.json",
    "*.npz",
    "*.npy",
    "assets/**/*.json",
    "THIRD_PARTY_NOTICES.md",
])
# Models
data_files += include([
    "model_latest_1.pth",
    "model_epoch_46.pth",
    "model_ep47_vacc91.60.pth",
    "model_ep48_vacc97.46.pth",
    "single_input_model.pth",
])
# Quantized LLM folder
for p in (root / "quantized_model").rglob("*"):
    if p.is_file():
        rel = p.relative_to(root)
        data_files.append((str(p), str(rel.parent)))

# GUI resources and other assets
data_files += include(["gui_resources/**/*"], ".")

a = Analysis(
    ['main_gui.py'],
    pathex=pathex,
    binaries=[],
    datas=data_files,
    hiddenimports=[
        'tkinterdnd2', 'PIL', 'cv2', 'numpy', 'astropy', 'torch', 'torchvision',
        'status_panel', 'ui_state', 'network_copy', 'download_pipeline', 'meteor_sky_viewer',
        'coordinate_manager', 'config', 'file_utils', 'video_processing', 'astrometry',
        'image_processing', 'model', 'utils', 'location_utils', 'sun_times', 'auto_time_updater',
        'long_exposure_map', 'distortion_correction', 'meteor_angle_analysis',
        'lighten_blend_video', 'lighten_blend_image', 'requests', 'camera_control', 'gui_camera_control', 'gui_dialogs', 'gui_synthesis', 'gui_live_preview', 'gui_tools', 'gui_masks', 'gui_processing', 'gui_plate_solve', 'gui_preview', 'gui_advanced', 'gui_settings', 'gui_analysis', 'gui_source', 'gui_navigation', 'gui_common', 'gui_usage', 'timelapse_creator',
        'scipy.special.cython_special', 'tkinterdnd2.TkinterDnD'
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='MeteorDetectorAllInOne',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=['icon.ico'] if os.path.exists('icon.ico') else None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='MeteorDetectorAllInOne'
)
