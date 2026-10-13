# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path

block_cipher = None

root = Path.cwd()
pathex = [str(root)]


# Files required by runtime app (main_gui + local module deps).
PAYLOAD_FILES = [
    "main_gui.py",
    "gui_app.py",
    "rtsp_gate_workers.py",
    "event_pixels.py",
    "component_statistics.py",
    "astrometry.py",
    "auto_time_updater.py",
    "bright_area_detector.py",
    "chat_gui.py",
    "config.py",
    "coordinate_manager.py",
    "detection_preview.py",
    "distortion_correction.py",
    "download_pipeline.py",
    "file_utils.py",
    "image_processing.py",
    "lighten_blend_image.py",
    "lighten_blend_video.py",
    "local_wideangle_astrometry.py",
    "location_utils.py",
    "long_exposure_map.py",
    "meteor_angle_analysis.py",
    "meteor_sky_viewer.py",
    "model.py",
    "model_catalog.py",
    "network_copy.py",
    "noise_twin.py",
    "noise_twin_pipeline.py",
    "noise_twin_training.py",
    "noise_twin_worker.py",
    "status_panel.py",
    "sun_times.py",
    "timelapse_creator.py",
    "temporal_mean.py",
    "tracking.py",
    "train_labeled_backup0826.py",
    "ui_state.py",
    "utils.py",
    "video_creation.py",
    "video_processing.py",
    "video_processor.py",
    "icon.ico",
    "model_epoch_47.pth",
    "distortion_map_x.npy",
    "distortion_map_y.npy",
]

datas = []


def add_data(src: Path, dest_dir: str) -> None:
    if not src.exists() or not src.is_file():
        raise FileNotFoundError(f"Required file for launcher bundle not found: {src}")
    datas.append((str(src), dest_dir))


# bootstrap_launcher.py expects this file in bundle root (not payload/).
add_data(root / "bootstrap_requirements.txt", ".")

# Bundle app payload under payload/.
for rel in PAYLOAD_FILES:
    rel_path = Path(rel)
    add_data(root / rel_path, str(Path("payload") / rel_path.parent))
# GUI and detector modules are split over multiple source files. Include the
# complete local Python dependency set, including additions after this list.
for module in sorted(root.glob("*.py")):
    if module.name not in PAYLOAD_FILES:
        add_data(module,"payload")
for module in sorted((root / "obs_meteor_overlay").rglob("*.py")):
    add_data(module,str(Path("payload") / module.relative_to(root).parent))
add_data(root / "assets" / "constellations.lines.json", "payload/assets")
add_data(root / "assets" / "component_statistics.c", "payload/assets")
# Bootstrap runs the extracted source in its managed Python environment;
# ship the accelerator there so end users need no C compiler.
import tempfile
import subprocess
from component_statistics import build_library, _library_name
native_build = tempfile.TemporaryDirectory(prefix="meteor_launcher_native_")
try:
    native_library = build_library(Path(native_build.name) / _library_name())
    add_data(native_library,"payload/assets")
except (OSError,RuntimeError,subprocess.SubprocessError) as exc:
    print(f"Optional native accelerator unavailable; reference fallback retained: {exc}")
add_data(root / "THIRD_PARTY_NOTICES.md", "payload")


a = Analysis(
    ["bootstrap_launcher.py"],
    pathex=pathex,
    binaries=[],
    datas=datas,
    hiddenimports=[],
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
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name="MeteorDetectorBootstrap",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=["icon.ico"] if (root / "icon.ico").exists() else None,
)
