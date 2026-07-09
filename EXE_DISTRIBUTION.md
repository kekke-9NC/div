# EXE Distribution Guide

## Goal
Distribute **one file only**:

- `MeteorDetectorBootstrap.exe`

When this EXE is launched on another PC, it will:

1. Copy bundled app payload to `%LOCALAPPDATA%\MeteorDetector\app`
2. Build Python runtime automatically (Miniforge)
3. Install dependencies automatically
4. Detect NVIDIA GPU and choose a suitable Torch variant (fallback to CPU)
5. Launch `main_gui.py`

## Build

From `div`:

```bat
build_launcher.bat
```

Output:

- `dist\MeteorDetectorBootstrap.exe`

## Runtime Behavior

Install root (default):

- `%LOCALAPPDATA%\MeteorDetector`

Important files:

- `%LOCALAPPDATA%\MeteorDetector\installer.log`
- `%LOCALAPPDATA%\MeteorDetector\install_state.json`
- `%LOCALAPPDATA%\MeteorDetector\app\app_settings.json`
- `%LOCALAPPDATA%\MeteorDetector\app\app_masks.npz`

## Optional CLI Flags

- `--portable` : install next to EXE under `portable_root`
- `--cpu` : force CPU torch
- `--force` : force reinstall/update
- `--install-only` : install but do not launch app

## Notes

- Internet connection is required on first run (runtime + pip install).
- Local LLM (Qwen) feature requires NVIDIA CUDA environment.
- Normal meteor detection workflow still runs with CPU torch.
