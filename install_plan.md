# MeteorDetector Installer Plan

Goal: ship a tiny bootstrap EXE that downloads/installs everything needed at first run, keeping the downloadable EXE small while reliably setting up the full app.

## High-level approach
- Build a onefile **bootstrap launcher** with PyInstaller (Windows-only) that:
  1) Creates an app home (default `%LOCALAPPDATA%/MeteorDetector` or alongside the EXE if portable mode is requested).
  2) Ensures a Python runtime exists (prefer bundled Miniconda/miniforge ZIP; fallback to downloading if missing; allow offline seed ZIP).
  3) Installs Python packages via `pip` using `div/requirements.txt` (GPU/CPU variants selectable for torch).
  4) Downloads required assets (models, masks, tokenizer, configs) with checksum verification and resume support.
  5) Writes shortcuts/config markers; then launches `main_gui.py` (or the already-built onedir) with the prepared environment.
- Keep the bootstrap under ~50-80 MB by not bundling torch/LLM weights.
- Provide an **asset manifest** (JSON) listing URLs + SHA256 + target paths to enable updates and integrity checks.

## Components to add
- `bootstrap_launcher.py`: orchestrates install; can also run in "install-only" or "launch" mode.
- `Launcher.spec`: PyInstaller onefile spec for the bootstrap.
- `assets_manifest.json`: URLs + hashes for large files (model_latest_1.pth, tokenizer files, etc.).
- `install_plan.md` (this file) + a user-facing `INSTALL.md` describing usage and offline/online flows.
- Optional `seed_cache/` folder support for offline installs (copy ZIPs here to skip downloads).

## Runtime layout (proposal)
- `%LOCALAPPDATA%/MeteorDetector/`
  - `env/` (Miniconda/miniforge or venv)
  - `app/` (PyInstaller onedir payload OR source tree)
  - `assets/` (models, masks, tokenizer, distortion maps)
  - `logs/installer.log`
  - `.installed` (marker with versions and hashes)
- Portable mode: sibling `./portable_root/` instead of `%LOCALAPPDATA%` when `--portable` flag is passed.

## Installer flow (bootstrap logic)
1) Resolve install root (env var override > portable flag > default `%LOCALAPPDATA%`).
2) Locking: create a lockfile to avoid concurrent installs; retry/backoff.
3) Check `.installed` manifest; if versions/hashes match and not forced, skip to launch.
4) Python runtime:
   - If `env/python.exe` exists and reports correct version (3.10), reuse.
   - Else unpack bundled miniforge ZIP (preferred) or download it (URL + SHA256 in manifest).
5) Pip deps:
   - Choose torch variant: `torch+cu118` (default) or `torch+cpu` if no NVIDIA GPU/driver detected.
   - Install `requirements.txt` into the env; cache wheels in `wheels/` to speed re-runs.
6) Assets:
   - Download files listed in `assets_manifest.json` with SHA256 check and resume (HTTP Range).
   - Large items: `model_latest_1.pth` (or `model_epoch_46.pth`), tokenizer/config set under `quantized_model/`, masks `app_masks.npz`, distortion maps.
7) Post steps: write shortcuts (optional), emit `.installed` JSON (versions, hashes, timestamps).
8) Launch app: run `python app/main_gui.py` (source) or `app/MeteorDetector/main_gui.exe` if we later bundle onedir; pass `--portable` flag mapping.

## Build pipeline
- Keep existing onedir build (`div/build_exe.py` / `MeteorDetector.spec`) for producing `dist/MeteorDetector/` if we still want a self-contained runtime.
- Add `Launcher.spec` (onefile) targeting `bootstrap_launcher.py` with minimal datas (icons).
- Update `build_launcher.bat` to call the new spec; ensure PyInstaller installed in the build env.

## Open questions / decisions
- Hosting for assets and miniforge ZIP (GitHub Releases? S3? local network share?).
- Whether to ship CPU-only torch by default and offer GPU as optional download to reduce size.
- Whether to bundle the app code as source (so launcher runs `main_gui.py`) or bundle the onedir payload and let launcher only fetch models.
- Shortcut creation (Start Menu / Desktop) vs portable-only.
- Proxy/corporate network handling; provide `--no-verify` toggle? (default: verify hashes).

## Next actions
- [ ] Create `assets_manifest.json` skeleton with placeholders for URLs and SHA256.
- [ ] Implement `bootstrap_launcher.py` (logging, lock, download, env setup, asset fetch, launch).
- [ ] Add `Launcher.spec` and update `build_launcher.bat` to use it.
- [ ] Document user flow in `INSTALL.md` (online/offline, CPU/GPU choice).
