"""MeteorDetector bootstrap launcher (single-EXE distribution).

This launcher is intended to be packaged as a PyInstaller onefile EXE.
It performs:
1) Extract/copy bundled app payload to a persistent directory.
2) Create Python runtime (Miniforge) if missing.
3) Install Python dependencies with GPU/CPU-aware torch selection.
4) Launch main_gui.py from the prepared runtime.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple


APP_NAME = "MeteorDetector"
APP_VERSION = "2026.02.16.7"

DEFAULT_ROOT = Path(os.getenv("LOCALAPPDATA", str(Path.home() / "AppData" / "Local"))) / APP_NAME
INSTALL_MARKER = "install_state.json"
LOG_NAME = "installer.log"

PAYLOAD_DIRNAME = "payload"
APP_DIRNAME = "app"
ENV_DIRNAME = "env"

MINIFORGE_URL = (
    "https://github.com/conda-forge/miniforge/releases/latest/download/"
    "Miniforge3-Windows-x86_64.exe"
)
MINIFORGE_SHA256 = ""  # Optional: set fixed hash when pinning installer version.

TORCH_VERSION = "2.5.1"
TORCHVISION_VERSION = "0.20.1"
TORCH_INDEX_URLS = {
    "cu128": "https://download.pytorch.org/whl/cu128",
    "cu126": "https://download.pytorch.org/whl/cu126",
    "cu124": "https://download.pytorch.org/whl/cu124",
    "cu121": "https://download.pytorch.org/whl/cu121",
    "cu118": "https://download.pytorch.org/whl/cu118",
    "cpu": "https://download.pytorch.org/whl/cpu",
}

CUDA_VARIANT_MIN_DRIVER = {
    "cu128": (570, 0),
    "cu126": (560, 0),
    "cu124": (550, 0),
    "cu121": (530, 0),
    "cu118": (520, 0),
}

GPU_FAMILY_VARIANT_PRIORITY = {
    "rtx50": ["cu128", "cu126", "cu124", "cu121", "cu118"],
    "rtx40": ["cu126", "cu124", "cu121", "cu118"],
    "rtx30": ["cu124", "cu121", "cu118"],
    "rtx20_turing": ["cu121", "cu118"],
    "gtx10_pascal": ["cu118"],
    "unknown_nvidia": ["cu124", "cu121", "cu118"],
}

PRESERVE_APP_FILES = {"app_settings.json", "app_masks.npz"}


def _message_box(title: str, text: str) -> None:
    try:
        ctypes.windll.user32.MessageBoxW(None, text, title, 0x00000000)
    except Exception:
        pass


def log(msg: str, log_file: Path) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def run(cmd, log_file: Path, env: Optional[Dict[str, str]] = None) -> None:
    log(f"RUN: {' '.join(map(str, cmd))}", log_file)
    proc = subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        line = line.rstrip()
        if not line:
            continue
        print(line)
        with log_file.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
    rc = proc.wait()
    if rc != 0:
        raise subprocess.CalledProcessError(rc, cmd)


def get_bundle_dir() -> Path:
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        return Path(sys._MEIPASS)
    return Path(__file__).resolve().parent


def download_file(url: str, dest: Path, log_file: Path, expected_sha256: Optional[str] = None) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    downloaded = 0
    attempt_errors = []

    def _escape_ps(s: str) -> str:
        return s.replace("'", "''")

    def _download_with_urllib() -> None:
        nonlocal downloaded
        from urllib.request import Request, urlopen

        req = Request(url)
        if tmp.exists():
            downloaded = tmp.stat().st_size
            req.add_header("Range", f"bytes={downloaded}-")

        with urlopen(req, timeout=30) as response:
            code = getattr(response, "status", None) or getattr(response, "code", None)
            if downloaded > 0 and code != 206:
                # Server ignored Range; restart file to avoid corruption.
                downloaded = 0
                if tmp.exists():
                    tmp.unlink(missing_ok=True)
            with tmp.open("ab") as f:
                while True:
                    chunk = response.read(1024 * 512)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)

    def _download_with_curl() -> None:
        nonlocal downloaded
        curl_exe = shutil.which("curl.exe") or shutil.which("curl")
        if not curl_exe:
            raise RuntimeError("curl is not available")
        cmd = [curl_exe, "-L", "--fail", "--retry", "5", "--retry-delay", "2", "-C", "-", "-o", str(tmp), url]
        run(cmd, log_file)
        downloaded = tmp.stat().st_size

    def _download_with_powershell() -> None:
        nonlocal downloaded
        ps_exe = shutil.which("powershell.exe") or shutil.which("powershell")
        if not ps_exe:
            raise RuntimeError("powershell is not available")
        if tmp.exists():
            tmp.unlink(missing_ok=True)
        ps = (
            "$ProgressPreference='SilentlyContinue';"
            "[Net.ServicePointManager]::SecurityProtocol=[Net.SecurityProtocolType]::Tls12;"
            f"Invoke-WebRequest -Uri '{_escape_ps(url)}' -OutFile '{_escape_ps(str(tmp))}' -UseBasicParsing"
        )
        run([ps_exe, "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", ps], log_file)
        downloaded = tmp.stat().st_size

    log(f"Downloading: {url}", log_file)
    download_methods = (
        ("curl", _download_with_curl),
        ("powershell", _download_with_powershell),
        ("urllib", _download_with_urllib),
    )
    success = False
    for method_name, method in download_methods:
        try:
            log(f"Download method: {method_name}", log_file)
            method()
            success = True
            break
        except Exception as method_error:
            attempt_errors.append(f"{method_name}: {method_error}")
            log(f"Download method failed ({method_name}): {method_error}", log_file)

    if not success:
        joined = " | ".join(attempt_errors) if attempt_errors else "unknown error"
        raise RuntimeError(f"download failed for {url}: {joined}")

    tmp.replace(dest)
    if expected_sha256:
        actual = sha256sum(dest)
        if actual.lower() != expected_sha256.lower():
            dest.unlink(missing_ok=True)
            raise RuntimeError(f"SHA256 mismatch: {dest.name}")
    log(f"Downloaded: {dest.name} ({downloaded} bytes)", log_file)


def sha256sum(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sync_payload(payload_dir: Path, app_dir: Path, log_file: Path, force_overwrite: bool) -> None:
    if not payload_dir.exists():
        raise RuntimeError(f"Payload folder not found in launcher bundle: {payload_dir}")

    copied = 0
    skipped = 0

    for src in payload_dir.rglob("*"):
        if not src.is_file():
            continue
        rel = src.relative_to(payload_dir)
        dst = app_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)

        if dst.exists():
            if rel.name in PRESERVE_APP_FILES and not force_overwrite:
                skipped += 1
                continue
            if not force_overwrite and dst.stat().st_size == src.stat().st_size:
                skipped += 1
                continue

        shutil.copy2(src, dst)
        copied += 1

    log(f"Payload sync complete: copied={copied}, skipped={skipped}", log_file)


def ensure_python_env(root: Path, log_file: Path) -> Path:
    env_dir = root / ENV_DIRNAME
    python_exe = env_dir / "python.exe"
    if python_exe.exists():
        return python_exe

    log("Python runtime not found. Installing Miniforge...", log_file)
    tmp_dir = Path(tempfile.mkdtemp(prefix="md_miniforge_"))
    installer = tmp_dir / "Miniforge3-Windows-x86_64.exe"
    download_file(MINIFORGE_URL, installer, log_file, expected_sha256=MINIFORGE_SHA256 or None)

    env_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(installer),
        "/InstallationType=JustMe",
        "/RegisterPython=0",
        "/AddToPath=0",
        "/NoShortcuts=1",
        "/S",
        f"/D={env_dir}",
    ]
    run(cmd, log_file)

    if not python_exe.exists():
        raise RuntimeError("Miniforge installation failed: python.exe not found")

    out = subprocess.check_output([str(python_exe), "-c", "import sys;print(sys.version)"], text=True).strip()
    log(f"Python runtime ready: {out}", log_file)
    return python_exe


def install_base_requirements(python_exe: Path, bundle_dir: Path, log_file: Path) -> None:
    req_file = bundle_dir / "bootstrap_requirements.txt"
    if not req_file.exists():
        raise RuntimeError(f"bootstrap_requirements.txt not found: {req_file}")

    run([str(python_exe), "-m", "pip", "install", "--no-input", "--upgrade", "pip", "setuptools", "wheel"], log_file)
    run([str(python_exe), "-m", "pip", "install", "--no-input", "-r", str(req_file)], log_file)


def parse_version_tuple(value: str) -> Optional[Tuple[int, int]]:
    try:
        parts = value.strip().split(".")
        major = int(parts[0]) if len(parts) >= 1 else 0
        minor = int(parts[1]) if len(parts) >= 2 else 0
        return major, minor
    except Exception:
        return None


def parse_compute_capability(value: str) -> Optional[Tuple[int, int]]:
    return parse_version_tuple(value)


def classify_gpu_family(name: str, cc: Optional[Tuple[int, int]]) -> str:
    lower = name.lower()

    if "rtx 50" in lower or "blackwell" in lower:
        return "rtx50"
    if "rtx 40" in lower or "ada" in lower:
        return "rtx40"
    if "rtx 30" in lower or "ampere" in lower:
        return "rtx30"
    if "rtx 20" in lower or "turing" in lower or "gtx 16" in lower:
        return "rtx20_turing"
    if "gtx 10" in lower or "pascal" in lower:
        return "gtx10_pascal"

    if cc:
        major, minor = cc
        if major >= 12:
            return "rtx50"
        if major == 8 and minor >= 9:
            return "rtx40"
        if major == 8:
            return "rtx30"
        if major == 7:
            return "rtx20_turing"
        if major == 6:
            return "gtx10_pascal"

    return "unknown_nvidia"


def detect_nvidia_gpus(log_file: Path) -> List[Dict[str, object]]:
    gpus: List[Dict[str, object]] = []

    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,driver_version,compute_cap", "--format=csv,noheader,nounits"],
            stderr=subprocess.STDOUT,
            text=True,
            timeout=8,
        )
        for line in out.splitlines():
            row = [x.strip() for x in line.split(",")]
            if len(row) < 2:
                continue
            name = row[0]
            driver_version = row[1] if len(row) >= 2 else "unknown"
            compute_cap = row[2] if len(row) >= 3 else "unknown"
            cc_tuple = parse_compute_capability(compute_cap) if compute_cap and compute_cap != "unknown" else None
            family = classify_gpu_family(name, cc_tuple)
            gpus.append(
                {
                    "name": name,
                    "driver_version": driver_version,
                    "compute_cap": compute_cap,
                    "family": family,
                }
            )
    except Exception as e:
        log(f"Detailed nvidia-smi query failed: {e}", log_file)

    if not gpus:
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "-L"],
                stderr=subprocess.STDOUT,
                text=True,
                timeout=8,
            )
            for line in out.splitlines():
                if "GPU " not in line:
                    continue
                name = line.split(":", 1)[1].split("(")[0].strip() if ":" in line else line.strip()
                family = classify_gpu_family(name, None)
                gpus.append(
                    {
                        "name": name,
                        "driver_version": "unknown",
                        "compute_cap": "unknown",
                        "family": family,
                    }
                )
        except Exception:
            pass

    if not gpus:
        system32 = Path(os.environ.get("WINDIR", "C:/Windows")) / "System32"
        if (system32 / "nvcuda.dll").exists():
            gpus.append(
                {
                    "name": "NVIDIA GPU (unknown model)",
                    "driver_version": "unknown",
                    "compute_cap": "unknown",
                    "family": "unknown_nvidia",
                }
            )

    if not gpus:
        log("NVIDIA GPU was not detected. CPU mode will be used.", log_file)
        return gpus

    for idx, gpu in enumerate(gpus, start=1):
        log(
            f"Detected GPU[{idx}]: name={gpu['name']} family={gpu['family']} "
            f"driver={gpu['driver_version']} cc={gpu['compute_cap']}",
            log_file,
        )
    return gpus


def driver_meets_requirement(driver_version: str, required: Tuple[int, int]) -> bool:
    parsed = parse_version_tuple(driver_version)
    if not parsed:
        # Unknown driver version: do not block; try fallback installation paths.
        return True
    return parsed >= required


def choose_torch_variants_for_hardware(gpus: List[Dict[str, object]], force_cpu: bool, log_file: Path) -> List[str]:
    if force_cpu:
        log("CPU mode was forced by --cpu option.", log_file)
        return ["cpu"]
    if not gpus:
        return ["cpu"]

    family_rank = {
        "unknown_nvidia": 0,
        "gtx10_pascal": 1,
        "rtx20_turing": 2,
        "rtx30": 3,
        "rtx40": 4,
        "rtx50": 5,
    }
    selected_family = max(gpus, key=lambda g: family_rank.get(str(g.get("family", "unknown_nvidia")), 0)).get(
        "family", "unknown_nvidia"
    )
    variants = list(GPU_FAMILY_VARIANT_PRIORITY.get(str(selected_family), GPU_FAMILY_VARIANT_PRIORITY["unknown_nvidia"]))

    known_drivers = [str(g.get("driver_version", "")) for g in gpus if str(g.get("driver_version", "unknown")) != "unknown"]
    min_driver = None
    if known_drivers:
        parsed = [parse_version_tuple(v) for v in known_drivers]
        parsed = [p for p in parsed if p is not None]
        if parsed:
            min_driver = min(parsed)

    if min_driver:
        filtered = []
        for v in variants:
            req = CUDA_VARIANT_MIN_DRIVER.get(v)
            if req and min_driver < req:
                log(
                    f"Skipping {v}: driver {min_driver[0]}.{min_driver[1]} is lower than required {req[0]}.{req[1]}",
                    log_file,
                )
                continue
            filtered.append(v)
        variants = filtered

    if not variants:
        variants = ["cpu"]
    elif "cpu" not in variants:
        variants.append("cpu")

    log(
        f"Selected GPU family profile: {selected_family}; torch variant order: {', '.join(variants)}",
        log_file,
    )
    return variants


def uninstall_torch(python_exe: Path, log_file: Path) -> None:
    try:
        run(
            [str(python_exe), "-m", "pip", "uninstall", "-y", "torch", "torchvision", "torchaudio"],
            log_file,
        )
    except Exception:
        pass


def verify_torch_install(python_exe: Path, log_file: Path) -> Dict[str, object]:
    code = (
        "import json, torch;"
        "print(json.dumps({'version': torch.__version__, 'cuda': bool(torch.cuda.is_available())}))"
    )
    out = subprocess.check_output([str(python_exe), "-c", code], text=True).strip()
    info = json.loads(out)
    log(f"Torch verification: version={info.get('version')} cuda={info.get('cuda')}", log_file)
    return info


def get_python_version_info(python_exe: Path) -> Dict[str, object]:
    code = (
        "import json,sys;"
        "print(json.dumps({'major':sys.version_info[0],'minor':sys.version_info[1],'micro':sys.version_info[2],'version':sys.version}))"
    )
    out = subprocess.check_output([str(python_exe), "-c", code], text=True).strip()
    return json.loads(out)


def pip_install_torch_packages(
    python_exe: Path,
    log_file: Path,
    index_url: Optional[str],
    use_extra_index: bool,
    pin_versions: bool,
) -> None:
    pkg_torch = f"torch=={TORCH_VERSION}" if pin_versions else "torch"
    pkg_tv = f"torchvision=={TORCHVISION_VERSION}" if pin_versions else "torchvision"
    cmd = [
        str(python_exe),
        "-m",
        "pip",
        "install",
        "--no-input",
        "--upgrade",
        pkg_torch,
        pkg_tv,
    ]
    if index_url:
        cmd.extend(["--extra-index-url" if use_extra_index else "--index-url", index_url])
    run(cmd, log_file)


def conda_install_torch_cpu_fallback(python_exe: Path, log_file: Path) -> Dict[str, object]:
    env_prefix = python_exe.parent
    conda_exe = env_prefix / "Scripts" / "conda.exe"
    conda_bat = env_prefix / "condabin" / "conda.bat"
    if conda_exe.exists():
        conda_cmd = [str(conda_exe)]
    elif conda_bat.exists():
        conda_cmd = ["cmd", "/c", str(conda_bat)]
    else:
        raise RuntimeError(f"conda executable not found under: {env_prefix}")

    attempts = [
        conda_cmd + [
            "install",
            "-y",
            "-p",
            str(env_prefix),
            "pytorch",
            "torchvision",
            "cpuonly",
            "-c",
            "pytorch",
            "-c",
            "conda-forge",
        ],
        conda_cmd + [
            "install",
            "-y",
            "-p",
            str(env_prefix),
            "pytorch",
            "torchvision",
            "-c",
            "pytorch",
            "-c",
            "conda-forge",
        ],
    ]

    last_error = None
    for idx, cmd in enumerate(attempts, start=1):
        try:
            log(f"Trying conda torch fallback attempt {idx}/2", log_file)
            run(cmd, log_file)
            return verify_torch_install(python_exe, log_file)
        except Exception as e:
            last_error = e
            log(f"Conda torch fallback attempt {idx} failed: {e}", log_file)

    raise RuntimeError(f"Conda torch fallback failed: {last_error}")


def install_torch_with_fallback(
    python_exe: Path,
    log_file: Path,
    variant_names: List[str],
) -> Tuple[str, Dict[str, object]]:
    py_info = get_python_version_info(python_exe)
    log(
        f"Python for torch install: {py_info.get('major')}.{py_info.get('minor')}.{py_info.get('micro')} "
        f"({py_info.get('version', '').splitlines()[0]})",
        log_file,
    )
    log(f"Torch variant install sequence: {', '.join(variant_names)}", log_file)

    errors = []
    for name in variant_names:
        index_url = TORCH_INDEX_URLS.get(name)
        if not index_url:
            log(f"Skipping unknown torch variant name: {name}", log_file)
            continue
        is_gpu = name != "cpu"
        install_plans = [
            ("pinned-index", True, False),
            ("unpinned-index", False, False),
            ("pinned-extra-index", True, True),
            ("unpinned-extra-index", False, True),
        ]
        if name == "cpu":
            install_plans.append(("unpinned-pypi", False, False))

        for plan_name, pin_versions, use_extra_index in install_plans:
            uninstall_torch(python_exe, log_file)
            try:
                log(f"Trying torch variant={name} plan={plan_name}", log_file)
                url = None if plan_name == "unpinned-pypi" else index_url
                pip_install_torch_packages(
                    python_exe=python_exe,
                    log_file=log_file,
                    index_url=url,
                    use_extra_index=use_extra_index,
                    pin_versions=pin_versions,
                )
                info = verify_torch_install(python_exe, log_file)
                if is_gpu and not info.get("cuda", False):
                    raise RuntimeError("CUDA build installed but torch.cuda.is_available() is False")
                return f"{name}:{plan_name}", info
            except Exception as e:
                err_msg = f"variant={name} plan={plan_name} error={e}"
                errors.append(err_msg)
                log(f"Torch install failed: {err_msg}", log_file)

    # Final rescue path: conda CPU install, even on non-NVIDIA systems with new Python versions.
    try:
        uninstall_torch(python_exe, log_file)
        info = conda_install_torch_cpu_fallback(python_exe, log_file)
        return "conda-cpu-fallback", info
    except Exception as e:
        errors.append(f"conda-cpu-fallback error={e}")
        log(f"Torch install failed: conda-cpu-fallback error={e}", log_file)

    raise RuntimeError(f"All torch install variants failed: {' | '.join(errors)}")


def install_optional_gpu_packages(python_exe: Path, log_file: Path, torch_variant: str) -> None:
    if not torch_variant.startswith("cu"):
        log("Skipping GPU-only packages (CPU mode).", log_file)
        return
    try:
        run([str(python_exe), "-m", "pip", "install", "--no-input", "bitsandbytes>=0.43.0"], log_file)
        log("Installed optional GPU package: bitsandbytes", log_file)
    except Exception as e:
        log(f"Warning: bitsandbytes install failed (continuing): {e}", log_file)


def write_marker(marker: Path, payload: Dict[str, object]) -> None:
    marker.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def read_marker(marker: Path) -> Dict[str, object]:
    if not marker.exists():
        return {}
    try:
        return json.loads(marker.read_text(encoding="utf-8"))
    except Exception:
        return {}


def launch_app(root: Path, log_file: Path) -> None:
    python_exe = root / ENV_DIRNAME / "python.exe"
    app_dir = root / APP_DIRNAME
    main_py = app_dir / "main_gui.py"

    if not python_exe.exists():
        raise RuntimeError(f"Python runtime missing: {python_exe}")
    if not main_py.exists():
        raise RuntimeError(f"App entrypoint missing: {main_py}")

    env = os.environ.copy()
    env.setdefault("PYTHONUTF8", "1")
    env.setdefault("HF_HOME", str(app_dir / "hf_cache"))
    env.setdefault("TRANSFORMERS_CACHE", str(app_dir / "hf_cache"))

    cmd = [str(python_exe), str(main_py)]
    log(f"Launching app: {' '.join(cmd)}", log_file)
    subprocess.Popen(cmd, cwd=str(app_dir), env=env)


def parse_args():
    p = argparse.ArgumentParser(description="MeteorDetector bootstrap launcher")
    p.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="Install root directory")
    p.add_argument("--portable", action="store_true", help="Install near EXE instead of LOCALAPPDATA")
    p.add_argument("--force", action="store_true", help="Force reinstall")
    p.add_argument("--cpu", action="store_true", help="Force CPU-only torch")
    p.add_argument("--install-only", action="store_true", help="Install only; do not launch app")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    bundle_dir = get_bundle_dir()

    root = args.root
    if args.portable:
        root = Path(sys.executable).resolve().parent / "portable_root" if getattr(sys, "frozen", False) else bundle_dir / "portable_root"
    root.mkdir(parents=True, exist_ok=True)
    log_file = root / LOG_NAME
    marker_path = root / INSTALL_MARKER

    log(f"{APP_NAME} bootstrap started", log_file)
    log(f"Bundle dir: {bundle_dir}", log_file)
    log(f"Install root: {root}", log_file)

    payload_dir = bundle_dir / PAYLOAD_DIRNAME
    app_dir = root / APP_DIRNAME
    app_dir.mkdir(parents=True, exist_ok=True)

    marker = read_marker(marker_path)
    first_install = not marker
    needs_reinstall = args.force or marker.get("app_version") != APP_VERSION

    if first_install:
        _message_box(APP_NAME, "初回起動のため環境構築を開始します。\n完了まで時間がかかる場合があります。")

    sync_payload(payload_dir, app_dir, log_file, force_overwrite=needs_reinstall)

    python_exe = ensure_python_env(root, log_file)

    if first_install or needs_reinstall or args.force:
        install_base_requirements(python_exe, bundle_dir, log_file)
        gpus = detect_nvidia_gpus(log_file)
        variant_order = choose_torch_variants_for_hardware(gpus=gpus, force_cpu=args.cpu, log_file=log_file)
        torch_variant, torch_info = install_torch_with_fallback(
            python_exe=python_exe,
            log_file=log_file,
            variant_names=variant_order,
        )
        install_optional_gpu_packages(python_exe, log_file, torch_variant)

        write_marker(
            marker_path,
            {
                "app_version": APP_VERSION,
                "installed_at": time.time(),
                "python": str(python_exe),
                "torch_variant": torch_variant,
                "torch_info": torch_info,
                "root": str(root),
            },
        )
        log("Installation/update complete", log_file)
    else:
        log("Existing installation is up-to-date. Skipping dependency install.", log_file)

    if not args.install_only:
        launch_app(root, log_file)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        root = DEFAULT_ROOT
        if "--portable" in sys.argv:
            root = Path(sys.executable).resolve().parent / "portable_root" if getattr(sys, "frozen", False) else Path(__file__).resolve().parent / "portable_root"
        err_log = root / LOG_NAME
        try:
            log(f"FATAL ERROR: {exc}", err_log)
            log("TRACEBACK START", err_log)
            log(traceback.format_exc(), err_log)
            log("TRACEBACK END", err_log)
        except Exception:
            pass
        _message_box(APP_NAME, f"起動に失敗しました。\n{exc}\n\n詳細ログ: {err_log}")
        sys.exit(1)
