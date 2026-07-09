import os
import shutil
import tempfile
from pathlib import Path
import ctypes
from typing import Tuple, Optional


class CancelledCopy(Exception):
    """Raised when a file copy is cancelled via cancel_flag."""
    pass


def _ensure_trailing_sep(root: str) -> str:
    if not root.endswith('\\') and not root.endswith('/'):
        return root + ('\\' if os.name == 'nt' else '/')
    return root


def is_remote_path(path: str) -> bool:
    """Return True if the given path refers to a remote/network drive (Windows).

    Uses GetDriveTypeW to detect DRIVE_REMOTE (4). For UNC paths the anchor
    returned by Path(path).anchor is used (e.g. \\\\server\\share\\).
    On non-Windows platforms this returns False.
    """
    try:
        if os.name != 'nt':
            return False
        p = Path(path)
        root = p.anchor
        if not root:
            # fallback: use drive from splitdrive
            drive, _ = os.path.splitdrive(path)
            root = drive + '\\' if drive else ''
        if not root:
            return False
        root = _ensure_trailing_sep(root)
        kernel32 = ctypes.windll.kernel32
        GetDriveTypeW = kernel32.GetDriveTypeW
        GetDriveTypeW.argtypes = [ctypes.c_wchar_p]
        dtype = GetDriveTypeW(root)
        # DRIVE_REMOTE == 4
        return int(dtype) == 4
    except Exception:
        # On error, conservatively assume local to avoid unnecessary copies
        return False


def ensure_local_copy(
    path: str,
    tmp_root: Optional[str] = None,
    cancel_flag: Optional[object] = None,
    chunk_size: int = 8 * 1024 * 1024,
) -> Tuple[str, Optional[str]]:
    """Ensure the given file path is available locally.

    If the path is on a remote drive (Windows), copy it to a temporary directory
    and return (local_path, tmp_dir). If no copy was needed, returns (path, None).

    tmp_root: optional directory under which to create the temp dir.
    cancel_flag: optional threading.Event-like object with .is_set() to cooperatively
                 cancel an in-progress copy.
    chunk_size: size of chunks (bytes) when copying to allow timely cancellation.
    """
    path = str(path)
    if not is_remote_path(path):
        return path, None

    # create a temporary directory
    if tmp_root:
        os.makedirs(tmp_root, exist_ok=True)
        tmp_dir = tempfile.mkdtemp(prefix='netcopy_', dir=tmp_root)
    else:
        tmp_dir = tempfile.mkdtemp(prefix='netcopy_')

    try:
        # Preserve original directory hierarchy under tmp_dir so downstream
        # code that extracts metadata from the path (e.g. date/time from the
        # path components) can still work.
        drive, tail = os.path.splitdrive(path)
        # tail may start with a separator; remove leading slashes/backslashes
        tail = tail.lstrip('\\/')
        parts = tail.split(os.sep) if tail else [os.path.basename(path)]

        # sanitize drive/root to a usable folder name
        def _sanitize_drive(d: str) -> str:
            if not d:
                return 'local'
            return d.replace(':', '').replace('\\', '_').replace('/', '_').strip('_')

        drive_name = _sanitize_drive(drive)
        dest_dir = os.path.join(tmp_dir, drive_name, *parts[:-1]) if parts[:-1] else os.path.join(tmp_dir, drive_name)
        os.makedirs(dest_dir, exist_ok=True)
        local_copy = os.path.join(dest_dir, parts[-1])

        # Copy in chunks to allow cancellation during long copies
        src_f = dst_f = None
        try:
            if cancel_flag is not None and getattr(cancel_flag, 'is_set', None) and cancel_flag.is_set():
                raise CancelledCopy("copy cancelled before start")
            src_f = open(path, 'rb')
            dst_f = open(local_copy, 'wb')
            while True:
                if cancel_flag is not None and getattr(cancel_flag, 'is_set', None) and cancel_flag.is_set():
                    raise CancelledCopy("copy cancelled")
                buf = src_f.read(chunk_size)
                if not buf:
                    break
                dst_f.write(buf)
            # flush/close before copying metadata
            dst_f.flush()
            os.fsync(dst_f.fileno()) if hasattr(os, 'fsync') else None
        finally:
            try:
                if src_f:
                    src_f.close()
            except Exception:
                pass
            try:
                if dst_f:
                    dst_f.close()
            except Exception:
                pass

        # If we reached here without cancellation, copy metadata like copy2 does
        shutil.copystat(path, local_copy, follow_symlinks=True)
        return local_copy, tmp_dir
    except CancelledCopy:
        # Remove partial file and temp dir on cancellation
        try:
            if 'local_copy' in locals() and os.path.exists(local_copy):
                os.remove(local_copy)
        except Exception:
            pass
        try:
            if os.path.exists(tmp_dir):
                shutil.rmtree(tmp_dir)
        except Exception:
            pass
        raise
    except Exception:
        # cleanup on failure
        try:
            shutil.rmtree(tmp_dir)
        except Exception:
            pass
        raise


def cleanup_tempdir(tmp_dir: Optional[str]) -> None:
    if not tmp_dir:
        return
    try:
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir)
    except Exception:
        pass
