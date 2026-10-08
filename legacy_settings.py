"""Safe migration helpers for settings saved by the legacy macOS app."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping


LEGACY_SETTINGS_RELATIVE_PATH = Path(
    "Library", "Application Support", "MeteorDetector", "app_settings.json"
)
SETTINGS_FILENAME = "app_settings.json"
MIGRATION_MARKER = ".legacy-settings-migration.json"
BACKUP_DIRECTORY = ".legacy-settings-backup"
IMPORTED_ASSETS_DIRECTORY = ".legacy-import-assets"
ASSET_SETTING_KEYS = frozenset(
    {"detection_mask_path", "mask_path_or_status", "rtsp_fixed_pattern_path"}
)
PATH_SETTING_KEYS = frozenset(
    {
        *ASSET_SETTING_KEYS,
        "plate_solve_model_path",
        "plate_solve_wcs_path",
        "wcs_path",
        "selected_model_path",
        "custom_model_paths",
        "noise_twin_model_path",
        "folder_paths",
        "periodic_scan_directory",
        "meteor_save_path",
        "not_meteor_save_path",
        "camera_model_source",
        "plate_solve_video_path",
        "ml_training_data_root",
    }
)
GLOBAL_WCS_PATH_KEYS = frozenset({"model_path", "wcs_file", "calibration_path"})
KNOWN_ASSET_NAMES = ("app_masks.npz", "rtsp_dark_frame.npz")


class LegacySettingsError(ValueError):
    """Raised when settings cannot be safely read or migrated."""


def _default_home() -> Path:
    return Path(os.environ.get("METEOR_LEGACY_SETTINGS_HOME", str(Path.home())))


def _settings_path(root: Path) -> Path:
    return root / SETTINGS_FILENAME


def _read_settings(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise LegacySettingsError(f"設定ファイルを読み込めません: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise LegacySettingsError(f"設定ファイルのトップレベルはJSONオブジェクトである必要があります: {path}")
    return value


def _deep_merge(base: Mapping[str, Any], preferred: Mapping[str, Any]) -> dict[str, Any]:
    """Keep all keys while recursively preferring values from ``preferred``."""
    merged = dict(base)
    for key, preferred_value in preferred.items():
        base_value = merged.get(key)
        if isinstance(base_value, dict) and isinstance(preferred_value, dict):
            merged[key] = _deep_merge(base_value, preferred_value)
        else:
            merged[key] = preferred_value
    return merged


def _deep_merge_with_origins(
    base: Mapping[str, Any],
    preferred: Mapping[str, Any],
    base_origin: Path,
    preferred_origin: Path,
    prefix: tuple[str, ...] = (),
) -> tuple[dict[str, Any], dict[tuple[str, ...], Path]]:
    """Deep-merge settings and remember which source supplied each value."""
    merged: dict[str, Any] = {}
    origins: dict[tuple[str, ...], Path] = {}
    for key in base.keys() | preferred.keys():
        base_value = base.get(key)
        preferred_value = preferred.get(key)
        value_path = (*prefix, str(key))
        if key in base and key in preferred and isinstance(base_value, dict) and isinstance(preferred_value, dict):
            value, nested_origins = _deep_merge_with_origins(
                base_value,
                preferred_value,
                base_origin,
                preferred_origin,
                value_path,
            )
            merged[key] = value
            origins.update(nested_origins)
        elif key in preferred:
            merged[key] = preferred_value
            origins[value_path] = preferred_origin
        else:
            merged[key] = base_value
            origins[value_path] = base_origin
    return merged, origins


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass


def _is_within(path: Path, directory: Path) -> bool:
    try:
        path.relative_to(directory)
        return True
    except ValueError:
        return False


def _safe_candidate(directory: Path, relative: Path) -> Path | None:
    candidate = (directory / relative).resolve(strict=False)
    resolved_directory = directory.resolve(strict=False)
    return candidate if _is_within(candidate, resolved_directory) else None


def _backup_existing_root(root: Path, current_settings: Mapping[str, Any] | None) -> Path | None:
    """Snapshot root settings and root-local mask assets before an import."""
    settings_path = _settings_path(root)
    candidates: set[Path] = set()
    if settings_path.is_file():
        candidates.add(settings_path)
    for name in KNOWN_ASSET_NAMES:
        candidate = (root / name).resolve(strict=False)
        if _is_within(candidate, root.resolve(strict=False)) and candidate.is_file():
            candidates.add(candidate)
    if current_settings:
        for key in ASSET_SETTING_KEYS:
            raw = current_settings.get(key)
            if not isinstance(raw, str) or not raw.strip():
                continue
            path = Path(raw).expanduser()
            if not path.is_absolute():
                path = root / path
            resolved = path.resolve(strict=False)
            if _is_within(resolved, root.resolve(strict=False)) and resolved.is_file():
                candidates.add(resolved)
    if not candidates:
        return None

    backup_root = root / BACKUP_DIRECTORY
    backup_root.mkdir(parents=True, exist_ok=True)
    backup = backup_root / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    backup = backup.with_name(f"{backup.name}-{uuid.uuid4().hex[:8]}")
    backup.mkdir()
    root_resolved = root.resolve(strict=False)
    try:
        for source in sorted(candidates):
            relative = source.relative_to(root_resolved)
            destination = backup / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
    except Exception:
        shutil.rmtree(backup, ignore_errors=True)
        raise
    return backup


def _available_asset_destination(root: Path, relative: Path, source: Path) -> Path:
    destination = _safe_candidate(root, relative)
    if destination is None:
        relative = Path(relative.name)
        destination = _safe_candidate(root, relative)
    assert destination is not None
    if not destination.exists() or destination.samefile(source):
        return destination
    if destination.is_file() and source.is_file() and destination.stat().st_size == source.stat().st_size:
        # Avoid a duplicate when the existing root asset is already identical.
        if _files_equal(destination, source):
            return destination

    imported_root = root / IMPORTED_ASSETS_DIRECTORY
    imported_relative = Path(IMPORTED_ASSETS_DIRECTORY) / relative
    imported = _safe_candidate(root, imported_relative)
    assert imported is not None
    if not imported.exists():
        return imported
    if imported.is_file() and source.is_file() and _files_equal(imported, source):
        return imported

    suffix = source.suffix
    stem = source.name[: -len(suffix)] if suffix else source.name
    for index in range(1, 10_000):
        name = f"{stem}.legacy-{index}{suffix}"
        candidate = imported_root / name
        if not candidate.exists():
            return candidate
        if candidate.is_file() and _files_equal(candidate, source):
            return candidate
    raise LegacySettingsError(f"設定資産の安全な保存先を確保できません: {source}")


def _files_equal(left: Path, right: Path) -> bool:
    try:
        if left.stat().st_size != right.stat().st_size:
            return False
        with left.open("rb") as left_handle, right.open("rb") as right_handle:
            while True:
                left_chunk = left_handle.read(1024 * 1024)
                right_chunk = right_handle.read(1024 * 1024)
                if left_chunk != right_chunk:
                    return False
                if not left_chunk:
                    return True
    except OSError:
        return False


def _copy_asset_without_overwrite(source: Path, destination: Path) -> Path:
    if destination.exists():
        if destination.is_file() and source.is_file() and _files_equal(destination, source):
            return destination
        raise LegacySettingsError(f"設定資産の保存先が既に使用されています: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent)
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        shutil.copy2(source, temporary)
        # A hard link is an atomic no-overwrite install on the same filesystem.
        try:
            os.link(temporary, destination)
        except FileExistsError:
            if destination.is_file() and _files_equal(destination, source):
                return destination
            raise LegacySettingsError(f"設定資産の保存先が既に使用されています: {destination}")
        return destination
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _repair_asset_references(
    settings: dict[str, Any],
    root: Path,
    source_roots: Iterable[Path],
    origins: Mapping[tuple[str, ...], Path] | None = None,
) -> dict[str, Any]:
    """Repair known local paths and copy mask assets without replacing files."""
    roots: list[Path] = []
    seen: set[Path] = set()
    for directory in source_roots:
        resolved = directory.resolve(strict=False)
        if resolved not in seen:
            roots.append(resolved)
            seen.add(resolved)
    root_resolved = root.resolve(strict=False)
    roots = [item for item in roots if item != root_resolved]
    copied: dict[Path, Path] = {}

    def is_path_key(key: str | None, parents: tuple[str, ...]) -> bool:
        return key in PATH_SETTING_KEYS or (
            key in GLOBAL_WCS_PATH_KEYS and "global_wcs_info" in parents
        )

    def source_root_order(path: tuple[str, ...]) -> list[Path]:
        origin = (origins or {}).get(path)
        ordered: list[Path] = []
        if origin is not None:
            origin_resolved = origin.resolve(strict=False)
            if origin_resolved == root_resolved:
                ordered.append(root_resolved)
            elif origin_resolved in roots:
                ordered.append(origin_resolved)
        ordered.extend(item for item in roots if item not in ordered)
        if root_resolved not in ordered:
            ordered.append(root_resolved)
        return ordered

    def copy_mask_asset(source: Path, relative: Path) -> str:
        source = source.resolve(strict=False)
        if source in copied:
            return str(copied[source])
        destination = _available_asset_destination(root_resolved, relative, source)
        if destination != source and not destination.exists():
            destination = _copy_asset_without_overwrite(source, destination)
        copied[source] = destination
        return str(destination)

    def repair_path(value: str, key: str | None, parents: tuple[str, ...], path: tuple[str, ...]) -> str:
        raw_path = Path(value).expanduser()
        is_mask_asset = key in ASSET_SETTING_KEYS
        ordered_roots = source_root_order(path)

        if raw_path.is_absolute():
            absolute = raw_path.resolve(strict=False)
            if absolute.exists() and (not is_mask_asset or absolute.is_file()):
                if _is_within(absolute, root_resolved):
                    return str(absolute)
                for source_root in ordered_roots:
                    if source_root != root_resolved and _is_within(absolute, source_root):
                        relative = absolute.relative_to(source_root)
                        if is_mask_asset:
                            return copy_mask_asset(absolute, relative)
                        return value
                # Existing external assets are intentional user paths.
                return value
            relative_candidates: list[Path] = []
            for base in [root_resolved, *[item for item in ordered_roots if item != root_resolved]]:
                if _is_within(absolute, base):
                    relative_candidates.append(absolute.relative_to(base))
                    break
            if not relative_candidates:
                parts = absolute.parts
                # A multi-component suffix is specific enough to repair an
                # old installation prefix; basename-only guesses are unsafe.
                relative_candidates.extend(
                    Path(*parts[start:])
                    for start in range(1, max(1, len(parts) - 1))
                )
            for base in [root_resolved, *[item for item in ordered_roots if item != root_resolved]]:
                for relative in relative_candidates:
                    candidate = _safe_candidate(base, relative)
                    if candidate is not None and candidate.exists() and (
                        not is_mask_asset or candidate.is_file()
                    ):
                        if base == root_resolved:
                            return str(candidate)
                        if is_mask_asset:
                            return copy_mask_asset(candidate, relative)
                        return str(candidate)
            return value

        for base in ordered_roots:
            candidate = _safe_candidate(base, raw_path)
            if candidate is None or not candidate.exists() or (is_mask_asset and not candidate.is_file()):
                continue
            if base == root_resolved:
                return str(candidate)
            if is_mask_asset:
                return copy_mask_asset(candidate, raw_path)
            return str(candidate)
        return value

    def visit(value: Any, key: str | None = None, parents: tuple[str, ...] = (), path: tuple[str, ...] = ()) -> Any:
        if isinstance(value, dict):
            return {
                child_key: visit(child, child_key, (*parents, str(key)) if key is not None else parents, (*path, str(child_key)))
                for child_key, child in value.items()
            }
        if isinstance(value, list):
            return [visit(item, key, parents, path) for item in value]
        if not is_path_key(key, parents) or not isinstance(value, str) or not value.strip():
            return value
        return repair_path(value, key, parents, path)

    return {key: visit(value, key, (), (str(key),)) for key, value in settings.items()}


def _copy_preferred_default_assets(
    settings: dict[str, Any], root: Path, preferred_root: Path
) -> dict[str, Any]:
    """Bring forward conventional mask files omitted by older settings files."""
    preferred_root = preferred_root.resolve(strict=False)
    root_resolved = root.resolve(strict=False)
    copied = dict(settings)
    for filename, setting_key in (
        ("app_masks.npz", "detection_mask_path"),
        ("rtsp_dark_frame.npz", "rtsp_fixed_pattern_path"),
    ):
        source = _safe_candidate(preferred_root, Path(filename))
        if source is None or not source.is_file():
            continue
        destination = source if preferred_root == root_resolved else _available_asset_destination(
            root_resolved, Path(filename), source
        )
        if destination != source and not destination.exists():
            destination = _copy_asset_without_overwrite(source, destination)
        if setting_key == "detection_mask_path":
            legacy_value = copied.get("mask_path_or_status")
            legacy_file_present = False
            if isinstance(legacy_value, str) and legacy_value.strip():
                legacy_path = Path(legacy_value).expanduser()
                if legacy_path.is_absolute():
                    legacy_file_present = legacy_path.is_file()
                else:
                    legacy_file_present = any(
                        (_safe_candidate(directory, legacy_path) or Path("/nonexistent")).is_file()
                        for directory in (preferred_root, root_resolved)
                    )
            if not copied.get(setting_key) and not legacy_file_present:
                copied[setting_key] = str(destination)
        elif not copied.get(setting_key):
            copied[setting_key] = str(destination)
    return copied


def _write_marker(root: Path, origin: str) -> None:
    marker = {
        "version": 1,
        "origin": origin,
        "completedAt": datetime.now(timezone.utc).isoformat(),
    }
    _atomic_write_json(root / MIGRATION_MARKER, marker)


def _auto_migrate(root: Path, home: Path) -> None:
    marker = root / MIGRATION_MARKER
    if marker.exists():
        return

    root_path = _settings_path(root)
    legacy_path = home / LEGACY_SETTINGS_RELATIVE_PATH
    root_exists = root_path.exists()
    legacy_exists = legacy_path.exists()
    if not root_exists and not legacy_exists:
        return

    # Parse every present source before modifying anything. A malformed file
    # must not silently cause defaults or another source to replace it.
    root_settings = _read_settings(root_path) if root_exists else {}
    legacy_settings = _read_settings(legacy_path) if legacy_exists else {}

    if root_exists and legacy_exists:
        root_mtime = root_path.stat().st_mtime_ns
        legacy_mtime = legacy_path.stat().st_mtime_ns
        if legacy_mtime > root_mtime:
            preferred, fallback = legacy_settings, root_settings
            preferred_path, fallback_path = legacy_path, root_path
        else:
            preferred, fallback = root_settings, legacy_settings
            preferred_path, fallback_path = root_path, legacy_path
        merged, origins = _deep_merge_with_origins(
            fallback,
            preferred,
            fallback_path.parent,
            preferred_path.parent,
        )
        origin = str(preferred_path)
        source_roots = [preferred_path.parent, fallback_path.parent, root]
    elif legacy_exists:
        merged = legacy_settings
        origins = {}
        preferred_path = legacy_path
        origin = str(legacy_path)
        source_roots = [legacy_path.parent, root]
        for key in merged:
            origins[(str(key),)] = legacy_path.parent
    else:
        merged = root_settings
        origins = {}
        preferred_path = root_path
        origin = str(root_path)
        source_roots = [root]
        for key in merged:
            origins[(str(key),)] = root

    # Snapshot first. Asset copy plans are no-overwrite, so root-local copies
    # remain intact even when a legacy asset has the same filename.
    _backup_existing_root(root, root_settings if root_exists else None)
    merged = _copy_preferred_default_assets(merged, root, preferred_path.parent)
    repaired = _repair_asset_references(merged, root, source_roots, origins)
    _atomic_write_json(root_path, repaired)
    _write_marker(root, origin)


def load_settings(root: Path, *, home: Path | None = None) -> dict[str, Any]:
    """Load root settings, importing the legacy macOS settings once if needed."""
    root = Path(root).expanduser().resolve(strict=False)
    _auto_migrate(root, Path(home).expanduser() if home is not None else _default_home())
    path = _settings_path(root)
    return _read_settings(path) if path.exists() else {}


def import_settings(root: Path, source: Path, *, home: Path | None = None) -> dict[str, Any]:
    """Explicitly import a settings JSON, preferring its values on conflicts."""
    del home  # Kept parallel with load_settings for injectable call sites/tests.
    root = Path(root).expanduser().resolve(strict=False)
    source = Path(source).expanduser().resolve(strict=False)
    imported = _read_settings(source)
    root_path = _settings_path(root)
    root_was_valid = True
    if root_path.exists():
        try:
            current = _read_settings(root_path)
        except LegacySettingsError:
            # An explicit, valid user-selected source is the recovery path for
            # corrupt root JSON. Preserve the raw bytes in the backup first.
            current = {}
            root_was_valid = False
    else:
        current = {}
    merged, origins = _deep_merge_with_origins(
        current,
        imported,
        root,
        source.parent,
    )
    _backup_existing_root(root, current if root_path.exists() and root_was_valid else None)
    merged = _copy_preferred_default_assets(merged, root, source.parent)
    repaired = _repair_asset_references(merged, root, [source.parent, root], origins)
    _atomic_write_json(root_path, repaired)
    _write_marker(root, str(source))
    return repaired


def save_settings(root: Path, updates: Mapping[str, Any]) -> dict[str, Any]:
    """Merge updates into root settings and preserve unknown top-level/nested keys."""
    if not isinstance(updates, Mapping):
        raise LegacySettingsError("設定更新はオブジェクトである必要があります")
    root = Path(root).expanduser().resolve(strict=False)
    path = _settings_path(root)
    current = _read_settings(path) if path.exists() else {}
    merged = _deep_merge(current, updates)
    _atomic_write_json(path, merged)
    return merged
