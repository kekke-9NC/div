import json
import os
from pathlib import Path

import pytest

from legacy_settings import (
    BACKUP_DIRECTORY,
    IMPORTED_ASSETS_DIRECTORY,
    LEGACY_SETTINGS_RELATIVE_PATH,
    MIGRATION_MARKER,
    LegacySettingsError,
    import_settings,
    load_settings,
    save_settings,
)


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def legacy_settings_path(home: Path) -> Path:
    return home / LEGACY_SETTINGS_RELATIVE_PATH


def test_load_migrates_legacy_settings_once_and_keeps_unknown_values(tmp_path):
    home = tmp_path / "test-home"
    root = tmp_path / "app-root"
    source = legacy_settings_path(home)
    write_json(source, {"legacy_option": {"future": True}, "interval": 2.5})

    loaded = load_settings(root, home=home)
    assert loaded == {"legacy_option": {"future": True}, "interval": 2.5}
    assert (root / MIGRATION_MARKER).is_file()

    write_json(source, {"interval": 9, "new_legacy_key": "later"})
    loaded_again = load_settings(root, home=home)
    assert loaded_again == {"legacy_option": {"future": True}, "interval": 2.5}


def test_newer_source_wins_conflicts_but_missing_nested_keys_and_root_are_kept(tmp_path):
    home = tmp_path / "isolated-home"
    root = tmp_path / "app-root"
    legacy = legacy_settings_path(home)
    root_settings = root / "app_settings.json"
    write_json(legacy, {"shared": {"old": 1, "conflict": "legacy"}, "legacy_only": 2})
    write_json(root_settings, {"shared": {"root_only": 3, "conflict": "root"}, "root_only": 4})
    os.utime(root_settings, ns=(1_000_000_000, 1_000_000_000))
    os.utime(legacy, ns=(2_000_000_000, 2_000_000_000))

    loaded = load_settings(root, home=home)
    assert loaded == {
        "shared": {"old": 1, "root_only": 3, "conflict": "legacy"},
        "legacy_only": 2,
        "root_only": 4,
    }
    backups = list((root / BACKUP_DIRECTORY).glob("*/app_settings.json"))
    assert len(backups) == 1
    assert json.loads(backups[0].read_text(encoding="utf-8"))["root_only"] == 4


def test_asset_collision_preserves_both_assets_and_points_to_imported_copy(tmp_path):
    home = tmp_path / "home"
    root = tmp_path / "root"
    legacy = legacy_settings_path(home)
    root.mkdir()
    (legacy.parent).mkdir(parents=True)
    source_asset = legacy.parent / "app_masks.npz"
    source_asset.write_bytes(b"legacy-mask")
    root_asset = root / "app_masks.npz"
    root_asset.write_bytes(b"root-mask")
    write_json(legacy, {"detection_mask_path": "app_masks.npz"})

    loaded = load_settings(root, home=home)

    imported_asset = root / IMPORTED_ASSETS_DIRECTORY / "app_masks.npz"
    assert loaded["detection_mask_path"] == str(imported_asset)
    assert imported_asset.read_bytes() == b"legacy-mask"
    assert source_asset.read_bytes() == b"legacy-mask"
    assert root_asset.read_bytes() == b"root-mask"
    backup_asset = next((root / BACKUP_DIRECTORY).glob("*/app_masks.npz"))
    assert backup_asset.read_bytes() == b"root-mask"


def test_explicit_import_prefers_source_and_deep_merges_unknown_nested_keys(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    source = tmp_path / "chosen" / "app_settings.json"
    write_json(root / "app_settings.json", {"nested": {"root_only": 1, "same": "root"}})
    write_json(source, {"nested": {"source_only": 2, "same": "source"}, "future": {"flag": True}})

    imported = import_settings(root, source)
    assert imported == {
        "nested": {"root_only": 1, "source_only": 2, "same": "source"},
        "future": {"flag": True},
    }
    assert list((root / BACKUP_DIRECTORY).glob("*/app_settings.json"))
    assert (root / MIGRATION_MARKER).is_file()


def test_invalid_legacy_json_does_not_fall_back_to_root_or_write_marker(tmp_path):
    home = tmp_path / "home"
    root = tmp_path / "root"
    root.mkdir()
    root_before = {"known": "root"}
    write_json(root / "app_settings.json", root_before)
    legacy = legacy_settings_path(home)
    legacy.parent.mkdir(parents=True)
    legacy.write_text("{broken", encoding="utf-8")

    with pytest.raises(LegacySettingsError):
        load_settings(root, home=home)

    assert json.loads((root / "app_settings.json").read_text(encoding="utf-8")) == root_before
    assert not (root / MIGRATION_MARKER).exists()


def test_invalid_explicit_import_source_leaves_existing_root_untouched(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    root_settings = root / "app_settings.json"
    write_json(root_settings, {"kept": {"value": 1}})
    before = root_settings.read_bytes()
    source = tmp_path / "invalid.json"
    source.write_text("[]", encoding="utf-8")

    with pytest.raises(LegacySettingsError):
        import_settings(root, source)

    assert root_settings.read_bytes() == before


def test_explicit_import_can_replace_corrupt_root_after_backing_up_raw_json(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    root_settings = root / "app_settings.json"
    root_settings.write_bytes(b"{broken root json")
    source = tmp_path / "selected" / "app_settings.json"
    write_json(source, {"recovered": {"valid": True}})

    imported = import_settings(root, source)

    assert imported == {"recovered": {"valid": True}}
    assert json.loads(root_settings.read_text(encoding="utf-8")) == imported
    backup = next((root / BACKUP_DIRECTORY).glob("*/app_settings.json"))
    assert backup.read_bytes() == b"{broken root json"
    assert (root / MIGRATION_MARKER).is_file()


def test_save_settings_preserves_existing_unknown_nested_keys(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    write_json(root / "app_settings.json", {"future": {"nested": 7}, "known": {"keep": True, "replace": 1}})

    saved = save_settings(root, {"known": {"replace": 2, "added": 3}})
    assert saved == {"future": {"nested": 7}, "known": {"keep": True, "replace": 2, "added": 3}}


def test_default_mask_and_fixed_pattern_assets_migrate_when_legacy_keys_are_missing(tmp_path):
    home = tmp_path / "home"
    root = tmp_path / "root"
    legacy = legacy_settings_path(home)
    legacy.parent.mkdir(parents=True)
    (legacy.parent / "app_masks.npz").write_bytes(b"mask")
    (legacy.parent / "rtsp_dark_frame.npz").write_bytes(b"fixed-pattern")
    write_json(legacy, {"interval": 3})

    loaded = load_settings(root, home=home)

    assert Path(loaded["detection_mask_path"]).read_bytes() == b"mask"
    assert Path(loaded["rtsp_fixed_pattern_path"]).read_bytes() == b"fixed-pattern"
    assert (legacy.parent / "app_masks.npz").read_bytes() == b"mask"


def test_path_origin_selects_the_asset_from_the_setting_value_source(tmp_path):
    home = tmp_path / "home"
    root = tmp_path / "root"
    legacy = legacy_settings_path(home)
    legacy.parent.mkdir(parents=True)
    root.mkdir()
    (legacy.parent / "app_masks.npz").write_bytes(b"legacy-mask")
    (root / "app_masks.npz").write_bytes(b"root-mask")
    write_json(legacy, {"detection_mask_path": "app_masks.npz"})
    write_json(root / "app_settings.json", {"root_only": True})
    # Root settings are newer overall, but this individual setting comes from
    # the older legacy file and must continue to refer to its matching asset.
    os.utime(legacy, ns=(1_000_000_000, 1_000_000_000))
    os.utime(root / "app_settings.json", ns=(2_000_000_000, 2_000_000_000))

    loaded = load_settings(root, home=home)

    selected_asset = Path(loaded["detection_mask_path"])
    assert selected_asset.read_bytes() == b"legacy-mask"
    assert selected_asset != root / "app_masks.npz"
    assert (root / "app_masks.npz").read_bytes() == b"root-mask"


def test_root_setting_keeps_root_asset_when_legacy_asset_has_same_relative_name(tmp_path):
    home = tmp_path / "home"
    root = tmp_path / "root"
    legacy = legacy_settings_path(home)
    legacy.parent.mkdir(parents=True)
    root.mkdir()
    (legacy.parent / "app_masks.npz").write_bytes(b"legacy-mask")
    (root / "app_masks.npz").write_bytes(b"root-mask")
    write_json(legacy, {"detection_mask_path": "app_masks.npz"})
    write_json(root / "app_settings.json", {"detection_mask_path": "app_masks.npz"})
    os.utime(legacy, ns=(1_000_000_000, 1_000_000_000))
    os.utime(root / "app_settings.json", ns=(2_000_000_000, 2_000_000_000))

    loaded = load_settings(root, home=home)

    assert loaded["detection_mask_path"] == str(root / "app_masks.npz")
    assert (root / "app_masks.npz").read_bytes() == b"root-mask"
    assert (legacy.parent / "app_masks.npz").read_bytes() == b"legacy-mask"


def test_existing_external_model_path_is_preserved_and_missing_old_prefix_uses_root_suffix(tmp_path):
    home = tmp_path / "home"
    root = tmp_path / "root"
    legacy = legacy_settings_path(home)
    legacy.parent.mkdir(parents=True)
    external_model = tmp_path / "custom" / "external.pth"
    external_model.parent.mkdir()
    external_model.write_bytes(b"external")
    repaired_model = root / "models" / "detector.pth"
    repaired_model.parent.mkdir(parents=True)
    repaired_model.write_bytes(b"root-copy")
    write_json(
        legacy,
        {
            "selected_model_path": str(external_model),
            "noise_twin_model_path": "/old/installation/payload/models/detector.pth",
            "global_wcs_info": {
                "model_path": "/old/installation/payload/models/detector.pth"
            },
        },
    )

    loaded = load_settings(root, home=home)

    assert loaded["selected_model_path"] == str(external_model)
    assert loaded["noise_twin_model_path"] == str(repaired_model)
    assert loaded["global_wcs_info"]["model_path"] == str(repaired_model)


@pytest.mark.parametrize("saved_path", [
    "calibration/camera.json",
    "/old/application/payload/calibration/camera.json",
])
def test_plate_solve_wcs_path_resolves_from_source_or_relocated_root(tmp_path, saved_path):
    home = tmp_path / "home"
    root = tmp_path / "root"
    legacy = legacy_settings_path(home)
    legacy.parent.mkdir(parents=True)
    source_calibration = legacy.parent / "calibration" / "camera.json"
    source_calibration.parent.mkdir()
    source_calibration.write_text("{}", encoding="utf-8")
    if saved_path.startswith("/"):
        root_calibration = root / "calibration" / "camera.json"
        root_calibration.parent.mkdir(parents=True)
        root_calibration.write_text("{}", encoding="utf-8")
    write_json(legacy, {"plate_solve_wcs_path": saved_path})

    loaded = load_settings(root, home=home)

    expected = root / "calibration" / "camera.json" if saved_path.startswith("/") else source_calibration
    assert loaded["plate_solve_wcs_path"] == str(expected)


def test_relative_input_and_output_paths_resolve_against_settings_source(tmp_path):
    home = tmp_path / "home"
    root = tmp_path / "root"
    legacy = legacy_settings_path(home)
    legacy.parent.mkdir(parents=True)
    videos = legacy.parent / "videos"
    videos.mkdir()
    sample = legacy.parent / "sample.mp4"
    sample.write_bytes(b"video")
    output = legacy.parent / "results"
    output.mkdir()
    write_json(
        legacy,
        {
            "folder_paths": ["videos"],
            "camera_model_source": "sample.mp4",
            "periodic_scan_directory": "videos",
            "meteor_save_path": "results",
        },
    )

    loaded = load_settings(root, home=home)

    assert loaded["folder_paths"] == [str(videos)]
    assert loaded["camera_model_source"] == str(sample)
    assert loaded["periodic_scan_directory"] == str(videos)
    assert loaded["meteor_save_path"] == str(output)


def test_invalid_root_json_does_not_fall_back_to_valid_legacy_source(tmp_path):
    home = tmp_path / "home"
    root = tmp_path / "root"
    root.mkdir()
    (root / "app_settings.json").write_text("{broken", encoding="utf-8")
    write_json(legacy_settings_path(home), {"legacy": "valid"})

    with pytest.raises(LegacySettingsError):
        load_settings(root, home=home)

    assert not (root / MIGRATION_MARKER).exists()
