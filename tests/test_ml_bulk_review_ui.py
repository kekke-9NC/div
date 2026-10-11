import json
import time
import tkinter as tk

import pytest

import ml_bulk_review as bulk
import ml_bulk_review_ui as ui
import ml_review
import ml_training_data as data


@pytest.fixture
def reviewer(tmp_path, monkeypatch):
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("Tk display unavailable")
    root.withdraw()
    monkeypatch.setattr(ml_review.TrainingDataReviewWindow, "_show_current", lambda self: None)
    window = ml_review.TrainingDataReviewWindow(root, str(tmp_path))
    window.withdraw()
    monkeypatch.setattr(ui.messagebox, "askyesno", lambda *_args, **_kwargs: True)
    yield root, window
    root.destroy()


def sample(root, name, probability):
    path = root / "pending/not_meteor" / name
    path.mkdir(parents=True)
    (path / "metadata.json").write_text(json.dumps(dict(
        predicted_label="not_meteor", meteor_probability=probability,
        review_status="pending", detection_time="2026-10-12T23:00:00")))


def finish(root, dialog):
    deadline = time.monotonic() + 5
    while dialog.winfo_exists() and dialog.busy and time.monotonic() < deadline:
        root.update()
        time.sleep(0.001)
    assert not dialog.winfo_exists() or not dialog.busy


def test_dialog_scan_invalidation_apply_and_persistent_undo(reviewer, tmp_path):
    root, window = reviewer
    sample(tmp_path, "low", 0.1)
    sample(tmp_path, "high", 0.9)
    window.refresh()
    window._open_bulk_review()
    dialog = window.bulk_dialog
    dialog.withdraw()
    assert all(button.instate(["disabled"]) for button in dialog.actions)
    dialog._scan(); finish(root, dialog)
    assert len(dialog.selection.candidates) == 1
    assert all(not button.instate(["disabled"]) for button in dialog.actions)
    dialog.threshold.set("0.05")
    assert dialog.selection is None
    assert all(button.instate(["disabled"]) for button in dialog.actions)
    dialog.threshold.set("0.1")
    dialog._scan(); finish(root, dialog)
    dialog._apply("skip"); finish(root, dialog)
    assert len(window.skip_stack) == 1 and len(dialog.batch_records) == 1
    assert not dialog.undo_button.instate(["disabled"])
    dialog._close()
    window._open_bulk_review()
    dialog = window.bulk_dialog; dialog.withdraw()
    assert len(dialog.batch_records) == 1
    dialog._undo(); finish(root, dialog)
    assert len(data.pending_events(str(tmp_path))) == 2
    assert not window.skip_stack


def test_close_cancels_worker_and_waits_before_destroy(reviewer, tmp_path, monkeypatch):
    root, window = reviewer
    sample(tmp_path, "a", 0.1)
    window._open_bulk_review()
    dialog = window.bulk_dialog; dialog.withdraw()
    dialog._scan(); finish(root, dialog)
    def delayed_apply(_root, _selection, _action, cancel, *_args):
        assert cancel.wait(2)
        return bulk.BatchResult(cancelled=True)
    monkeypatch.setattr(bulk, "apply_selection", delayed_apply)
    dialog._apply("meteor")
    assert dialog.busy
    dialog._close()
    assert dialog.winfo_exists()
    finish(root, dialog)
    assert not dialog.winfo_exists()
    assert len(data.pending_events(str(tmp_path))) == 1
