"""Real Tk shell checks, with detection and page contents replaced by inert frames."""
import ast
from pathlib import Path
import queue
from types import SimpleNamespace
import tkinter as tk
from tkinter import ttk
from unittest.mock import Mock

import pytest
import status_panel
import ui_theme
import ui_state


def ui_methods(filename, name, methods, namespace):
    tree = ast.parse(Path(filename).read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == name)
    cls.bases = []
    cls.body = [node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name in methods]
    exec(compile(ast.Module(body=[cls], type_ignores=[]), filename, 'exec'), namespace)
    return namespace[name]


namespace = dict(tk=tk, ttk=ttk, PanedWindow=tk.PanedWindow, ui_theme=ui_theme,
                 queue=queue, status_panel=status_panel, common=SimpleNamespace(),
                 chat_gui=SimpleNamespace(create_tab=lambda parent, **_: ttk.Frame(parent)),
                 ui_state=ui_state)
Shell = ui_methods('gui_app.py', 'App', {'setup_ui', '_select_page', '_sync_page_chrome',
    '_on_page_changed', '_set_activity_visible', '_toggle_activity', '_focus_activity'}, namespace)
Preview = ui_methods('gui_preview.py', 'PreviewMixin', {'create_info_panel'}, namespace)
Source = ui_methods('gui_source.py', 'SourceMixin', {'_refresh_execution_source',
    'set_processing_source_priority'}, namespace)


class LayoutApp(Shell, Preview, Source, tk.Tk):
    def __init__(self):
        super().__init__()
        self.progress_queue = queue.Queue()
        self.start_processing = Mock()
        self.cancel_processing = Mock()
        self.open_rtsp_live_preview = Mock()
        self.open_ml_training_review = Mock()
        self.open_meteor_folder_in_finder = Mock()
        self._update_live_preview_button_state = Mock()
        self._init_summary_log_hover_preview = Mock()
        self._create_application_menu = Mock()
        self.folder_paths = []
        self.rtsp_urls = []
        self.processing_source_priority = list(ui_state.SOURCE_PRIORITY_DEFAULT)
        self.periodic_scan_var = tk.BooleanVar(value=False)
        self.setup_ui()

    def create_empty_page(self, parent):
        return ttk.Frame(parent)

    create_usage_tab = create_source_tab = create_settings_tab = create_empty_page
    create_analysis_tab = create_advanced_settings_tab = create_empty_page


@pytest.fixture
def app():
    try:
        window = LayoutApp()
    except tk.TclError as exc:
        pytest.skip(f'Tk display unavailable: {exc}')
    yield window
    window.destroy()


@pytest.mark.parametrize('size', [(1480, 900), (1180, 720)])
def test_controls_stay_visible_on_every_page_with_activity_open_or_closed(app, size):
    width, height = size
    app.geometry(f'{width}x{height}')
    app.update()
    assert not app._activity_visible
    for visible in (True, False, True, False):
        app._set_activity_visible(visible)
        app.update()
        assert app._activity_frame.winfo_viewable() == visible
        for tab in app._page_tabs:
            app._select_page(tab)
            app.update_idletasks()
            for widget in (app.start_button, app.cancel_button, app.live_preview_button, app.progress):
                assert widget.winfo_viewable()
                assert app.winfo_rootx() <= widget.winfo_rootx()
                assert widget.winfo_rootx() + widget.winfo_width() <= app.winfo_rootx() + width
                assert app.winfo_rooty() <= widget.winfo_rooty()
                assert widget.winfo_rooty() + widget.winfo_height() <= app.winfo_rooty() + height
    app.start_button.invoke()
    app.start_processing.assert_called_once()


def test_log_shortcut_opens_activity_and_keeps_events(app):
    app.update()
    app.log_text.configure(state='normal')
    app.log_text.insert('end', 'Existing observation event')
    app.log_text.configure(state='disabled')
    app._focus_activity()
    app.update()
    assert app._activity_visible
    assert app.status_panel.notebook.select() == str(app.status_panel.log_frame)
    app._toggle_activity()
    app.update()
    assert not app._activity_visible
    app._focus_activity()
    assert 'Existing observation event' in app.log_text.get('1.0', 'end')


def test_next_input_summary_follows_the_actual_priority_selection(app):
    app._refresh_execution_source()
    assert app.execution_source_var.get() == '入力ソースを選択してください'
    app.folder_paths = ['one.mp4', 'two.mp4']
    app._refresh_execution_source()
    assert app.execution_source_var.get() == '動画・フォルダ 2件'
    app.rtsp_urls = [('rtsp://example.invalid', 30)]
    app._refresh_execution_source()
    assert app.execution_source_var.get() == 'RTSPカメラ 1件'
    app.periodic_scan_var.set(True)
    app._refresh_execution_source()
    assert app.execution_source_var.get() == '監視フォルダ'
    app.set_processing_source_priority(['folder', 'rtsp', 'periodic'])
    assert app.execution_source_var.get() == '動画・フォルダ 2件'
