import ast
from pathlib import Path
from unittest.mock import Mock


def app_class():
    tree=ast.parse(Path('gui_app.py').read_text())
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='App')
    cls.bases=[]
    cls.body=[n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name in ('_begin_graceful_shutdown','_wait_for_recording_shutdown')]
    scope={}
    exec(compile(ast.Module(body=[cls],type_ignores=[]),'gui_app.py','exec'),scope)
    return scope['App']


def test_shutdown_cancels_once_waits_for_workers_then_finishes():
    cls=app_class();app=cls();app.cancel_processing=Mock();app._finish_closing=Mock();app.after=Mock()
    app.rtsp_thread=Mock();app.rtsp_thread.is_alive.return_value=True
    app.worker_thread=None;app.periodic_scan_thread=None
    app._begin_graceful_shutdown()
    app.cancel_processing.assert_called_once()
    app._finish_closing.assert_not_called()
    app.after.assert_called_once_with(100,app._wait_for_recording_shutdown)
    app._begin_graceful_shutdown()
    app.cancel_processing.assert_called_once()
    app.rtsp_thread.is_alive.return_value=False
    app._wait_for_recording_shutdown()
    app._finish_closing.assert_called_once()
