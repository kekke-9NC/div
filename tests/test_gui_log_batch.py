"""GUI polling checks without a window, models or a camera connection."""
import ast
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
import queue
import threading
import time
from unittest.mock import Mock


def mixin(filename, class_name, methods):
    module = ast.parse(Path(filename).read_text())
    cls = next(node for node in module.body if isinstance(node, ast.ClassDef) and node.name == class_name)
    cls.body = [node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name in methods]
    namespace = dict(threading=threading, datetime=datetime, tk=SimpleNamespace(END='end', DISABLED='disabled'), queue=queue, time=time)
    exec(compile(ast.Module(body=[cls], type_ignores=[]), filename, 'exec'), namespace)
    return namespace[class_name]


Preview = mixin('gui_preview.py', 'PreviewMixin', {'append_log', 'append_logs'})
Processing = mixin('gui_processing.py', 'ProcessingMixin', {'update_progress'})


class App(Processing, Preview):
    def __init__(self):
        self.log_text = Mock()
        self.log_text.index.return_value = '1.0'
        self._summary_log_line_map = {}
        self._extract_summary_video_ref = lambda msg: msg if 'summary' in msg else None
        self._resolve_summary_video_path = lambda ref: '/copy/' + ref
        self.progress_queue = queue.Queue()
        self.start_time_gui = None
        self.status_panel = Mock()
        self.after = Mock()


def test_batch_keeps_all_text_and_summary_line_offsets():
    app = App()
    app.append_logs(['first\nsecond', 'summary.mp4', 'last'])
    app.log_text.insert.assert_called_once()
    text = app.log_text.insert.call_args.args[1]
    assert 'first\nsecond' in text and 'summary.mp4' in text and 'last' in text
    app.log_text.see.assert_called_once_with('end')
    app.log_text.tag_add.assert_called_once_with('summary_hover', '3.0', '3.end')
    assert app._log_line_count == 4
    assert list(app._summary_log_line_map) == ['3']
    assert app.log_text.config.call_args_list[-1].kwargs == {'state': 'disabled'}


def test_large_batch_trims_blocks_and_shifts_retained_summary_map():
    app = App()
    app._log_line_count = 4999
    app.log_text.index.return_value = '5000.0'
    app._summary_log_line_map = {'2': {'old': True}, '4998': {'retained': True}}
    app.append_logs(['summary.mp4'] + ['next'] * 1200)
    app.log_text.delete.assert_called_once_with('1.0', '1501.0')
    assert app._log_line_count == 4700
    assert app._summary_log_line_map == {'3498': {'retained': True}, '3500': {'summary_ref': 'summary.mp4', 'resolved_path': '/copy/summary.mp4'}}


def test_poll_batches_logs_coalesces_status_and_leaves_bounded_backlog():
    app = App()
    for i in range(130):
        app.progress_queue.put((f'log {i}', None))
    app.update_progress()
    assert app.progress_queue.qsize() == 10
    app.log_text.insert.assert_called_once()
    app.log_text.see.assert_called_once()
    assert app._pending_log_batch is None
    app.progress_queue.put((None, {'pipeline_status': {'pending_sources': 1}}))
    app.progress_queue.put((None, {'pipeline_status': {'pending_sources': 2}}))
    app.update_progress()
    app.status_panel.update_status.assert_called_once_with({'pending_sources': 2})
    app.after.assert_called_with(100, app.update_progress)


def test_worker_log_never_calls_tk():
    app = App()
    worker = threading.Thread(target=app.append_log, args=('worker',))
    worker.start(); worker.join()
    assert app.progress_queue.get_nowait() == ('worker', None)
    app.log_text.config.assert_not_called()
