import json
from pathlib import Path
from unittest.mock import patch
import rtsp_live


def test_publication_sidecar_is_read_once_and_replacement_invalidates(tmp_path):
    path = tmp_path/'chunk_000000001.mp4'
    path.write_bytes(b'raw')
    marker = Path(str(path)+'.archived.json')
    rtsp_live.write_json(marker, {'archive': 'first.mp4'})
    archiver = rtsp_live.MinuteArchiver(tmp_path)
    original = Path.read_text
    reads = []
    def read(self, *args, **kwargs):
        reads.append(self)
        return original(self, *args, **kwargs)
    with patch.object(Path, 'read_text', read):
        assert archiver._archive_target(path) == tmp_path/'first.mp4'
        assert archiver._archive_target(path) == tmp_path/'first.mp4'
        assert reads == [marker]
        rtsp_live.write_json(marker, {'archive': 'other.mp4'})
        assert archiver._archive_target(path) == tmp_path/'other.mp4'
        assert reads == [marker, marker]
        marker.unlink()
        assert archiver._archive_target(path) is None
        assert path not in archiver.archive_targets


def test_cached_target_is_checked_again_after_external_archive_deletion(tmp_path):
    path = tmp_path/'chunk_000000001.mp4'
    path.write_bytes(b'raw')
    output = tmp_path/'minute.mp4'
    output.write_bytes(b'published')
    rtsp_live.write_json(str(path)+'.archived.json', {'archive': output.name})
    archiver = rtsp_live.MinuteArchiver(tmp_path)
    archiver.catalog.paths = lambda: [path]
    archiver._recover = lambda: None
    with patch.object(archiver, '_cleanup') as cleanup:
        archiver.compact()
        cleanup.assert_called_once_with(path)
    output.unlink()
    with patch.object(archiver, '_publish'), patch.object(rtsp_live, 'timing', return_value={'start_time':'2026-10-13T04:00:00', 'frame_count':12,'fps':25,'size':[96,64]}):
        archiver.compact(flush=True)
    assert not Path(str(path)+'.archived.json').exists()
    assert path.exists()


def test_catalog_checks_completion_metadata_once_and_reuses_csv_inventory(tmp_path):
    directory = rtsp_live.capture_directory(tmp_path)
    path = directory/'chunk_000000001.mp4'
    path.write_bytes(b'raw')
    marker = Path(str(path)+'.capture.json')
    rtsp_live.write_json(marker, {'start_time':'2026-10-13T04:00:00','frame_count':12,'fps':25,'size':[96,64]})
    listing = directory/'segments.csv'
    listing.write_text('')
    catalog = rtsp_live.ChunkCatalog(tmp_path)
    original_stat = Path.stat
    counted = []
    def stat(self, *args, **kwargs):
        counted.append(self)
        return original_stat(self, *args, **kwargs)
    with patch.object(Path,'stat',stat), patch.object(rtsp_live.rtsp_segments,'is_finalized_mp4',return_value=True):
        assert catalog.paths() == [path]
        counted.clear()
        assert catalog.paths() == [path]
        assert counted.count(marker) == 1
        assert counted.count(listing) == 0
        assert catalog.present == {path}
        marker.unlink()
        assert catalog.paths() == []
        assert catalog.present == {path}
        path.unlink()
        assert catalog.paths() == []
        assert catalog.present == set()


def test_backlog_discovery_uses_buffer_and_retains_live_tail_cadence():
    import queue, threading
    runner = rtsp_live.RealtimeRunner('unused', lambda *_:None, threading.Event())
    with patch.object(rtsp_live.config, 'RTSP_SCAN_INTERVAL', .1):
        for _ in range(100):runner.queue.put('chunk')
        assert runner._discovery_interval() == 5.0
        assert runner._archiving_interval() == 5.0
        runner.queue.get()
        assert runner._discovery_interval() == 1.0
        assert runner._archiving_interval() == 1.0
        while not runner.queue.empty():runner.queue.get()
        assert runner._discovery_interval() == .1
