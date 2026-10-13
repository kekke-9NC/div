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
