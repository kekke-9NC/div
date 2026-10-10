"""Integration coverage for legacy persistence and the live OBS file watcher."""
from pathlib import Path
import json
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from zipfile import BadZipFile

import numpy as np
from PIL import Image

from legacy_mask_sync import persist_detection_mask

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'obs_meteor_overlay'))
from legacy_detection_mask import LegacyDetectionMask


class LegacyMaskSyncTests(unittest.TestCase):
    def test_saved_mask_path_toggle_and_partial_write(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            settings = root / 'app_settings.json'
            settings.write_text(json.dumps({'unrelated': {'preserve': 42}}))
            archive = root / 'custom' / 'masks.npz'
            mask = np.array([[0, 255], [1, 0]], dtype=np.uint8)
            ps_mask = np.full((2, 2), 255, dtype=np.uint8)
            persist_detection_mask(settings, archive, mask, ps_mask, True)
            reader = LegacyDetectionMask(settings)
            self.assertTrue(reader.refresh())
            self.assertEqual(reader.mask_path, archive.resolve())
            np.testing.assert_array_equal(reader.mask, [[0, 255], [255, 0]])
            self.assertEqual(json.loads(settings.read_text())['unrelated'], {'preserve': 42})
            with np.load(archive, allow_pickle=False) as arrays:
                np.testing.assert_array_equal(arrays['plate_solve_mask_image'], ps_mask)
            revision = reader.revision
            archive.write_bytes(b'partial archive')
            with self.assertRaises((ValueError, EOFError, BadZipFile)):
                reader.refresh()
            self.assertEqual(reader.revision, revision)
            np.testing.assert_array_equal(reader.mask, [[0, 255], [255, 0]])
            new_archive = root / 'moved.npz'
            persist_detection_mask(settings, new_archive, mask, ps_mask, False)
            self.assertTrue(reader.refresh())
            self.assertIsNone(reader.mask)
            self.assertEqual(reader.mask_path, new_archive.resolve())
            png = root / 'mask.png'
            reader.write_png(png, 4, 4)
            self.assertTrue(np.all(np.asarray(Image.open(png)) == 255))

    @unittest.skipUnless(shutil.which('ffmpeg'), 'ffmpeg required for GIF integration')
    def test_running_worker_updates_same_event_after_mask_change(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            events, output = root / 'meteor', root / 'overlay'
            events.mkdir()
            output.mkdir()
            raw = root / 'raw.mp4'
            ffmpeg = shutil.which('ffmpeg')
            subprocess.run([ffmpeg, '-v', 'error', '-y', '-f', 'lavfi', '-i',
                            'color=c=white:s=320x320:r=12:d=0.5', '-c:v', 'libx264',
                            '-pix_fmt', 'yuv420p', str(raw)], check=True, timeout=15)
            event = '20261010_232741111_meteor_1_prob1.00'
            shutil.copyfile(raw, events / (event + '_summary.mp4'))
            (events / (event + '_info.txt')).write_text(
                f'Source: {raw}\nFrame Range (Clip): 0 - 5\n'
                'Detected Line Center (px): (160.00, 160.00)\n')
            settings, archive = root / 'app_settings.json', root / 'app_masks.npz'
            mask = np.full((320, 320), 255, dtype=np.uint8)
            mask[:, :160] = 0
            persist_detection_mask(settings, archive, mask, None, True)
            process = subprocess.Popen([
                sys.executable, str(ROOT / 'obs_meteor_overlay/latest_meteor_gif.py'),
                '--meteor-dir', str(events), '--output-dir', str(output),
                '--legacy-settings', str(settings), '--ffmpeg', ffmpeg, '--poll-seconds', '0.5',
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            def wait_until(predicate):
                deadline = time.monotonic() + 25
                while time.monotonic() < deadline:
                    if process.poll() is not None:
                        self.fail('worker exited: ' + process.stderr.read())
                    if predicate():
                        return
                    time.sleep(0.1)
                self.fail('timed out waiting for automatic mask update')
            try:
                full, zoom = output / 'latest_meteor.gif', output / 'latest_meteor_zoom.gif'
                wait_until(lambda: full.exists() and zoom.exists() and (output / 'latest_meteor.json').exists())
                # Check actual decoded pixels, including mask-before-crop for zoom.
                for image in (full, zoom):
                    with Image.open(image) as gif:
                        pixels = np.asarray(gif.convert('RGB'))
                    y = pixels.shape[0] // 2
                    self.assertTrue(np.all(pixels[y, pixels.shape[1] // 4] == 0))
                    self.assertTrue(np.all(pixels[y, pixels.shape[1] * 3 // 4] > 200))
                previous = full.stat().st_mtime_ns
                state = json.loads((output / 'mask_state.json').read_text())
                # Replace the mask while the event and enabled checkbox stay
                # unchanged. The watcher must update the existing replay.
                inverse = 255 - mask
                persist_detection_mask(settings, archive, inverse, None, True)
                wait_until(lambda: full.stat().st_mtime_ns > previous and
                           json.loads((output / 'mask_state.json').read_text())['revision'] != state['revision'])
                for image in (full, zoom):
                    with Image.open(image) as gif:
                        pixels = np.asarray(gif.convert('RGB'))
                    y = pixels.shape[0] // 2
                    self.assertTrue(np.all(pixels[y, pixels.shape[1] // 4] > 200))
                    self.assertTrue(np.all(pixels[y, pixels.shape[1] * 3 // 4] == 0))
                previous = full.stat().st_mtime_ns
                # No new event is saved: toggling the legacy UI checkbox must
                # still regenerate both views and the camera filter's PNG.
                persist_detection_mask(settings, archive, mask, None, False)
                wait_until(lambda: full.stat().st_mtime_ns > previous and
                           not json.loads((output / 'mask_state.json').read_text())['enabled'])
                for image in (full, zoom):
                    with Image.open(image) as gif:
                        pixels = np.asarray(gif.convert('RGB'))
                    self.assertTrue(np.all(pixels[pixels.shape[0] // 2, pixels.shape[1] // 4] > 200))
                self.assertNotEqual(state['revision'], json.loads((output / 'mask_state.json').read_text())['revision'])
                self.assertTrue(np.all(np.asarray(Image.open(output / 'detection_mask.png')) == 255))
                self.assertEqual(json.loads((output / 'latest_meteor.json').read_text())['event'], event)
            finally:
                process.terminate()
                try:
                    process.communicate(timeout=15)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.communicate()


if __name__ == '__main__':
    unittest.main()
