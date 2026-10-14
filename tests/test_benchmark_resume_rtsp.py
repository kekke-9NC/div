from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from tools.benchmark_resume_rtsp import rebase


class ResumeBenchmarkTests(unittest.TestCase):
    def test_rebase_preserves_snapshot_paths_in_dictionary_keys_and_values(self):
        source, target = Path('/original/rtsp'), Path('/isolated/rtsp')
        data = {'versions': {'/original/rtsp/a.mp4': [1, 2]},
                'outputs': ['/original/rtsp/products/movie.mp4'], 'count': 3}
        self.assertEqual(rebase(data, source, target),
                         {'versions': {'/isolated/rtsp/a.mp4': [1, 2]},
                          'outputs': ['/isolated/rtsp/products/movie.mp4'], 'count': 3})

    def test_nested_output_is_rejected_before_any_source_writes(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)/'original'
            source.mkdir()
            output = source/'benchmark'
            script = Path(__file__).resolve().parents[1]/'tools/benchmark_legacy_detection.py'
            result = subprocess.run([sys.executable,str(script),'--code-root',str(source),
                '--settings',str(source/'missing.json'),'--output',str(output),
                '--resume-root',str(source)],capture_output=True,text=True)
            self.assertEqual(result.returncode, 2)
            self.assertIn('separate from the original resume root',result.stderr)
            self.assertFalse(output.exists())
            self.assertEqual(list(source.iterdir()), [])
