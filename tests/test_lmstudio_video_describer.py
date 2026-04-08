import unittest

import lmstudio_video_describer as app


class LMStudioVideoDescriberTests(unittest.TestCase):
    def test_normalize_server_url_trims_known_suffixes(self):
        self.assertEqual(app.normalize_server_url("http://localhost:1234/v1"), "http://localhost:1234")
        self.assertEqual(app.normalize_server_url("http://localhost:1234/api/v1"), "http://localhost:1234")

    def test_video_extension_detection(self):
        self.assertTrue(app.is_video_file("movie.mp4"))
        self.assertTrue(app.is_video_file("movie.MKV"))
        self.assertFalse(app.is_video_file("image.png"))

    def test_format_seconds(self):
        self.assertEqual(app.format_seconds(0), "00:00.000")
        self.assertEqual(app.format_seconds(12.345), "00:12.345")
        self.assertEqual(app.format_seconds(61.001), "01:01.001")


if __name__ == "__main__":
    unittest.main()
