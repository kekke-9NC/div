import unittest

import video_processor


class VideoProcessorAutoBitrateTests(unittest.TestCase):
    def test_parse_quality_metrics(self):
        text = "SSIM Y:0.99 All:0.987654 (18.2) PSNR average:39.75 min:35.0"
        self.assertEqual(video_processor.parse_quality_metrics(text), (0.987654, 39.75))

    def test_candidates_are_ascending_and_codec_aware(self):
        h264 = video_processor.automatic_bitrate_candidates("h264", 1920, 1080, 25)
        h265 = video_processor.automatic_bitrate_candidates("h265", 1920, 1080, 25)
        self.assertEqual(h264[0], "750k")
        self.assertEqual(h265[0], "500k")
        self.assertLess(int(h265[-1][:-1]), int(h264[-1][:-1]))


if __name__ == "__main__":
    unittest.main()
