import unittest

import video_processor


class IndependentConcatFilterTests(unittest.TestCase):
    def test_each_video_is_decoded_and_timestamped_independently(self):
        graph, output = video_processor._build_independent_concat_filter(
            3, 25.0, "W-w-8:H-h-8"
        )

        self.assertIn("[0:v]settb=AVTB,setpts=PTS-STARTPTS[segment0]", graph)
        self.assertIn("[1:v]settb=AVTB,setpts=PTS-STARTPTS[segment1]", graph)
        self.assertIn("[2:v]settb=AVTB,setpts=PTS-STARTPTS[segment2]", graph)
        self.assertIn("concat=n=3:v=1:a=0[joined]", graph)
        self.assertIn("[joined]fps=25.0[base]", graph)
        self.assertIn("[3:v]setpts=PTS-STARTPTS[clock]", graph)
        self.assertEqual(output, "v")

    def test_filter_without_overlay_returns_joined_video(self):
        graph, output = video_processor._build_independent_concat_filter(2, None)

        self.assertIn("concat=n=2:v=1:a=0[joined]", graph)
        self.assertEqual(output, "joined")
