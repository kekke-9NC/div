import tempfile
import unittest
import json
from pathlib import Path

import cv2
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS, Sip
from unittest import mock

import astrometry


class MeteorAnnotationTests(unittest.TestCase):
    def test_marker_draws_only_a_rectangle(self):
        image = np.zeros((200, 200, 3), dtype=np.uint8)

        marked = astrometry._draw_meteor_marker(image, ((80, 80), (120, 120)))

        # The detected trajectory remains unobscured; the marker is only the border.
        np.testing.assert_array_equal(marked[100, 100], np.zeros(3, dtype=np.uint8))
        self.assertTrue(np.any(marked[44, 44] > 0))

    def test_annotation_preserves_source_orientation_by_default(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source_path = Path(temp_dir) / "orientation.png"
            source = np.zeros((12, 16, 3), dtype=np.uint8)
            source[0, 0] = (10, 20, 30)
            source[-1, -1] = (200, 180, 160)
            self.assertTrue(cv2.imwrite(str(source_path), source))

            annotated_path = astrometry.annotate_image_with_wcs(
                str(source_path), {}, timestamp=None
            )

            self.assertIsNotNone(annotated_path)
            annotated = cv2.imread(str(annotated_path))
            np.testing.assert_array_equal(annotated, source)

    def test_wcs_annotation_preserves_top_left_camera_orientation(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source_path = root / "orientation_composite.png"
            source = np.zeros((120, 160, 3), dtype=np.uint8)
            source[:40, :] = (0, 0, 255)      # red at camera top (BGR)
            source[-40:, :] = (255, 0, 0)     # blue at camera bottom
            self.assertTrue(cv2.imwrite(str(source_path), source))
            wcs = WCS(naxis=2)
            wcs.wcs.crpix = [80, 60]
            wcs.wcs.cdelt = [-0.1, 0.1]
            wcs.wcs.crval = [230.0, 52.0]
            wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
            wcs_path = root / "orientation.wcs"
            fits.PrimaryHDU(data=np.zeros((120, 160), dtype=np.uint8),
                            header=wcs.to_header()).writeto(wcs_path)

            with mock.patch.object(astrometry, "_annotate_stars_and_grid"):
                annotated_path = astrometry.annotate_image_with_wcs(
                    str(source_path), {"wcs_file": str(wcs_path)}, timestamp=None
                )

            self.assertIsNotNone(annotated_path)
            annotated = cv2.imread(str(annotated_path))
            red_y = np.where(
                annotated[:, :, 2].astype(int) - annotated[:, :, 0].astype(int) > 80
            )[0]
            blue_y = np.where(
                annotated[:, :, 0].astype(int) - annotated[:, :, 2].astype(int) > 80
            )[0]
            self.assertGreater(len(red_y), 100)
            self.assertGreater(len(blue_y), 100)
            self.assertLess(float(np.median(red_y)), float(np.median(blue_y)))

    def test_local_sip_annotation_uses_distortion_aware_renderer(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source_path = root / "local_composite.png"
            self.assertTrue(cv2.imwrite(
                str(source_path), np.zeros((120, 160, 3), dtype=np.uint8)
            ))
            wcs = WCS(naxis=2)
            wcs.wcs.crpix = [80, 60]
            wcs.wcs.cdelt = [-0.1, 0.1]
            wcs.wcs.crval = [230.0, 52.0]
            wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
            header = wcs.to_header()
            header["CALTYPE"] = "LOCAL-SIP"
            wcs_path = root / "wideangle_sip.wcs"
            fits.PrimaryHDU(header=header).writeto(wcs_path)
            (root / "calibration.json").write_text(json.dumps({
                "wcs_path": str(wcs_path),
                "reference_datetime": "2026-07-10T01:00:00",
                "width": 160,
                "height": 120,
                "sip_support_hull": [[20, 20], [140, 20], [140, 100], [20, 100]],
            }), encoding="utf-8")
            expected = str(root / "local_annotated.png")

            with mock.patch.object(
                astrometry, "_annotate_local_wideangle_image", return_value=expected
            ) as local_renderer:
                actual = astrometry.annotate_image_with_wcs(
                    str(source_path), {"wcs_file": str(wcs_path)}, timestamp=None
                )

            self.assertEqual(actual, expected)
            local_renderer.assert_called_once()

    def test_local_sip_does_not_report_coordinates_outside_support(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source_path = root / "local_composite.png"
            self.assertTrue(cv2.imwrite(
                str(source_path), np.zeros((120, 160, 3), dtype=np.uint8)
            ))
            wcs = WCS(naxis=2)
            wcs.wcs.crpix = [80, 60]
            wcs.wcs.cdelt = [-0.1, 0.1]
            wcs.wcs.crval = [230.0, 52.0]
            wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
            header = wcs.to_header()
            header["CALTYPE"] = "LOCAL-SIP"
            wcs_path = root / "wideangle_sip.wcs"
            fits.PrimaryHDU(header=header).writeto(wcs_path)
            calibration_path = root / "calibration.json"
            calibration_path.write_text(json.dumps({
                "wcs_path": str(wcs_path),
                "reference_datetime": "2026-07-10T01:00:00",
                "width": 160,
                "height": 120,
                "sip_support_hull": [[60, 40], [100, 40], [100, 80], [60, 80]],
            }), encoding="utf-8")

            with mock.patch.object(
                astrometry.cv2, "putText", wraps=cv2.putText
            ) as put_text:
                astrometry._annotate_local_wideangle_image(
                    str(source_path),
                    {"wcs_file": str(wcs_path), "calibration_path": str(calibration_path)},
                    [(5.0, 110.0)], None, None, False, None,
                )

            labels = [call.args[1] for call in put_text.call_args_list]
            self.assertIn("RA/Dec unavailable", labels)
            self.assertIn("outside calibrated area", labels)
            self.assertFalse(any(label.startswith("RA: ") for label in labels))

    def test_vertical_wcs_conversion_preserves_sip_sky_coordinates(self):
        height = 180
        wcs = WCS(naxis=2)
        wcs.wcs.crpix = [160, 90]
        wcs.wcs.cd = np.asarray([[-0.08, 0.003], [0.002, 0.08]])
        wcs.wcs.crval = [230.0, 52.0]
        wcs.wcs.ctype = ["RA---TAN-SIP", "DEC--TAN-SIP"]
        a = np.zeros((4, 4)); b = np.zeros((4, 4))
        a[2, 0], a[1, 1], a[0, 2] = 2e-5, -1e-5, 1.5e-5
        b[2, 0], b[1, 1], b[0, 2] = -1e-5, 2e-5, -1.5e-5
        wcs.sip = Sip(a, b, None, None, wcs.wcs.crpix)
        flipped = astrometry._create_flipped_wcs(wcs, (height, 320))
        x = np.asarray([40.0, 160.0, 275.0])
        y = np.asarray([25.0, 90.0, 150.0])

        old_ra, old_dec = wcs.pixel_to_world_values(x, y)
        new_ra, new_dec = flipped.pixel_to_world_values(x, height - 1 - y)

        np.testing.assert_allclose(new_ra, old_ra, atol=1e-9)
        np.testing.assert_allclose(new_dec, old_dec, atol=1e-9)


if __name__ == "__main__":
    unittest.main()
