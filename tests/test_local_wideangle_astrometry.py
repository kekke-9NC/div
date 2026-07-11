import json
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest import mock

import cv2
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

import local_wideangle_astrometry as local_astro


def _tan_wcs(width=640, height=360):
    wcs = WCS(naxis=2)
    wcs.wcs.crpix = [width / 2, height / 2]
    wcs.wcs.cd = np.asarray([[-0.12, 0.008], [0.006, 0.12]])
    wcs.wcs.crval = [230.0, 52.0]
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    return wcs


class _FakeCapture:
    def isOpened(self):
        return True

    def get(self, prop):
        if prop == cv2.CAP_PROP_FRAME_WIDTH:
            return 640
        if prop == cv2.CAP_PROP_FRAME_HEIGHT:
            return 360
        return 0

    def release(self):
        pass


class LocalWideangleAstrometryTests(unittest.TestCase):
    def test_capture_datetime_uses_recorder_path(self):
        value = local_astro._capture_datetime("/camera/20260710/01/01.mp4")
        self.assertEqual(value, datetime(2026, 7, 10, 1, 1))

    def test_night_cache_is_returned_without_decoding_video(self):
        with tempfile.TemporaryDirectory() as directory:
            source = str(Path(directory) / "20260710" / "01" / "01.mp4")
            paths = local_astro._calibration_paths(source, 640, 360, directory)
            paths["wcs"].write_bytes(b"cached")
            paths["metadata"].write_text(json.dumps({
                "algorithm_version": local_astro.ALGORITHM_VERSION,
                "reference_datetime": "2026-07-10T01:01:00",
                "wcs_path": str(paths["wcs"]),
            }), encoding="utf-8")
            with mock.patch.object(local_astro, "_open_video", return_value=_FakeCapture()), \
                 mock.patch.object(local_astro, "_video_sample_stack") as decode:
                result = local_astro.solve_video_local(source, cache_root=directory)
            decode.assert_not_called()
            self.assertEqual(result["job_id"], "local-wideangle-cache")

    def test_annotation_draws_forward_grid_without_inverse_sip(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            wcs_path = root / "calibration.wcs"
            header = _tan_wcs(320, 180).to_header(relax=True)
            header["IMAGEW"] = 320
            header["IMAGEH"] = 180
            header["DATE-OBS"] = "2026-07-10T01:00:00"
            fits.PrimaryHDU(header=header).writeto(wcs_path)
            metadata_path = root / "calibration.json"
            metadata_path.write_text(json.dumps({
                "wcs_path": str(wcs_path),
                "reference_datetime": "2026-07-10T01:00:00",
                "width": 320, "height": 180, "center_ra_deg": 230.0,
                "sip_support_hull": [[5, 5], [315, 5], [315, 175], [5, 175]],
            }), encoding="utf-8")
            frame = np.zeros((180, 320, 3), dtype=np.uint8)
            output = local_astro.annotate_frame(
                frame, datetime(2026, 7, 10, 1, 30), str(metadata_path)
            )
            self.assertEqual(output.shape, frame.shape)
            self.assertGreater(np.count_nonzero(output), 300)

    def test_catalog_bootstrap_fits_real_sip_coefficients(self):
        width, height = 640, 360
        base = _tan_wcs(width, height)
        ideal_x, ideal_y = np.meshgrid(
            np.linspace(100, 540, 9), np.linspace(45, 315, 6)
        )
        ideal = np.column_stack((ideal_x.ravel(), ideal_y.ravel()))
        sky = base.pixel_to_world(ideal[:, 0], ideal[:, 1])
        normalized_x = (ideal[:, 0] - width / 2) / (width / 2)
        normalized_y = (ideal[:, 1] - height / 2) / (height / 2)
        radius2 = normalized_x ** 2 + normalized_y ** 2
        scale = 1.0 + 0.035 * radius2 + 0.012 * radius2 ** 2
        observed = np.column_stack((
            width / 2 + (ideal[:, 0] - width / 2) * scale,
            height / 2 + (ideal[:, 1] - height / 2) * scale,
        ))
        # Add unrelated hot-pixel candidates; mutual matching must ignore them.
        rng = np.random.default_rng(4)
        observed = np.vstack((observed, rng.uniform([0, 0], [width, height], (80, 2))))
        catalog = [
            {"ra_deg": float(coord.ra.deg), "dec_deg": float(coord.dec.deg)}
            for coord in sky
        ]

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = {
                "wcs": root / "wideangle.wcs",
                "reference": root / "reference.jpg",
                "validation": root / "validation.jpg",
            }
            fits.PrimaryHDU(header=base.to_header(relax=True)).writeto(paths["wcs"])
            cv2.imwrite(str(paths["reference"]), np.zeros((height, width), dtype=np.uint8))
            result = local_astro._refine_sip_wcs(
                paths, observed.tolist(), catalog, width, height,
                datetime(2026, 7, 10, 1, 0),
            )
            self.assertTrue(result["sip_refined"])
            self.assertLess(result["sip_residual_p95_px"], 2.5)
            with fits.open(paths["wcs"]) as hdul:
                self.assertGreaterEqual(hdul[0].header["A_ORDER"], 4)


if __name__ == "__main__":
    unittest.main()
