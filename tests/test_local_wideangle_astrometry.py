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

    def test_registered_model_falls_back_to_best_same_camera_model_on_unseen_night(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = str(root / "20260811" / "02" / "21.mp4")
            _date, camera = local_astro._night_identity(source, 640, 360)
            models_root = root / "camera_models"
            candidates = [
                ("v3", "20260807", 0.42, 3.0),
                ("v4", "20260809", 0.80, 1.8),
            ]
            for name, valid_date, support, p95 in candidates:
                model_dir = models_root / name
                model_dir.mkdir(parents=True)
                wcs_path = model_dir / "model.wcs"
                wcs_path.write_bytes(b"wcs")
                (model_dir / "camera_model.json").write_text(json.dumps({
                    "model_type": "fixed-camera-stg-poly",
                    "algorithm_version": "camera-model-test",
                    "enabled": True,
                    "width": 640,
                    "height": 360,
                    "camera_aliases": [camera],
                    "valid_dates": [valid_date],
                    "support_fraction": support,
                    "residual_p95_px": p95,
                    "reference_datetime": f"{valid_date}T01:00:00",
                    "wcs_path": str(wcs_path),
                    "model_label": name,
                }), encoding="utf-8")

            with mock.patch.object(local_astro, "_open_video", return_value=_FakeCapture()), \
                 mock.patch.object(local_astro, "_video_sample_stack") as decode:
                result = local_astro.solve_video_local(source, cache_root=directory)

            decode.assert_not_called()
            self.assertEqual(result["model_label"], "v4")
            self.assertFalse(result["_model_date_match"])

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

    def test_overlapping_contour_segments_keep_distinct_parts(self):
        first = np.asarray([[0.0, 0.0], [10.0, 0.0], [20.0, 0.0], [30.0, 0.0]])
        second = np.asarray([
            [0.0, 1.0], [10.0, 1.0], [20.0, 10.0], [25.0, 10.0], [30.0, 10.0],
        ])
        selected, occupied = local_astro._split_overlapping_contour_segments(
            [first, second], np.empty((0, 2)), 3.0
        )
        self.assertEqual(len(selected), 2)
        np.testing.assert_allclose(selected[0], first)
        np.testing.assert_allclose(selected[1], second[2:])
        self.assertEqual(len(occupied), len(first) + len(second[2:]))

    def test_bare_wcs_recovers_sibling_support_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            wcs_path = root / "wideangle_sip.wcs"
            header = _tan_wcs(320, 180).to_header(relax=True)
            header["IMAGEW"] = 320
            header["IMAGEH"] = 180
            header["DATE-OBS"] = "2026-07-10T01:00:00"
            header["CALTYPE"] = "LOCAL-SIP"
            fits.PrimaryHDU(header=header).writeto(wcs_path)
            support_hull = [[40, 30], [280, 30], [280, 150], [40, 150]]
            metadata_path = root / "calibration.json"
            metadata_path.write_text(json.dumps({
                "wcs_path": str(wcs_path),
                "reference_datetime": "2026-07-10T01:00:00",
                "width": 320,
                "height": 180,
                "sip_support_hull": support_hull,
            }), encoding="utf-8")

            metadata, _wcs = local_astro._load_calibration(str(wcs_path))

            self.assertEqual(metadata["sip_support_hull"], support_hull)
            self.assertTrue(Path(metadata["calibration_path"]).samefile(metadata_path))

    def test_fixed_camera_model_preserves_its_fitted_sidereal_epoch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            wcs_path = root / "wideangle_sip.wcs"
            header = _tan_wcs(320, 180).to_header(relax=True)
            header["IMAGEW"] = 320
            header["IMAGEH"] = 180
            header["DATE-OBS"] = "2026-08-09T04:00:00"
            fits.PrimaryHDU(header=header).writeto(wcs_path)
            calibration_path = root / "calibration.json"
            calibration_path.write_text(json.dumps({
                "wcs_path": str(wcs_path),
                "catalog_stars": [{"ra_deg": 12.0, "dec_deg": 34.0}],
                "sip_residual_p95_px": 2.5,
            }), encoding="utf-8")
            model_path = root / "camera_model.json"
            model_path.write_text(json.dumps({
                "model_type": "fixed-camera-stg-poly",
                "width": 320,
                "height": 180,
                "polynomial_degree": 0,
                "stg_parameters": [0.0, 0.0, 0.0, 4.605170186, 4.605170186, 160.0, 90.0],
                "correction_coefficients": [[0.0, 0.0]],
                "wcs_path": str(wcs_path),
                "reference_datetime": "2026-08-07T19:59:15.469000",
            }), encoding="utf-8")

            metadata, _model = local_astro._load_calibration(str(model_path))

            self.assertEqual(metadata["reference_datetime"], "2026-08-07T19:59:15.469000")
            self.assertEqual(metadata["catalog_stars"][0]["ra_deg"], 12.0)

    def test_annotation_can_draw_constellation_lines(self):
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
            lines = (np.asarray([[229.0, 52.0], [231.0, 52.0]]),)
            with mock.patch.object(local_astro, "_constellation_lines", lines):
                grid_only = local_astro.annotate_frame(
                    frame, datetime(2026, 7, 10, 1, 0), str(metadata_path)
                )
                with_lines = local_astro.annotate_frame(
                    frame, datetime(2026, 7, 10, 1, 0), str(metadata_path),
                    draw_constellations=True,
                )
            self.assertGreater(np.count_nonzero(with_lines), np.count_nonzero(grid_only))

    def test_legacy_verified_constellation_flag_enables_anchor_filter(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            wcs_path = root / "calibration.wcs"
            header = _tan_wcs(320, 180).to_header(relax=True)
            header["DATE-OBS"] = "2026-07-10T01:00:00"
            fits.PrimaryHDU(header=header).writeto(wcs_path)
            metadata_path = root / "calibration.json"
            metadata_path.write_text(json.dumps({
                "wcs_path": str(wcs_path),
                "reference_datetime": "2026-07-10T01:00:00",
                "width": 320, "height": 180, "center_ra_deg": 230.0,
                "sip_support_hull": [[5, 5], [315, 5], [315, 175], [5, 175]],
                "verified_constellation_only": True,
                "catalog_stars": [{"ra_deg": 230.0, "dec_deg": 52.0}],
            }), encoding="utf-8")
            with mock.patch.object(
                local_astro, "_extract_stars",
                return_value=([[10.0, 10.0]], None, None),
            ), mock.patch.object(local_astro, "_draw_constellation_lines") as draw, \
                mock.patch.object(
                    local_astro, "_estimate_constellation_pixel_offset",
                    return_value=(0.0, 0.0),
                ) as align:
                local_astro.annotate_frame(
                    np.zeros((180, 320, 3), dtype=np.uint8),
                    datetime(2026, 7, 10, 1, 0), str(metadata_path),
                    draw_constellations=True,
                )
            self.assertIsNotNone(draw.call_args.kwargs["anchor_points"])
            align.assert_called_once()

    def test_continuous_trajectory_constellations_do_not_depend_on_cloudy_frame_stars(self):
        metadata = {
            "reference_datetime": "2026-08-13T00:00:00",
            "constellation_render_policy": "model-supported-continuous",
            "constellation_anchor_filter": True,
            "constellation_anchor_tolerance_px": 4.0,
        }
        strict = np.zeros((18, 32), dtype=np.uint8)
        strict[:, 4:28] = 255
        display = strict.copy()
        display[:, 2:30] = 255
        grid = {
            "support_mask": strict,
            "display_support_mask": display,
        }
        with mock.patch.object(local_astro, "_load_calibration", return_value=(metadata, object())), \
             mock.patch.object(local_astro, "_forward_grid_model", return_value=grid), \
             mock.patch.object(local_astro, "_extract_stars") as extract, \
             mock.patch.object(local_astro, "_draw_constellation_lines") as draw:
            local_astro.annotate_frame(
                np.zeros((18, 32, 3), dtype=np.uint8),
                datetime(2026, 8, 13, 0, 1),
                draw_grid=False,
                draw_constellations=True,
            )
        extract.assert_not_called()
        self.assertIsNone(draw.call_args.kwargs["anchor_points"])
        self.assertTrue(draw.call_args.kwargs["allow_partial_segments"])
        self.assertIs(draw.call_args.args[3], display)

    def test_cloudy_frame_hides_all_constellation_lines(self):
        metadata = {
            "reference_datetime": "2026-08-13T00:00:00",
            "constellation_render_policy": "model-supported-continuous",
            "constellation_cloud_filter": True,
            "constellation_cloud_threshold": 0.10,
        }
        support = np.full((18, 32), 255, dtype=np.uint8)
        grid = {"support_mask": support, "display_support_mask": support}
        with mock.patch.object(local_astro, "_load_calibration", return_value=(metadata, object())), \
             mock.patch.object(local_astro, "_forward_grid_model", return_value=grid), \
             mock.patch.object(local_astro, "_estimate_constellation_cloud_fraction", return_value=0.45), \
             mock.patch.object(local_astro, "_draw_constellation_lines") as draw:
            local_astro.annotate_frame(
                np.zeros((18, 32, 3), dtype=np.uint8),
                datetime(2026, 8, 13, 0, 1),
                draw_grid=False,
                draw_constellations=True,
            )
        draw.assert_not_called()

    def test_detected_endpoint_policy_suppresses_lines_without_current_stars(self):
        metadata = {
            "reference_datetime": "2026-08-13T00:00:00",
            "constellation_render_policy": "model-supported-detected-endpoints",
            "constellation_anchor_tolerance_px": 6.0,
        }
        strict = np.zeros((18, 32), dtype=np.uint8)
        strict[:, 4:28] = 255
        display = strict.copy()
        display[:, 2:30] = 255
        grid = {"support_mask": strict, "display_support_mask": display}
        with mock.patch.object(local_astro, "_load_calibration", return_value=(metadata, object())), \
             mock.patch.object(local_astro, "_forward_grid_model", return_value=grid), \
             mock.patch.object(local_astro, "_extract_stars", return_value=([], None, None)) as extract, \
             mock.patch.object(local_astro, "_draw_constellation_lines") as draw:
            local_astro.annotate_frame(
                np.zeros((18, 32, 3), dtype=np.uint8),
                datetime(2026, 8, 13, 0, 1),
                draw_grid=False,
                draw_constellations=True,
            )
        extract.assert_called_once()
        self.assertEqual(len(draw.call_args.kwargs["anchor_points"]), 0)
        self.assertFalse(draw.call_args.kwargs["allow_partial_segments"])
        self.assertIs(draw.call_args.args[3], display)

    def test_short_constellation_detection_gaps_are_bridged_only_when_bounded(self):
        raw = np.asarray([
            [True, False, False, True, False, False, False, False, True],
            [True, False, False, False, False, False, False, False, True],
        ], dtype=bool)
        bridged = local_astro._bridge_boolean_gaps(raw, 2)
        np.testing.assert_array_equal(
            bridged[0],
            [True, True, True, True, False, False, False, False, True],
        )
        np.testing.assert_array_equal(bridged[1], raw[1])

    def test_temporally_held_constellation_edge_draws_without_current_anchors(self):
        class ForwardOnlyFixedModel(local_astro.FixedCameraPlateModel):
            def __init__(self):
                pass

            def world_to_pixel_values(self, ra, _dec):
                wrapped = (np.asarray(ra, dtype=float) + 180.0) % 360.0 - 180.0
                return wrapped * 10.0 + 10.0, np.full_like(wrapped, 90.0)

            def pixel_to_world_values(self, x, _y):
                return (np.asarray(x, dtype=float) - 10.0) / 10.0, np.full_like(
                    np.asarray(x, dtype=float), 20.0
                )

        support = np.full((180, 320), 255, dtype=np.uint8)
        line = (np.asarray([[0.0, 20.0], [5.0, 20.0]]),)
        output = np.zeros((180, 320, 3), dtype=np.uint8)
        with mock.patch.object(local_astro, "_constellation_lines", line):
            local_astro._draw_constellation_lines(
                output,
                ForwardOnlyFixedModel(),
                0.0,
                support,
                anchor_points=np.asarray([[999.0, 999.0]], dtype=float),
                allow_partial_segments=False,
                edge_admission=np.asarray([True], dtype=bool),
            )
        self.assertGreater(np.count_nonzero(output), 20)

    def test_continuous_constellation_policy_draws_supported_partial_edge(self):
        class ForwardOnlyFixedModel(local_astro.FixedCameraPlateModel):
            def __init__(self):
                pass

            def world_to_pixel_values(self, ra, _dec):
                wrapped = (np.asarray(ra, dtype=float) + 180.0) % 360.0 - 180.0
                return wrapped * 10.0 + 10.0, np.full_like(wrapped, 90.0)

            def pixel_to_world_values(self, x, _y):
                return (np.asarray(x, dtype=float) - 10.0) / 10.0, np.full_like(
                    np.asarray(x, dtype=float), 20.0
                )

        support = np.full((180, 320), 255, dtype=np.uint8)
        line = (np.asarray([[-2.0, 20.0], [5.0, 20.0]]),)
        strict_output = np.zeros((180, 320, 3), dtype=np.uint8)
        partial_output = strict_output.copy()
        with mock.patch.object(local_astro, "_constellation_lines", line):
            local_astro._draw_constellation_lines(
                strict_output, ForwardOnlyFixedModel(), 0.0, support,
                allow_partial_segments=False,
            )
            local_astro._draw_constellation_lines(
                partial_output, ForwardOnlyFixedModel(), 0.0, support,
                allow_partial_segments=True,
            )
        self.assertEqual(np.count_nonzero(strict_output), 0)
        self.assertGreater(np.count_nonzero(partial_output), 20)

    def test_constellation_projection_rejects_wrong_fixed_model_inverse_branch(self):
        class FoldedFixedModel(local_astro.FixedCameraPlateModel):
            def __init__(self):
                pass

            def world_to_pixel_values(self, ra, dec):
                return np.asarray(ra, dtype=float) * 2.0, np.asarray(dec, dtype=float) * 2.0

            def pixel_to_world_values(self, x, y):
                # The detector position belongs to a different sky branch.
                return np.asarray(x, dtype=float) / 2.0 + 20.0, np.asarray(y, dtype=float) / 2.0

        _x, _y, usable = local_astro._project_constellation_samples(
            FoldedFixedModel(), np.asarray([10.0, 11.0]), np.asarray([20.0, 20.0])
        )
        self.assertFalse(np.any(usable))

    def test_detected_star_mode_draws_hollow_markers_from_frame_pixels(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            wcs_path = root / "calibration.wcs"
            header = _tan_wcs(320, 180).to_header(relax=True)
            header["IMAGEW"] = 320
            header["IMAGEH"] = 180
            fits.PrimaryHDU(header=header).writeto(wcs_path)
            metadata_path = root / "calibration.json"
            metadata_path.write_text(json.dumps({
                "wcs_path": str(wcs_path),
                "reference_datetime": "2026-07-10T01:00:00",
                "width": 320, "height": 180, "center_ra_deg": 230.0,
            }), encoding="utf-8")
            frame = np.zeros((180, 320, 3), dtype=np.uint8)
            for x, y in ((80, 50), (120, 70), (180, 90), (240, 110)):
                frame[y, x] = 255
            with mock.patch.object(local_astro, "_forward_grid_model") as grid_model:
                output = local_astro.annotate_frame(
                    frame, datetime(2026, 7, 10, 1, 0), str(metadata_path),
                    draw_grid=False, draw_detected_stars=True,
                )
            grid_model.assert_not_called()
            # The source star remains visible at the empty center while the
            # green circumference is added around it.
            np.testing.assert_array_equal(output[50, 80], frame[50, 80])
            self.assertGreater(int(output[50, 84, 1]), int(frame[50, 84, 1]))

    def test_grid_can_be_disabled_independently(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            wcs_path = root / "calibration.wcs"
            fits.PrimaryHDU(header=_tan_wcs(320, 180).to_header(relax=True)).writeto(wcs_path)
            metadata_path = root / "calibration.json"
            metadata_path.write_text(json.dumps({
                "wcs_path": str(wcs_path),
                "reference_datetime": "2026-07-10T01:00:00",
                "width": 320, "height": 180,
            }), encoding="utf-8")
            with mock.patch.object(local_astro, "_forward_grid_model") as grid_model:
                local_astro.annotate_frame(
                    np.zeros((180, 320, 3), dtype=np.uint8),
                    datetime(2026, 7, 10, 1, 0), str(metadata_path), draw_grid=False,
                )
            grid_model.assert_not_called()

    def test_display_grid_bridges_only_surrounded_internal_holes(self):
        values = np.asarray([
            [0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0],
            [0, 1, 0, 1, 0],
            [0, 1, 1, 1, 0],
            [0, 0, 0, 0, 0],
        ], dtype=np.uint8)
        display = local_astro._bridge_support_grid_for_display(values)
        self.assertEqual(int(display[2, 2]), 1)
        # The display-only operation must not expand the outside boundary.
        self.assertTrue(np.all(display[0] == 0))
        self.assertTrue(np.all(display[:, 0] == 0))
        self.assertEqual(int(values[2, 2]), 0)

    def test_display_grid_bridges_only_short_bounded_polyline_gaps(self):
        support = np.zeros((41, 81), dtype=np.uint8)
        support[15:26, 5:31] = 255
        support[15:26, 36:61] = 255
        visibility = support.copy()
        line = np.asarray([[5.0, 20.0], [60.0, 20.0]])
        local_astro._add_short_polyline_bridges(
            visibility, line, support, maximum_gap_px=10.0, thickness=3,
        )
        self.assertTrue(np.all(visibility[20, 31:36] > 0))
        # A longer unsupported run remains hidden.
        distant = np.zeros((41, 81), dtype=np.uint8)
        distant[15:26, 5:21] = 255
        distant[15:26, 40:61] = 255
        local_astro._add_short_polyline_bridges(
            distant, line, distant.copy(), maximum_gap_px=10.0, thickness=3,
        )
        self.assertTrue(np.all(distant[20, 21:40] == 0))

    def test_selected_video_frame_uses_distinct_cache_key_and_timestamp(self):
        class Capture:
            def isOpened(self): return True
            def set(self, *_args): pass
            def read(self): return True, np.zeros((20, 30, 3), dtype=np.uint8)
            def get(self, prop): return 20.0 if prop == cv2.CAP_PROP_FPS else 0
            def release(self): pass

        with tempfile.TemporaryDirectory() as directory:
            source = str(Path(directory) / "20260710" / "01" / "clip.mp4")
            with mock.patch.object(local_astro, "_open_video", return_value=Capture()), \
                 mock.patch.object(local_astro, "_solve_samples", return_value={"calibration_path": "x"}) as solve:
                result = local_astro.solve_video_frame_local(source, 40, cache_root=directory)
        self.assertEqual(result["calibration_path"], "x")
        self.assertEqual(solve.call_args.kwargs["reference_key"].split("_")[1], "40")
        self.assertEqual(solve.call_args.kwargs["reference_frame_index"], 40)

    def test_solver_normalizes_bgr_selected_frame_before_star_extraction(self):
        frame = np.zeros((20, 30, 3), dtype=np.uint8)

        def extract(average, _samples, maximum_stars):
            self.assertEqual(average.ndim, 2)
            return [[1.0, 1.0]] * 20, average, average

        with tempfile.TemporaryDirectory() as directory:
            with (
                mock.patch.object(local_astro, "_extract_stars", side_effect=extract),
                mock.patch.object(local_astro, "_solve_stars", return_value={"catalog_stars": []}),
                mock.patch.object(local_astro, "_refine_sip_wcs", return_value={"sip_refined": True}),
                mock.patch.object(local_astro, "_persist_calibration", return_value={}),
            ):
                local_astro._solve_samples(
                    frame, None, str(Path(directory) / "input.mp4"),
                    datetime(2026, 7, 10, 1, 0), directory, None,
                )

    def test_constellation_projection_skips_points_rounded_outside_frame(self):
        class BorderWCS:
            def world_to_pixel_values(self, ra, _dec):
                return np.asarray([100.0, 110.0]), np.asarray([1079.6, 1079.7])

        output = np.zeros((1080, 1920, 3), dtype=np.uint8)
        support = np.full((1080, 1920), 255, dtype=np.uint8)
        with mock.patch.object(
            local_astro, "_constellation_lines",
            (np.asarray([[10.0, 20.0], [11.0, 20.0]]),),
        ):
            local_astro._draw_constellation_lines(output, BorderWCS(), 0.0, support)
        self.assertEqual(np.count_nonzero(output), 0)

    def test_constellation_line_interpolation_handles_ra_wrap(self):
        sampled = local_astro._sample_constellation_line(
            np.asarray([[179.0, 10.0], [-179.0, 10.0]])
        )
        self.assertGreater(len(sampled), 2)
        self.assertLess(float(np.max(np.abs(np.diff(sampled[:, 0])))), 2.0)

    def test_constellation_projection_does_not_bridge_large_projected_gap(self):
        points = np.asarray([[10.0, 10.0], [250.0, 10.0], [260.0, 20.0]])
        runs = list(local_astro._constellation_polyline_runs(
            points, np.asarray([True, True, True]), max_pixel_step=100.0
        ))
        self.assertEqual(len(runs), 1)
        np.testing.assert_array_equal(runs[0], points[1:])

    def test_constellation_segment_rejects_unsafe_interior_without_drawing_samples(self):
        support = np.full((180, 320), 255, dtype=np.uint8)
        with mock.patch.object(
            local_astro,
            "_project_constellation_samples",
            return_value=(
                np.asarray([50.0, 1000.0, 60.0]),
                np.asarray([90.0, 90.0, 90.0]),
                np.asarray([True, True, True]),
            ),
        ) as project:
            safe = local_astro._constellation_segment_is_safe(
                object(), np.asarray([[10.0, 20.0], [11.0, 20.0]]),
                0.0, support, 320, 180, 100.0,
            )
        self.assertFalse(safe)
        project.assert_called_once()

    def test_constellation_projection_refines_inaccurate_inverse_sip_estimate(self):
        class ForwardOnlyWCS:
            def world_to_pixel_values(self, ra, dec):
                # Deliberately emulate a 30px inaccurate inverse SIP estimate.
                return np.asarray(ra) + 130.0, np.asarray(dec)

            def pixel_to_world_values(self, x, y):
                return np.asarray(x) - 100.0, np.asarray(y)

        x, y, valid = local_astro._project_sky_with_forward_wcs(
            ForwardOnlyWCS(), np.asarray([10.0, 11.0]), np.asarray([20.0, 20.0])
        )
        np.testing.assert_allclose(x, [110.0, 111.0], atol=0.05)
        np.testing.assert_allclose(y, [20.0, 20.0], atol=0.05)
        self.assertTrue(valid.all())

    def test_fixed_camera_constellation_projection_uses_forward_model_only(self):
        class ForwardOnlyFixedModel(local_astro.FixedCameraPlateModel):
            def __init__(self):
                pass

            def world_to_pixel_values(self, ra, dec):
                return np.asarray(ra, dtype=float) + 100.0, np.asarray(dec, dtype=float)

            def pixel_to_world_values(self, _x, _y):
                raise AssertionError("fixed-camera constellation projection used inverse")

        x, y, valid = local_astro._project_sky_with_forward_wcs(
            ForwardOnlyFixedModel(), np.asarray([10.0, 11.0]), np.asarray([20.0, 20.0])
        )
        np.testing.assert_allclose(x, [110.0, 111.0])
        np.testing.assert_allclose(y, [20.0, 20.0])
        self.assertTrue(valid.all())

    def test_constellation_boundary_rejects_fragment_with_endpoint_outside(self):
        output = np.zeros((180, 320, 3), dtype=np.uint8)
        support = np.full((180, 320), 255, dtype=np.uint8)
        with mock.patch.object(
            local_astro,
            "_constellation_lines",
            (np.asarray([[10.0, 20.0], [12.0, 20.0]]),),
        ), mock.patch.object(
            local_astro,
            "_project_sky_with_forward_wcs",
            return_value=(
                np.asarray([-40.0, 120.0]),
                np.asarray([90.0, 90.0]),
                np.asarray([True, True]),
            ),
        ), mock.patch.object(
            local_astro,
            "_project_constellation_samples",
            return_value=(
                np.asarray([-40.0, 20.0, 80.0]),
                np.asarray([90.0, 90.0, 90.0]),
                np.asarray([True, True, True]),
            ),
        ):
            local_astro._draw_constellation_lines(
                output, object(), 0.0, support
            )
        self.assertEqual(np.count_nonzero(output), 0)

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
