import json
from pathlib import Path
import unittest

import numpy as np

from camera_plate_model import FixedCameraPlateModel


class FixedCameraPlateModelTests(unittest.TestCase):
    def test_forward_inverse_round_trip(self):
        payload = {
            "model_type": "fixed-camera-stg-poly",
            "width": 1920,
            "height": 1080,
            "polynomial_degree": 2,
            "stg_parameters": [0.1, -0.2, 0.05, np.log(1200), np.log(1180), 960, 540],
            "correction_coefficients": [
                [2.0, -1.0], [1.0, 0.3], [-0.2, 0.8],
                [0.4, -0.1], [0.2, 0.3], [-0.3, 0.2],
            ],
            "residual_grid_x": [0.0, 1920.0],
            "residual_grid_y": [0.0, 1080.0],
            "residual_grid": [
                [[1.0, -0.5], [2.0, 0.5]],
                [[-1.0, 1.0], [0.5, -1.0]],
            ],
            "micro_correction_degree": 1,
            "micro_correction_coefficients": [[0.2, -0.1], [0.1, 0.2], [-0.2, 0.1]],
        }
        model = FixedCameraPlateModel(payload)
        x = np.asarray([300.0, 700.0, 960.0, 1400.0, 1750.0])
        y = np.asarray([180.0, 820.0, 540.0, 260.0, 850.0])
        ra, dec = model.pixel_to_world_values(x, y)
        actual_x, actual_y = model.world_to_pixel_values(ra, dec)
        np.testing.assert_allclose(actual_x, x, atol=1e-4)
        np.testing.assert_allclose(actual_y, y, atol=1e-4)

    def test_rejects_wrong_coefficient_shape(self):
        with self.assertRaises(ValueError):
            FixedCameraPlateModel({
                "model_type": "fixed-camera-stg-poly", "width": 2, "height": 2,
                "polynomial_degree": 1, "stg_parameters": [0] * 7,
                "correction_coefficients": [[0, 0]],
            })


if __name__ == "__main__":
    unittest.main()
