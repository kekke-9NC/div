"""Reusable astrometric model for a fixed ultra-wide camera.

The model uses a stereographic camera projection followed by a low-order,
detector-space polynomial.  Unlike a per-frame SIP fit, the lens parameters
are shared by every night; only right ascension advances with sidereal time.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
from astropy.coordinates import SkyCoord
from astropy import units as u
from scipy.spatial.transform import Rotation
from scipy.interpolate import RegularGridInterpolator


MODEL_TYPE = "fixed-camera-stg-poly"


class FixedCameraPlateModel:
    """Small WCS-compatible adapter backed by a fixed-camera JSON model."""

    is_celestial = True
    pixel_n_dim = 2
    world_n_dim = 2
    sip = None

    def __init__(self, payload: Dict[str, Any]):
        if payload.get("model_type") != MODEL_TYPE:
            raise ValueError(f"Unsupported camera model: {payload.get('model_type')}")
        self.payload = payload
        self.width = int(payload["width"])
        self.height = int(payload["height"])
        self.degree = int(payload["polynomial_degree"])
        self.parameters = np.asarray(payload["stg_parameters"], dtype=float)
        self.coefficients = np.asarray(payload["correction_coefficients"], dtype=float)
        expected_terms = (self.degree + 1) * (self.degree + 2) // 2
        if self.parameters.shape != (7,) or self.coefficients.shape != (expected_terms, 2):
            raise ValueError("Invalid fixed-camera parameter dimensions")
        self._rotation = Rotation.from_rotvec(self.parameters[:3]).as_matrix()
        self._fx, self._fy = np.exp(self.parameters[3:5])
        self._cx, self._cy = self.parameters[5:7]
        self._residual_interpolators = None
        if payload.get("residual_grid") is not None:
            grid = np.asarray(payload["residual_grid"], dtype=float)
            grid_x = np.asarray(payload["residual_grid_x"], dtype=float)
            grid_y = np.asarray(payload["residual_grid_y"], dtype=float)
            if grid.shape != (len(grid_y), len(grid_x), 2):
                raise ValueError("Invalid residual-grid dimensions")
            self._residual_interpolators = tuple(
                RegularGridInterpolator(
                    (grid_y, grid_x), grid[:, :, axis], bounds_error=False,
                    fill_value=None,
                )
                for axis in range(2)
            )
        self._micro_degree = payload.get("micro_correction_degree")
        self._micro_coefficients = None
        if self._micro_degree is not None:
            self._micro_degree = int(self._micro_degree)
            self._micro_coefficients = np.asarray(
                payload["micro_correction_coefficients"], dtype=float
            )
            micro_terms = (self._micro_degree + 1) * (self._micro_degree + 2) // 2
            if self._micro_coefficients.shape != (micro_terms, 2):
                raise ValueError("Invalid micro-correction dimensions")

    @staticmethod
    def _unit_vectors(ra_deg: np.ndarray, dec_deg: np.ndarray) -> np.ndarray:
        ra = np.deg2rad(ra_deg)
        dec = np.deg2rad(dec_deg)
        cos_dec = np.cos(dec)
        return np.column_stack((
            cos_dec * np.cos(ra), cos_dec * np.sin(ra), np.sin(dec)
        ))

    def _features_for_degree(self, points: np.ndarray, degree: int) -> np.ndarray:
        x = (points[:, 0] - self.width / 2) / (self.width / 2)
        y = (points[:, 1] - self.height / 2) / (self.height / 2)
        columns = []
        for total in range(degree + 1):
            for x_power in range(total, -1, -1):
                columns.append((x ** x_power) * (y ** (total - x_power)))
        return np.column_stack(columns)

    def _features(self, points: np.ndarray) -> np.ndarray:
        return self._features_for_degree(points, self.degree)

    def _distort(self, base_pixels: np.ndarray) -> np.ndarray:
        distorted = base_pixels + self._features(base_pixels) @ self.coefficients
        if self._residual_interpolators is not None:
            sample_points = np.column_stack((distorted[:, 1], distorted[:, 0]))
            residual = np.column_stack(tuple(
                interpolator(sample_points)
                for interpolator in self._residual_interpolators
            ))
            distorted = distorted + residual
        if self._micro_coefficients is not None:
            distorted = distorted + self._features_for_degree(
                distorted, self._micro_degree
            ) @ self._micro_coefficients
        return distorted

    def world_to_pixel_values(self, ra_deg, dec_deg) -> Tuple[np.ndarray, np.ndarray]:
        ra, dec = np.broadcast_arrays(
            np.asarray(ra_deg, dtype=float), np.asarray(dec_deg, dtype=float)
        )
        shape = ra.shape
        world = self._unit_vectors(ra.ravel(), dec.ravel())
        camera = world @ self._rotation.T
        rho = np.hypot(camera[:, 0], camera[:, 1])
        theta = np.arctan2(rho, camera[:, 2])
        radial = 2.0 * np.tan(theta / 2.0)
        direction_x = np.divide(camera[:, 0], rho, out=np.zeros_like(rho), where=rho > 1e-12)
        direction_y = np.divide(camera[:, 1], rho, out=np.zeros_like(rho), where=rho > 1e-12)
        base = np.column_stack((
            self._cx + self._fx * radial * direction_x,
            self._cy + self._fy * radial * direction_y,
        ))
        distorted = self._distort(base)
        return distorted[:, 0].reshape(shape), distorted[:, 1].reshape(shape)

    def _undistort(self, pixels: np.ndarray) -> np.ndarray:
        base = pixels.copy()
        step = 0.25
        for _ in range(12):
            current = self._distort(base)
            error = current - pixels
            if not len(error) or float(np.nanmax(np.abs(error))) < 1e-5:
                break
            dx = np.zeros_like(base)
            dy = np.zeros_like(base)
            dx[:, 0] = step
            dy[:, 1] = step
            jx = (self._distort(base + dx) - current) / step
            jy = (self._distort(base + dy) - current) / step
            determinant = jx[:, 0] * jy[:, 1] - jy[:, 0] * jx[:, 1]
            usable = np.isfinite(determinant) & (np.abs(determinant) > 1e-9)
            update_x = np.zeros(len(base))
            update_y = np.zeros(len(base))
            update_x[usable] = (
                jy[usable, 1] * error[usable, 0]
                - jy[usable, 0] * error[usable, 1]
            ) / determinant[usable]
            update_y[usable] = (
                -jx[usable, 1] * error[usable, 0]
                + jx[usable, 0] * error[usable, 1]
            ) / determinant[usable]
            base[:, 0] -= np.clip(update_x, -50.0, 50.0)
            base[:, 1] -= np.clip(update_y, -50.0, 50.0)
        return base

    def pixel_to_world_values(self, x, y) -> Tuple[np.ndarray, np.ndarray]:
        px, py = np.broadcast_arrays(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
        shape = px.shape
        base = self._undistort(np.column_stack((px.ravel(), py.ravel())))
        nx = (base[:, 0] - self._cx) / self._fx
        ny = (base[:, 1] - self._cy) / self._fy
        radial = np.hypot(nx, ny)
        theta = 2.0 * np.arctan(radial / 2.0)
        scale = np.divide(np.sin(theta), radial, out=np.ones_like(radial), where=radial > 1e-12)
        camera = np.column_stack((nx * scale, ny * scale, np.cos(theta)))
        world = camera @ self._rotation
        world /= np.linalg.norm(world, axis=1, keepdims=True)
        ra = np.rad2deg(np.arctan2(world[:, 1], world[:, 0])) % 360.0
        dec = np.rad2deg(np.arcsin(np.clip(world[:, 2], -1.0, 1.0)))
        return ra.reshape(shape), dec.reshape(shape)

    def pixel_to_world(self, x, y) -> SkyCoord:
        ra, dec = self.pixel_to_world_values(x, y)
        return SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs")

    def world_to_pixel(self, sky: SkyCoord) -> Tuple[np.ndarray, np.ndarray]:
        return self.world_to_pixel_values(sky.icrs.ra.deg, sky.icrs.dec.deg)
