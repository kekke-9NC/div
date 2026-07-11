"""Isolated backend for the third-party ``astrometry`` solver package.

The application already has an ``astrometry.py`` module.  Running this file in
an isolated subprocess keeps that historical module from shadowing the local
Astrometry.net Python package and also releases all solver memory after a solve.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import tempfile


def _import_solver_backend():
    repository = Path(__file__).resolve().parent
    filtered = []
    for item in sys.path:
        try:
            resolved = Path(item or os.getcwd()).resolve()
        except OSError:
            resolved = None
        if resolved != repository:
            filtered.append(item)
    sys.path[:] = filtered
    os.chdir(tempfile.gettempdir())
    import astrometry as backend  # type: ignore

    if Path(getattr(backend, "__file__", "")).resolve() == repository / "astrometry.py":
        raise ImportError("third-party astrometry package was shadowed by the app module")
    return backend


def main(request_path: str, response_path: str) -> int:
    request = json.loads(Path(request_path).read_text(encoding="utf-8"))
    response = {"matched": False}
    try:
        backend = _import_solver_backend()
        from astropy.io import fits

        scales = {int(value) for value in request.get("index_scales", [15, 16, 17, 18, 19])}
        indexes = backend.series_4100.index_files(
            cache_directory=Path(request["index_cache"]), scales=scales
        )
        position = request.get("position_hint")
        position_hint = None
        if position:
            position_hint = backend.PositionHint(
                ra_deg=float(position["ra_deg"]),
                dec_deg=float(position["dec_deg"]),
                radius_deg=float(position.get("radius_deg", 20.0)),
            )
        parameters = backend.SolutionParameters(
            sip_order=int(request.get("sip_order", 5)),
            sip_inverse_order=int(request.get("sip_inverse_order", 5)),
            positional_noise_pixels=float(request.get("positional_noise_pixels", 3.0)),
            distractor_ratio=float(request.get("distractor_ratio", 0.65)),
            maximum_matches=int(request.get("maximum_matches", 2)),
            maximum_quads=int(request.get("maximum_quads", 200000)),
        )
        with backend.Solver(indexes) as solver:
            solution = solver.solve(
                stars=request["stars"],
                size_hint=backend.SizeHint(
                    lower_arcsec_per_pixel=float(request.get("scale_lower_arcsec", 120.0)),
                    upper_arcsec_per_pixel=float(request.get("scale_upper_arcsec", 260.0)),
                ),
                position_hint=position_hint,
                solution_parameters=parameters,
            )
        if not solution.has_match():
            response["error"] = "no astrometric match"
        else:
            match = solution.best_match()
            wcs = match.astropy_wcs()
            source_header = wcs.to_header(relax=True)
            hdu = fits.PrimaryHDU()
            for card in source_header.cards:
                hdu.header.append(card, end=True)
            hdu.header["IMAGEW"] = int(request["width"])
            hdu.header["IMAGEH"] = int(request["height"])
            hdu.header["DATE-OBS"] = str(request["date_obs"])
            hdu.header["CALTYPE"] = ("LOCAL-SIP", "Local wide-angle calibration")
            hdu.header.add_history("Local Astrometry.net index solve; no web API used")
            output_wcs = Path(request["output_wcs"])
            output_wcs.parent.mkdir(parents=True, exist_ok=True)
            hdu.writeto(output_wcs, overwrite=True, output_verify="silentfix")
            response.update({
                "matched": True,
                "wcs_path": str(output_wcs),
                "center_ra_deg": float(match.center_ra_deg),
                "center_dec_deg": float(match.center_dec_deg),
                "scale_arcsec_per_pixel": float(match.scale_arcsec_per_pixel),
                "logodds": float(match.logodds),
                "catalog_stars": [
                    {
                        "ra_deg": float(star.ra_deg),
                        "dec_deg": float(star.dec_deg),
                        "metadata": star.metadata,
                    }
                    for star in match.stars
                ],
            })
    except Exception as exc:
        response["error"] = f"{type(exc).__name__}: {exc}"
    Path(response_path).write_text(
        json.dumps(response, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return 0 if response.get("matched") else 2


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("usage: local_wideangle_worker.py request.json response.json")
    raise SystemExit(main(sys.argv[1], sys.argv[2]))
