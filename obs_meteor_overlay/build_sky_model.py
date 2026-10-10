#!/usr/bin/env python3
"""Fit a private OBS camera model from raw recordings, without touching old UI."""
import os
for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[name] = '1'
import argparse
from datetime import timedelta
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import urllib.request
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import cv2
import numpy as np
from astropy.io import fits
from camera_plate_model import FixedCameraPlateModel
import local_wideangle_astrometry as astro


def ensure_indexes(cache_root):
    directory = Path(cache_root) / 'indexes' / '4100'
    directory.mkdir(parents=True, exist_ok=True)
    for scale in range(15, 20):
        path = directory / f'index-41{scale}.fits'
        if path.exists():
            continue
        url = f'https://data.astrometry.net/4100/{path.name}'
        temporary = path.with_suffix('.fits.tmp')
        try:
            with urllib.request.urlopen(url, timeout=60) as response, temporary.open('wb') as output:
                while chunk := response.read(65536):
                    output.write(chunk)
            with fits.open(temporary) as hdus:
                hdus.verify('exception')
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


def mean_frame(video, seconds, half_window=2):
    cap = cv2.VideoCapture(str(video))
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            raise ValueError('録画のフレームレートを読めません')
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, round((seconds-half_window)*fps)))
        total = None
        count = 0
        for _ in range(round(half_window*2*fps)+1):
            ok, frame = cap.read()
            if not ok:
                break
            if total is None:
                total = np.zeros_like(frame, dtype=np.float32)
            total += frame
            count += 1
        if not count:
            raise ValueError('録画から画像を読み出せません')
        return np.clip(total/count, 0, 255).astype(np.uint8), fps
    finally:
        cap.release()


def full_catalog(index_path, center_ra, center_dec):
    """Read the all-sky Tycho index coordinates in their documented native format."""
    with fits.open(index_path) as hdus:
        tables = {h.columns.names[0]: h for h in hdus[1:] if hasattr(h, 'columns')}
        header = tables['kdtree_header_stars'].header
        if header['KDT_DATA'] != 'u32' or header['KDT_NDIM'] != 3:
            raise ValueError('Unsupported Tycho index coordinate encoding')
        byte_order = header['ENDIAN'].strip()
        if byte_order not in {'04:03:02:01', '01:02:03:04'}:
            raise ValueError('Unsupported Tycho index byte order')
        endian = '<' if byte_order == '04:03:02:01' else '>'
        bounds = np.frombuffer(tables['kdtree_range_stars'].data.tobytes(), dtype=endian+'f8')
        encoded = np.frombuffer(tables['kdtree_data_stars'].data.tobytes(), dtype=endian+'u4').reshape(-1,3)
        vectors = encoded/bounds[-1]+bounds[:3]
        vectors /= np.linalg.norm(vectors,axis=1)[:,None]
        ra = np.rad2deg(np.arctan2(vectors[:,1],vectors[:,0]))%360
        dec = np.rad2deg(np.arcsin(vectors[:,2]))
        magnitudes = hdus[-1].data['MAG']
        center = FixedCameraPlateModel._unit_vectors(np.array([center_ra]),np.array([center_dec]))[0]
        keep = (vectors@center > np.cos(np.deg2rad(85))) & (magnitudes < 7)
        return [dict(ra_deg=float(r),dec_deg=float(d),metadata={'MAG':float(m)}) for r,d,m in zip(ra[keep],dec[keep],magnitudes[keep])]


def fit_full_camera(paths, stars, catalog, width, height, stamp):
    from astropy.wcs import WCS
    from scipy.optimize import least_squares
    from scipy.spatial import cKDTree, ConvexHull
    from camera_model_builder import _initial_parameters, _project, _feature_matrix, _unit_vectors, _support_grid
    with fits.open(paths['wcs']) as hdus:
        seed = WCS(hdus[0].header,relax=True)
    ra=np.array([s['ra_deg'] for s in catalog]); dec=np.array([s['dec_deg'] for s in catalog])
    world=_unit_vectors(ra,dec)
    parameters=_initial_parameters(seed,width,height,0)[:7]
    base=_project(np.r_[parameters,[0,0]],world,width,height,0)
    observed=np.asarray(stars)
    ox,oy=seed.world_to_pixel_values(ra,dec)
    predicted=np.column_stack((ox,oy))
    finite=np.isfinite(base).all(axis=1) & (np.abs(base[:,0]-width/2)<width*1.3) & (np.abs(base[:,1]-height/2)<height*1.7)
    indices=np.flatnonzero(finite)
    obs_tree=cKDTree(observed)
    def pairs(points,limit):
        valid=indices[np.isfinite(points[indices]).all(axis=1)]
        distance,nearest=obs_tree.query(points[valid])
        reverse=cKDTree(points[valid]).query(observed)[1]
        keep=(distance<limit) & (reverse[nearest]==np.arange(len(valid)))
        return valid[keep],nearest[keep]
    degree=3
    coefficients=None
    for degree,limit in [(2,6),(2,8),(3,10),(3,10),(4,8),(4,7),(5,6),(5,5)]:
        ci,oi=pairs(predicted,limit)
        if len(ci)<30:
            raise ValueError(f'対応する恒星が不足: {len(ci)}')
        features=_feature_matrix(base[ci],width,height,degree)
        target=observed[oi]-base[ci]
        fit=least_squares(lambda c:(features@c.reshape(-1,2)-target).ravel(),np.linalg.lstsq(features,target,rcond=None)[0].ravel(),loss='soft_l1',f_scale=.8,max_nfev=100)
        coefficients=fit.x.reshape(-1,2)
        residual=np.linalg.norm(features@coefficients-target,axis=1)
        keep=residual<3.0
        if keep.sum()>=len(coefficients)+10:
            coefficients=np.linalg.lstsq(features[keep],target[keep],rcond=None)[0]
        predicted=base+_feature_matrix(base,width,height,degree)@coefficients
        print(f'カメラ投影次数{degree}: 対応{len(ci)} / 正常{keep.sum()} / p95={np.percentile(residual[keep],95):.2f}px',flush=True)
    ci,oi=pairs(predicted,3)
    # Final forward fit directly to observed centroids avoids SIP inverse
    # divergence at the wide rim; the serialized adapter is the fitted model.
    features=_feature_matrix(base[ci],width,height,degree)
    coefficients=np.linalg.lstsq(features,observed[oi]-base[ci],rcond=None)[0]
    error=np.linalg.norm(base[ci]+features@coefficients-observed[oi],axis=1)
    hull=observed[oi][ConvexHull(observed[oi]).vertices]
    support,fraction=_support_grid(width,height,hull.tolist())
    model=dict(model_type='fixed-camera-stg-poly',width=width,height=height,polynomial_degree=degree,
        stg_parameters=parameters.tolist(),correction_coefficients=coefficients.tolist(),
        reference_datetime=stamp.isoformat(),support_grid=support,support_fraction=fraction,
        enabled=False,target_met=False,ready_for_temporal_refinement=True,
        model_revision='obs-camera-direct-stg-v1',model_label='OBS CAMERA FULL FIELD',wcs_path=str(paths['wcs']),
        catalog_stars=catalog,fit_stats=dict(matched_stars=len(ci),residual_median_px=float(np.median(error)),residual_p95_px=float(np.percentile(error,95))),
        fitted_correspondences=[dict(ra_deg=float(ra[c]),dec_deg=float(dec[c]),x=float(observed[o,0]),y=float(observed[o,1])) for c,o in zip(ci,oi)])
    return model


def solve_seed(video, *, cache_root, center_seconds, progress_callback=None, **_):
    emit = progress_callback or print
    frame, fps = mean_frame(video, center_seconds)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    stamp = astro._capture_datetime(video) + timedelta(seconds=center_seconds)
    paths = astro._calibration_paths(video, w, h, cache_root, 'obs_clean_mean')
    stars, diagnostic, _ = astro._extract_stars(gray, None, maximum_stars=1000, exclude_lower_region=False)
    cv2.imwrite(str(paths['reference']), frame)
    cv2.imwrite(str(paths['diagnostic']), diagnostic)
    # Do not subtract a same-time temporal median: it removes nearly stationary
    # stars and leaves compression noise as solver candidates in this camera.
    emit(f'生映像の平均から星を抽出: {len(stars)}個')
    central = [p for p in stars if w*.25<p[0]<w*.75 and h*.2<p[1]<h*.8][:65]
    request = dict(stars=central, width=w, height=h, date_obs=stamp.isoformat(),
                   output_wcs=str(paths['wcs']), index_cache=str(paths['index_cache']),
                   index_scales=[15,16,17,18,19], scale_lower_arcsec=100,
                   scale_upper_arcsec=300, sip_order=2, sip_inverse_order=2,
                   positional_noise_pixels=5, distractor_ratio=.3,
                   maximum_matches=1, maximum_quads=20000)
    with tempfile.TemporaryDirectory(prefix='obs-sky-solve-') as temporary:
        rp, out = Path(temporary)/'request.json', Path(temporary)/'response.json'
        rp.write_text(json.dumps(request))
        emit(f'中央の明るい星 {len(central)}個で絶対方向を照合中')
        subprocess.run([sys.executable,str(astro.__file__).replace('local_wideangle_astrometry.py','local_wideangle_worker.py'),str(rp),str(out)], check=False, timeout=240)
        response = json.loads(out.read_text())
    if not response.get('matched'):
        raise ValueError(response.get('error','照合結果がありません'))
    response['catalog_stars'] = full_catalog(paths['index_cache']/'4100/index-4115.fits',response['center_ra_deg'],response['center_dec_deg'])
    emit(f"全天星表から {len(response['catalog_stars'])}個を参照し、画面周辺も含めて歪みを推定中")
    payload = fit_full_camera(paths,stars,response['catalog_stars'],w,h,stamp)
    root = Path(cache_root)/'camera_models'/'obs-camera-full-field'
    root.mkdir(parents=True,exist_ok=True)
    model_path=root/'camera_model.json'
    model_path.write_text(json.dumps(payload,ensure_ascii=False,indent=2))
    return dict(success=True,model_path=str(model_path),enabled=payload['enabled'],support_fraction=payload['support_fraction'],**payload['fit_stats'])



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('video', type=Path)
    parser.add_argument('--cache-root', type=Path, default=Path(__file__).with_name('calibration'))
    parser.add_argument('--center-seconds', type=float, default=30)
    args = parser.parse_args()
    cv2.setNumThreads(1)
    ensure_indexes(args.cache_root)
    result = solve_seed(str(args.video.resolve()),cache_root=str(args.cache_root.resolve()),center_seconds=args.center_seconds,progress_callback=lambda m:print(m,flush=True))
    args.cache_root.mkdir(parents=True,exist_ok=True)
    (args.cache_root/'candidate_result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)

    return 0 if result['success'] else 1

if __name__ == '__main__':
    raise SystemExit(main())
