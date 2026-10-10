#!/usr/bin/env python3
"""Produce independent, inspectable camera-model checks; never enable OBS."""
from build_sky_model import mean_frame
import argparse
import hashlib
from datetime import timedelta
import json
from pathlib import Path
import cv2
import numpy as np
from scipy.spatial import cKDTree
from camera_plate_model import FixedCameraPlateModel
import local_wideangle_astrometry as astro
from sky_overlay import render_layer


def verify(model_path, video, output, seconds=30):
    payload=json.loads(model_path.read_text())
    output.mkdir(parents=True,exist_ok=True)
    frame,_=mean_frame(video,seconds)
    when=astro._capture_datetime(str(video))+timedelta(seconds=seconds)
    rgba=render_layer(payload,when)
    alpha=rgba[:,:,3:4]/255.
    composite=(frame*(1-alpha)+cv2.cvtColor(rgba[:,:,:3],cv2.COLOR_RGB2BGR)*alpha).astype(np.uint8)
    name=when.strftime('%Y%m%d_%H%M%S')
    cv2.imwrite(str(output/f'{name}_constellations.png'),composite)
    stars,_,_=astro._extract_stars(cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY),None,maximum_stars=1500,exclude_lower_region=False,build_diagnostic=False)
    records=payload['fitted_correspondences']
    ra=np.array([s['ra_deg'] for s in records]);dec=np.array([s['dec_deg'] for s in records])
    delta=(when-datetime_from_model(payload)).total_seconds()/astro.SIDEREAL_DAY_SECONDS*360
    x,y=FixedCameraPlateModel(payload).world_to_pixel_values((ra-delta)%360,dec)
    predicted=np.column_stack((x,y)); h,w=frame.shape[:2]
    valid=np.isfinite(predicted).all(axis=1)&(x>6)&(x<w-6)&(y>6)&(y<h-6)
    predicted=predicted[valid]
    detected=np.asarray(stars)
    distance,nearest=cKDTree(detected).query(predicted)
    reverse=cKDTree(predicted).query(detected)[1]
    mutual=reverse[nearest]==np.arange(len(predicted))
    # Report every expected star, including missing ones. Accepted residuals
    # alone can disguise a poorly aligned model in a dense/noisy image.
    matched=mutual&(distance<5)
    marked=frame.copy()
    for p,d,good in zip(predicted,detected[nearest],matched):
        cv2.drawMarker(marked,tuple(np.rint(p).astype(int)),(255,80,255),cv2.MARKER_CROSS,9,1)
        if good:
            cv2.circle(marked,tuple(np.rint(d).astype(int)),5,(80,255,80),1)
    cv2.imwrite(str(output/f'{name}_star_checks.png'),marked)
    report=dict(model_sha256=hashlib.sha256(model_path.read_bytes()).hexdigest(),video=str(video),time=when.isoformat(),expected_stars=len(predicted),matched_stars=int(matched.sum()),match_fraction=float(matched.mean()),
        all_nearest_median_px=float(np.median(distance)),all_nearest_p95_px=float(np.percentile(distance,95)),
        matched_median_px=float(np.median(distance[matched])) if matched.any() else None,
        matched_p95_px=float(np.percentile(distance[matched],95)) if matched.any() else None,
        outer_matched_stars=int(np.count_nonzero(matched&((predicted[:,0]<w*.2)|(predicted[:,0]>w*.8)|(predicted[:,1]<h*.2)|(predicted[:,1]>h*.8)))),
        visually_verified=False)
    (output/f'{name}_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)
    return report


def datetime_from_model(payload):
    from datetime import datetime
    return datetime.fromisoformat(payload['reference_datetime'])


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('model',type=Path)
    parser.add_argument('videos',type=Path,nargs='+')
    parser.add_argument('--output-dir',type=Path,default=Path(__file__).with_name('validation'))
    args=parser.parse_args()
    cv2.setNumThreads(1)
    for video in args.videos:
        verify(args.model,video,args.output_dir)

if __name__=='__main__':
    main()
