#!/usr/bin/env python3
"""Grow camera calibration using consistent catalog matches across recording times."""
from build_sky_model import mean_frame
import argparse
from datetime import timedelta,datetime
import json
from pathlib import Path
import cv2
import numpy as np
from scipy.spatial import cKDTree,ConvexHull
from scipy.optimize import least_squares
from camera_plate_model import FixedCameraPlateModel
from camera_model_builder import _project,_feature_matrix,_unit_vectors,_support_grid
import local_wideangle_astrometry as astro


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('model',type=Path)
    parser.add_argument('videos',type=Path,nargs='+')
    args=parser.parse_args()
    cv2.setNumThreads(1)
    recording_times=[astro._capture_datetime(str(video)) for video in args.videos]
    if len(set(recording_times)) < 3 or (max(recording_times)-min(recording_times)).total_seconds() < 900:
        raise ValueError('15分以上にわたる異なる時刻の録画を3本以上指定してください')
    p=json.loads(args.model.read_text())
    reference=datetime.fromisoformat(p['reference_datetime'])
    catalog=p['catalog_stars']
    ra=np.array([s['ra_deg'] for s in catalog]);dec=np.array([s['dec_deg'] for s in catalog])
    mag=np.array([s['metadata']['MAG'] for s in catalog])
    samples=[]
    for video in args.videos:
        frame,_=mean_frame(video,30)
        stars,_,_=astro._extract_stars(cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY),None,maximum_stars=1500,exclude_lower_region=False,build_diagnostic=False)
        delta=(astro._capture_datetime(str(video))+timedelta(seconds=30)-reference).total_seconds()/astro.SIDEREAL_DAY_SECONDS*360
        world=_unit_vectors((ra-delta)%360,dec)
        base=_project(np.r_[p['stg_parameters'],[0,0]],world,p['width'],p['height'],0)
        samples.append((np.asarray(stars),base,delta))
    for iteration in range(10):
        predicted=[]; offsets=[]; candidates=[]; nearests=[]
        for observed,base,delta in samples:
            x,y=FixedCameraPlateModel(p).world_to_pixel_values((ra-delta)%360,dec)
            points=np.column_stack((x,y)); valid=np.isfinite(points).all(axis=1)&(x>=8)&(x<1912)&(y>=8)&(y<1072)
            ids=np.flatnonzero(valid)
            distances=np.full(len(ra),np.inf);nearest=np.zeros(len(ra),int)
            distances[ids],nearest[ids]=cKDTree(observed).query(points[ids])
            back=cKDTree(points[ids]).query(observed)[1]
            mutual=np.zeros(len(ra),bool);mutual[ids]=back[nearest[ids]]==np.arange(len(ids))
            offsets.append(observed[nearest]-points)
            candidates.append(valid&mutual&(distances<max(8,45-iteration*4)))
            nearests.append(nearest)
        admitted=np.logical_and.reduce(candidates)
        # Static artifacts cannot masquerade as a sidereally moving catalog
        # star in three separated recordings. Also require consistent offsets.
        spread=np.max(np.linalg.norm(np.stack(offsets)-np.mean(offsets,axis=0),axis=2),axis=0)
        admitted &= spread<max(2.5,7-iteration*.5)
        # Bright stars establish expansion; dim stars refine only near the fit.
        admitted &= (mag<5.8)|(np.max(np.linalg.norm(np.stack(offsets),axis=2),axis=0)<5)
        if admitted.sum()<50:
            raise ValueError(f'時刻をまたいで対応する星が不足: {admitted.sum()}')
        degree=4 if iteration<4 else 5
        features=np.concatenate([_feature_matrix(base[admitted],1920,1080,degree) for _,base,_ in samples])
        target=np.concatenate([observed[nearest[admitted]]-base[admitted] for (observed,base,_),nearest in zip(samples,nearests)])
        coefficients=np.linalg.lstsq(features,target,rcond=None)[0]
        fit=least_squares(lambda c:(features@c.reshape(-1,2)-target).ravel(),coefficients.ravel(),loss='soft_l1',f_scale=.6,max_nfev=150)
        coefficients=fit.x.reshape(-1,2)
        error=np.linalg.norm(features@coefficients-target,axis=1)
        p['polynomial_degree']=degree;p['correction_coefficients']=coefficients.tolist()
        print(f'恒星の時刻照合 {iteration+1}: {admitted.sum()}星 / p95 {np.percentile(error,95):.2f}px',flush=True)
    actual=np.concatenate([observed[nearest[admitted]] for (observed,_,_),nearest in zip(samples,nearests)])
    hull=actual[ConvexHull(actual).vertices]
    support,fraction=_support_grid(1920,1080,hull.tolist())
    p.update(support_grid=support,support_fraction=fraction,model_revision='obs-camera-temporal-catalog-v2',
        fit_stats=dict(matched_stars=int(admitted.sum()),observations=len(error),residual_median_px=float(np.median(error)),residual_p95_px=float(np.percentile(error,95))),
        enabled=bool(fraction>=.70 and np.percentile(error,95)<=3),target_met=bool(fraction>=.80 and np.percentile(error,95)<=2),
        source_videos=[str(v) for v in args.videos],
        fitted_correspondences=[dict(ra_deg=float(r),dec_deg=float(d)) for r,d in zip(ra[admitted],dec[admitted])])
    args.model.write_text(json.dumps(p,ensure_ascii=False,indent=2))
    print(json.dumps(p['fit_stats']), 'support',fraction,'enabled',p['enabled'])

if __name__=='__main__':main()
