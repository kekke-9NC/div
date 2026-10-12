"""Conservative, learned foreground gate before expensive event exports.

The gate never labels an accepted candidate a meteor. It only rejects highly
confident background/non-meteor events; the full event classifier makes every
remaining decision. Training and runtime share this representation.
"""
from __future__ import annotations
import cv2
import numpy as np
from universal_meteor_model import _crop_square, _uint8_median
from event_boost_model import EventBoostClassifier
from temporal_statistics import temporal_median, half_integer_percentiles

GATE_PREPROCESS_VERSION = 2
BACKGROUND_BRIGHTNESS_MIN = 60.


def extract_noise_gate_features(frames, rect=None, frame_rate=25.):
    if not frames:
        raise ValueError('Noise gate needs chronological frames')
    stack = np.stack([_crop_square(frame, rect, 64) for frame in frames])
    values = stack.astype(np.float32)
    background = temporal_median(stack).astype(np.float32)
    delta = values - background
    positive = np.maximum(delta, 0.)
    maximum = positive.max(axis=0)
    noise = (max(.5, _uint8_median(np.abs(np.diff(stack.astype(np.int16), axis=0)).astype(np.uint8)) / .954)
             if len(values) > 1 else .5)
    result = [len(values)/max(1.,frame_rate), float(len(values)), noise]
    for array in (background, maximum, positive, np.maximum(-delta,0.)):
        result.extend([float(array.mean()),float(array.std()), *half_integer_percentiles(array,[0,50,90,99,99.9,100]).tolist()])
    for level in (1.,2.,4.,8.):
        result.extend([float((maximum>level*noise).mean()),float((positive>level*noise).mean())])
    for curve in (positive.mean(axis=(1,2)),positive.max(axis=(1,2)),(positive>3*noise).mean(axis=(1,2))):
        peak=max(float(curve.max()),1e-6); normalized=curve/peak
        result.extend([float(curve.mean()),float(curve.std()),peak,float(np.std(np.diff(curve))) if len(curve)>1 else 0.,float((normalized>.5).mean()),float((normalized>.2).mean())])
        for level in (.2,.5):
            active=normalized>level
            result.append(float(np.count_nonzero(np.diff(np.r_[False,active,False].astype(int))==1)))
    mask=(maximum>max(noise*4.,float(maximum.max())*.3)).astype(np.uint8)
    n,_,stats,_=cv2.connectedComponentsWithStats(mask,8)
    areas=stats[1:,cv2.CC_STAT_AREA]
    result.extend([float(n-1),float(max(areas,default=0)),float(mask.sum())])
    yy,xx=np.indices((64,64),dtype=np.float32);mass=np.maximum(positive.sum(axis=(1,2)),1e-6)
    positions=np.column_stack([(positive*xx).sum(axis=(1,2))/mass,(positive*yy).sum(axis=(1,2))/mass])
    active=mass>max(float(mass.max())*.3,noise*5.)
    if active.sum()>=3:
        times=np.flatnonzero(active)/max(1.,frame_rate);xy=positions[active];c=np.polyfit(times,xy,1)
        residual=xy-(times[:,None]*c[0]+c[1]);variance=float(np.square(xy-xy.mean(axis=0)).sum())
        result.extend([float(np.linalg.norm(c[0])),float(np.linalg.norm(xy[-1]-xy[0])),float(np.sqrt(np.square(residual).mean())),1.-float(np.square(residual).sum())/max(1e-6,variance)])
    else:result.extend([0.,0.,0.,0.])
    return np.nan_to_num(np.asarray(result,dtype=np.float32),nan=0.,posinf=1e6,neginf=-1e6)


class EventNoiseGate:
    def __init__(self,checkpoint):
        if checkpoint.get('gate_preprocess_version') != GATE_PREPROCESS_VERSION:
            raise ValueError('Unsupported foreground gate representation')
        self.classifier=EventBoostClassifier(checkpoint)
        self.threshold=float(checkpoint['gate_threshold'])
        self.brightness_min = checkpoint.get('gate_brightness_min', BACKGROUND_BRIGHTNESS_MIN)
        if not 0.<self.threshold<1.:
            raise ValueError('Foreground gate threshold must be between zero and one')

    def evaluate(self,frames,rect=None,frame_rate=25.):
        features=extract_noise_gate_features(frames,rect,frame_rate)
        # Dark-sky faint meteors keep the full classifier path. This is a
        # radiometric condition on the actual crop, never a clock/date rule.
        if self.brightness_min is not None and features[6] < self.brightness_min:
            return 1.,self.threshold
        probability=float(self.classifier.predict_proba(features)[0])
        return probability,self.threshold
