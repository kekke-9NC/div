# tracking.py

import numpy as np
from collections import deque
import threading
from typing import List, Tuple, Deque, Dict, Any, Optional

import config

tracked_objects: List['TrackedObject'] = []
_past_detections_deque: Deque[Tuple[int, int]] = deque(maxlen=config.PAST_DETECTIONS_MAXLEN)
_past_detections_lock = threading.Lock()


class TrackedObject:
    """
    検出された移動オブジェクトを追跡・分類するためのクラス。
    """
    def __init__(self, initial_position: Tuple[int, int], initial_frame: int):
        self.positions: Deque[Tuple[int, int]] = deque(maxlen=config.TRACKED_OBJECT_POSITIONS_MAXLEN)
        self.positions.append(initial_position)
        self.start_frame: int = initial_frame
        self.last_frame: int = initial_frame
        self.is_airplane: bool = False
        self._airplane_confirm_frames: int = 0

    def update(self, new_position: Tuple[int, int], current_frame: int):
        """オブジェクトの位置情報と最終検出フレームを更新する。"""
        self.positions.append(new_position)
        self.last_frame = current_frame

    def distance_to(self, position: Tuple[int, int]) -> float:
        """オブジェクトの最後の位置から指定位置までのユークリッド距離を計算する。"""
        if not self.positions:
            return float('inf')
        last_position = self.positions[-1]
        return np.sqrt((last_position[0] - position[0])**2 + (last_position[1] - position[1])**2)

    def classify(self, frame_rate: float):
        """オブジェクトを飛行機かどうか分類する。"""
        if not frame_rate or frame_rate <= 0:
            return

        if self.is_airplane:
            return

        duration_seconds = (self.last_frame - self.start_frame) / frame_rate

        if duration_seconds >= config.AIRPLANE_DURATION_THRESHOLD:
            self.is_airplane = True
            print(f"オブジェクト {id(self)} は持続時間 ({duration_seconds:.1f}秒) により飛行機と判定されました。")
            return

def update_tracked_objects(
    detections: List[Tuple[int, int]],
    current_frame: int,
    frame_rate: float
):
    """
    検出結果をもとに追跡オブジェクトリストを更新する。
    (新規追加、既存更新、古いオブジェクトの削除)
    """
    global tracked_objects
    distance_threshold = config.TRACKING_DISTANCE_THRESHOLD

    matched_indices = set()

    for obj in tracked_objects:
        best_match_dist = float('inf')
        best_match_idx = -1

        for i, det in enumerate(detections):
            if i in matched_indices:
                continue
            dist = obj.distance_to(det)
            if dist <= distance_threshold and dist < best_match_dist:
                best_match_dist = dist
                best_match_idx = i

        if best_match_idx != -1:
            obj.update(detections[best_match_idx], current_frame)
            matched_indices.add(best_match_idx)

    for i, det in enumerate(detections):
        if i not in matched_indices:
            tracked_objects.append(TrackedObject(det, current_frame))

    classify_objects(frame_rate)

    inactive_threshold_frames = frame_rate * 5
    objects_to_remove = [
        obj for obj in tracked_objects
        if (current_frame - obj.last_frame) > inactive_threshold_frames
    ]

    if objects_to_remove:
        print(f"{len(objects_to_remove)} 個の非アクティブなオブジェクトを削除します。")
        for obj in objects_to_remove:
            tracked_objects.remove(obj)

def classify_objects(frame_rate: float):
    """追跡中のすべてのオブジェクトに対して分類処理を実行する。"""
    global tracked_objects
    for obj in tracked_objects:
        obj.classify(frame_rate)

def clear_tracked_objects():
    """追跡中のオブジェクトリストをクリアする。"""
    global tracked_objects
    tracked_objects.clear()
    print("追跡オブジェクトリストをクリアしました。")

def get_current_tracked_objects() -> List['TrackedObject']:
    """現在追跡中のオブジェクトのリストを返す。"""
    global tracked_objects
    return tracked_objects.copy()

def add_detection_for_simple_airplane_check(position: Optional[Tuple[int, int]] = None):
    """簡易飛行機判定のために、検出の有無を記録する。"""
    global _past_detections_deque, _past_detections_lock
    with _past_detections_lock:
        if position is not None and position != (0, 0):
             _past_detections_deque.append((1, 1))
        else:
             _past_detections_deque.append((0, 0))

def is_airplane_simple() -> bool:
    """
    簡易的な飛行機判定。過去Nフレーム中、Mフレーム以上で検出があればTrue。
    """
    global _past_detections_deque, _past_detections_lock
    threshold = config.NON_ZERO_DETECTION_THRESHOLD_FOR_AIRPLANE
    with _past_detections_lock:
        non_zero_detections_count = sum(1 for coord in _past_detections_deque if coord != (0, 0))
        if non_zero_detections_count >= threshold:
            return True
        else:
            return False

# 以下のヘルパー関数は、輝度やフレーム情報を含む辞書形式の `detections` を想定
def check_brightness_consistency(
    detections: List[Dict[str, Any]],
    threshold: float = config.BRIGHTNESS_CONSISTENCY_THRESHOLD
) -> bool:
    """検出リスト内の輝度の一貫性を確認する。"""
    brightness_values = [d['brightness'] for d in detections if 'brightness' in d]
    if len(brightness_values) < 2:
        return True

    brightness_diff = np.max(brightness_values) - np.min(brightness_values)
    return brightness_diff <= threshold

def check_velocity_consistency(
    detections: List[Dict[str, Any]],
    threshold: float = config.VELOCITY_CONSISTENCY_THRESHOLD
) -> bool:
    """検出リスト内の速度の一貫性を確認する (標準偏差を使用)。"""
    velocities = []
    sorted_detections = sorted(detections, key=lambda d: d.get('frame', 0))

    for i in range(1, len(sorted_detections)):
        d_curr = sorted_detections[i]
        d_prev = sorted_detections[i - 1]

        if not all(k in d_curr for k in ['x', 'y', 'frame']) or \
           not all(k in d_prev for k in ['x', 'y', 'frame']):
            continue

        dx = d_curr['x'] - d_prev['x']
        dy = d_curr['y'] - d_prev['y']
        dt = d_curr['frame'] - d_prev['frame']

        if dt > 0:
            speed = np.sqrt(dx**2 + dy**2) / dt
            velocities.append(speed)

    if len(velocities) < 2:
        return True

    velocity_std = np.std(velocities)
    return velocity_std <= threshold


if __name__ == '__main__':
    print("tracking.py が直接実行されました。")

    print("\n--- TrackedObject テスト ---")
    obj1 = TrackedObject((100, 100), 0)
    obj1.update((110, 105), 1)
    obj1.update((120, 110), 2)
    print(f"オブジェクト1の最終位置: {obj1.positions[-1]}, 最終フレーム: {obj1.last_frame}")
    print(f"オブジェクト1の位置(130, 115)への距離: {obj1.distance_to((130, 115)):.2f}")
    obj1.last_frame = 15 * 8
    obj1.classify(frame_rate=15)
    print(f"オブジェクト1は飛行機か？: {obj1.is_airplane}")

    print("\n--- update_tracked_objects テスト ---")
    tracked_objects = [TrackedObject((50, 50), 100)]
    current_detections = [(55, 55), (200, 200)]
    update_tracked_objects(current_detections, 101, frame_rate=15)
    print(f"更新後の追跡オブジェクト数: {len(tracked_objects)}")
    print(f"オブジェクト0の最終位置: {tracked_objects[0].positions[-1]}")
    print(f"オブジェクト1の最終位置: {tracked_objects[1].positions[-1]}")

    print("\n--- is_airplane_simple テスト ---")
    _past_detections_deque.clear()
    add_detection_for_simple_airplane_check((1,1))
    add_detection_for_simple_airplane_check((0,0))
    add_detection_for_simple_airplane_check((1,1))
    add_detection_for_simple_airplane_check((1,1))
    add_detection_for_simple_airplane_check((1,1))
    add_detection_for_simple_airplane_check((0,0))
    add_detection_for_simple_airplane_check((1,1))
    print(f"簡易飛行機判定 (5検出): {is_airplane_simple()}")
    add_detection_for_simple_airplane_check((1,1))
    print(f"簡易飛行機判定 (6検出): {is_airplane_simple()}")

    print("\n--- check_consistency テスト ---")
    test_dets = [
        {'x': 10, 'y': 10, 'frame': 0, 'brightness': 100},
        {'x': 15, 'y': 12, 'frame': 1, 'brightness': 105},
        {'x': 20, 'y': 14, 'frame': 2, 'brightness': 110},
        {'x': 25, 'y': 16, 'frame': 3, 'brightness': 135},
        {'x': 40, 'y': 20, 'frame': 4, 'brightness': 140},
    ]
    print(f"輝度一貫性: {check_brightness_consistency(test_dets)}")
    print(f"速度一貫性: {check_velocity_consistency(test_dets)}")

    clear_tracked_objects()
