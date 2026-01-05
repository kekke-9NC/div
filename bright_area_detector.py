"""
明るいエリア検出モジュール

Qwen3-VL を使用して画像内の明るいエリア（月、街灯など）を検出し、
マスク画像を生成する。

使用方法:
    import bright_area_detector
    mask = bright_area_detector.detect_bright_areas(image, progress_callback)
"""

import numpy as np
import cv2
import base64
import re
from typing import Optional, Tuple, Callable, List
from io import BytesIO

# LM Studio 設定
LM_STUDIO_URL = "http://localhost:1234/v1"
MODEL_ID = "qwen/qwen3-vl-4b"

# openai クライアントの遅延インポート
_client = None


def _get_client():
    """OpenAI クライアントを取得（遅延初期化）"""
    global _client
    if _client is None:
        try:
            from openai import OpenAI
            _client = OpenAI(base_url=LM_STUDIO_URL, api_key="lm-studio")
        except ImportError:
            raise ImportError("openai-python がインストールされていません。pip install openai を実行してください。")
    return _client


def check_vlm_connection() -> bool:
    """
    LM Studio (VLM) への接続を確認する
    
    Returns:
        bool: 接続成功なら True
    """
    try:
        client = _get_client()
        client.models.list()
        return True
    except Exception:
        return False


def image_to_base64_data_uri(image: np.ndarray, downscale: bool = False) -> str:
    """
    OpenCV画像をBase64 data URIに変換
    
    Args:
        image: BGR形式のOpenCV画像
        downscale: Trueの場合、画像サイズを1/2にリサイズ（高速化のため）
        
    Returns:
        str: data:image/jpeg;base64,... 形式の文字列
    """
    # BGRからRGBに変換
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    if downscale:
        # 画像サイズを1/2にリサイズ
        height, width = rgb_image.shape[:2]
        new_width = width // 2
        new_height = height // 2
        rgb_image = cv2.resize(rgb_image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    
    # JPEGエンコード
    _, buffer = cv2.imencode('.jpg', rgb_image, [cv2.IMWRITE_JPEG_QUALITY, 85])
    base64_data = base64.b64encode(buffer).decode('utf-8')
    return f"data:image/jpeg;base64,{base64_data}"


def _parse_boxes(text: str) -> List[Tuple[int, int, int, int]]:
    """
    VLMレスポンスからボックス座標をパース
    
    Args:
        text: VLMからのレスポンステキスト
        
    Returns:
        ボックスのリスト [(x1,y1,x2,y2), ...]（0-1000正規化座標）
    """
    boxes = []
    
    # "NONE" または空の場合
    if not text or "NONE" in text.upper():
        return boxes
    
    # パターン: (x1,y1,x2,y2) または (x1, y1, x2, y2)
    # 4つの数値を含む括弧を検出
    pattern = r"\((\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\)"
    matches = re.findall(pattern, text)
    
    for match in matches:
        try:
            x1, y1, x2, y2 = map(int, match)
            # 座標は0-1000の範囲内であること
            if all(0 <= v <= 1000 for v in [x1, y1, x2, y2]):
                boxes.append((x1, y1, x2, y2))
        except ValueError:
            continue
    
    return boxes


def create_mask_from_boxes(
    image_shape: Tuple[int, int],
    boxes: List[Tuple[int, int, int, int]],
    margin: int = 20
) -> np.ndarray:
    """
    検出されたボックスからマスク画像を生成
    
    Args:
        image_shape: (height, width)
        boxes: 検出されたボックスのリスト [(x1,y1,x2,y2), ...] 0-1000正規化座標
        margin: マスクの余白ピクセル
        
    Returns:
        マスク画像 (255=通常領域, 0=マスク領域)
    """
    height, width = image_shape
    
    # 全体を255（通常領域）で初期化
    mask = np.ones((height, width), dtype=np.uint8) * 255
    
    for (x1, y1, x2, y2) in boxes:
        # 0-1000から実際のピクセル座標に変換
        px1 = int(x1 * width / 1000) - margin
        py1 = int(y1 * height / 1000) - margin
        px2 = int(x2 * width / 1000) + margin
        py2 = int(y2 * height / 1000) + margin
        
        # 範囲を画像内に制限
        px1 = max(0, px1)
        py1 = max(0, py1)
        px2 = min(width, px2)
        py2 = min(height, py2)
        
        # マスク領域を0に設定
        mask[py1:py2, px1:px2] = 0
    
    return mask


def detect_bright_areas(
    image: np.ndarray,
    progress_callback: Optional[Callable[[str], None]] = None,
    margin: int = 40
) -> Optional[np.ndarray]:
    """
    画像から明るいエリアを検出してマスクを返す
    
    Args:
        image: 入力画像 (BGR)
        progress_callback: 進捗コールバック
        margin: マスクの余白ピクセル
        
    Returns:
        マスク画像 (255=通常領域, 0=マスク領域) or None（エラー時または検出なし）
    """
    def log(msg: str):
        if progress_callback:
            progress_callback(msg)
    
    # 接続確認
    if not check_vlm_connection():
        log("エラー: LM Studio への接続に失敗しました。localhost:1234 で起動していることを確認してください。")
        return None
    
    log("VLM接続OK。明るいエリアの検出を実行中...")
    
    try:
        client = _get_client()
        
        # 画像をBase64に変換（ソース解像度を使用）
        data_uri = image_to_base64_data_uri(image, downscale=False)
        
        # プロンプト設定
        system_prompt = "Identify STATIC bright light sources (moon, streetlights). Ignore meteors/satellites."
        
        user_prompt = (
            "Detect STATIC bright areas. Ignore meteors/trails. "
            "Return bounding boxes (0-1000 norm coords): (x1,y1,x2,y2). "
            "Output format: (x1,y1,x2,y2);(x1,y1,x2,y2) "
            "If none, reply: NONE"
        )
        
        # VLM呼び出し
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": [
                {"type": "text", "text": user_prompt},
                {"type": "image_url", "image_url": {"url": data_uri}}
            ]}
        ]
        
        # Qwen3-VL 公式推奨パラメータ (Instruct model)
        response = client.chat.completions.create(
            model=MODEL_ID,
            messages=messages,
            temperature=0.7,
            top_p=0.8,
            max_tokens=512
        )
        
        result_text = response.choices[0].message.content
        log(f"VLM応答: {result_text[:100]}..." if len(result_text) > 100 else f"VLM応答: {result_text}")
        
        # ボックスをパース
        boxes = _parse_boxes(result_text)
        
        if not boxes:
            log("明るいエリアは検出されませんでした。")
            return None
        
        log(f"{len(boxes)}個の明るいエリアを検出しました。")
        
        # マスク生成
        height, width = image.shape[:2]
        mask = create_mask_from_boxes((height, width), boxes, margin=margin)
        
        return mask
        
    except Exception as e:
        log(f"明るいエリア検出中にエラーが発生しました: {e}")
        return None


def detect_bright_areas_with_boxes(
    image: np.ndarray,
    progress_callback: Optional[Callable[[str], None]] = None,
    margin: int = 20
) -> Optional[Tuple[np.ndarray, List[Tuple[int, int, int, int]]]]:
    """
    画像から明るいエリアを検出してマスクとボックス座標を返す
    
    Args:
        image: 入力画像 (BGR)
        progress_callback: 進捗コールバック
        margin: マスクの余白ピクセル
        
    Returns:
        (マスク画像, ボックスリスト) or None（エラー時）
    """
    def log(msg: str):
        if progress_callback:
            progress_callback(msg)
    
    # 接続確認
    if not check_vlm_connection():
        log("エラー: LM Studio への接続に失敗しました。localhost:1234 で起動していることを確認してください。")
        return None
    
    log("VLM接続OK。明るいエリアの検出を実行中...")
    
    try:
        client = _get_client()
        
        # 画像をBase64に変換（ソース解像度を使用）
        data_uri = image_to_base64_data_uri(image, downscale=False)
        
        # プロンプト設定
        system_prompt = "Identify STATIC bright light sources (moon, planets, streetlights). Ignore meteors/satellites."
        
        user_prompt = (
            "Detect STATIC bright areas. Ignore meteors/trails. "
            "Return bounding boxes (0-1000 norm coords): (x1,y1,x2,y2). "
            "Output format: (x1,y1,x2,y2);(x1,y1,x2,y2) "
            "If none, reply: NONE"
        )
        
        # VLM呼び出し
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": [
                {"type": "text", "text": user_prompt},
                {"type": "image_url", "image_url": {"url": data_uri}}
            ]}
        ]
        
        # Qwen3-VL 公式推奨パラメータ (Instruct model)
        response = client.chat.completions.create(
            model=MODEL_ID,
            messages=messages,
            temperature=0.7,
            top_p=0.8,
            max_tokens=512
        )
        
        result_text = response.choices[0].message.content
        log(f"VLM応答: {result_text[:100]}..." if len(result_text) > 100 else f"VLM応答: {result_text}")
        
        # ボックスをパース
        boxes = _parse_boxes(result_text)
        
        if not boxes:
            log("明るいエリアは検出されませんでした。")
            return None
        
        log(f"{len(boxes)}個の明るいエリアを検出しました。")
        
        # マスク生成
        height, width = image.shape[:2]
        mask = create_mask_from_boxes((height, width), boxes, margin=margin)
        
        return (mask, boxes)
        
    except Exception as e:
        log(f"明るいエリア検出中にエラーが発生しました: {e}")
        return None


def get_representative_frame(file_path: str) -> Optional[np.ndarray]:
    """
    ファイルから代表フレームを取得
    
    Args:
        file_path: 画像または動画ファイルのパス
        
    Returns:
        BGR形式の画像 or None
    """
    import os
    from pathlib import Path
    
    if not os.path.isfile(file_path):
        return None
    
    ext = Path(file_path).suffix.lower()
    image_ext = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    video_ext = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.m4v'}
    
    if ext in image_ext:
        return cv2.imread(file_path)
    elif ext in video_ext:
        cap = cv2.VideoCapture(file_path)
        if cap.isOpened():
            ret, frame = cap.read()
            cap.release()
            if ret:
                return frame
    
    return None


def create_inclusion_mask_from_boxes(
    image_shape: Tuple[int, int],
    boxes: List[Tuple[int, int, int, int]],
    margin: int = 40
) -> np.ndarray:
    """
    検出されたボックスから包含マスク画像を生成（検出領域=白、それ以外=黒）
    
    Args:
        image_shape: (height, width)
        boxes: 検出されたボックスのリスト [(x1,y1,x2,y2), ...] 0-1000正規化座標
        margin: マスクの余白ピクセル
        
    Returns:
        マスク画像 (255=検出領域, 0=それ以外)
    """
    height, width = image_shape
    
    # 全体を0（除外）で初期化
    mask = np.zeros((height, width), dtype=np.uint8)
    
    for (x1, y1, x2, y2) in boxes:
        # 0-1000から実際のピクセル座標に変換
        px1 = int(x1 * width / 1000) - margin
        py1 = int(y1 * height / 1000) - margin
        px2 = int(x2 * width / 1000) + margin
        py2 = int(y2 * height / 1000) + margin
        
        # 範囲を画像内に制限
        px1 = max(0, px1)
        py1 = max(0, py1)
        px2 = min(width, px2)
        py2 = min(height, py2)
        
        # 検出領域を255に設定
        mask[py1:py2, px1:px2] = 255
    
    return mask


def detect_meteors_with_boxes(
    image: np.ndarray,
    progress_callback: Optional[Callable[[str], None]] = None,
    margin: int = 40
) -> Optional[Tuple[np.ndarray, List[Tuple[int, int, int, int]]]]:
    """
    画像から流星を検出して包含マスクとボックス座標を返す
    
    Args:
        image: 入力画像 (BGR)
        progress_callback: 進捗コールバック
        margin: マスクの余白ピクセル
        
    Returns:
        (マスク画像, ボックスリスト) or None（エラー時）
        マスクは検出領域=255、それ以外=0
    """
    def log(msg: str):
        if progress_callback:
            progress_callback(msg)
    
    # 接続確認
    if not check_vlm_connection():
        log("エラー: LM Studio への接続に失敗しました。localhost:1234 で起動していることを確認してください。")
        return None
    
    log("VLM接続OK。流星の検出を実行中...")
    
    try:
        client = _get_client()
        
        # 画像をBase64に変換（ソース解像度を使用）
        data_uri = image_to_base64_data_uri(image, downscale=False)
        
        # プロンプト設定（流星検出用）
        system_prompt = "Identify meteors, shooting stars, and linear light trails. Ignore static lights like moon/streetlights."
        
        user_prompt = (
            "Detect meteors/shooting stars (linear light streaks). Ignore moon/stars/streetlights. "
            "Return bounding boxes (0-1000 norm coords): (x1,y1,x2,y2). "
            "Output format: (x1,y1,x2,y2);(x1,y1,x2,y2) "
            "If none, reply: NONE"
        )
        
        # VLM呼び出し
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": [
                {"type": "text", "text": user_prompt},
                {"type": "image_url", "image_url": {"url": data_uri}}
            ]}
        ]
        
        # Qwen3-VL 公式推奨パラメータ (Instruct model)
        response = client.chat.completions.create(
            model=MODEL_ID,
            messages=messages,
            temperature=0.7,
            top_p=0.8,
            max_tokens=512
        )
        
        result_text = response.choices[0].message.content
        log(f"VLM応答: {result_text[:100]}..." if len(result_text) > 100 else f"VLM応答: {result_text}")
        
        # ボックスをパース
        boxes = _parse_boxes(result_text)
        
        if not boxes:
            log("流星は検出されませんでした。")
            return None
        
        log(f"{len(boxes)}個の流星を検出しました。")
        
        # 包含マスク生成（検出領域=255、それ以外=0）
        height, width = image.shape[:2]
        mask = create_inclusion_mask_from_boxes((height, width), boxes, margin=margin)
        
        return (mask, boxes)
        
    except Exception as e:
        log(f"流星検出中にエラーが発生しました: {e}")
        return None

