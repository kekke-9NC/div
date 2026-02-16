"""
明るいエリア検出モジュール

Qwen3-VL を使用して画像内の明るいエリア（月、街灯など）を検出し、
マスク画像を生成する。

使用方法:
    import bright_area_detector
    mask = bright_area_detector.detect_bright_areas(image, progress_callback)
"""

import os
import numpy as np
import cv2
import re
import gc
import time
import threading
from pathlib import Path
from typing import Optional, Tuple, Callable, List
from PIL import Image

import torch
import shutil

Qwen3VLForConditionalGeneration = None
AutoProcessor = None
BitsAndBytesConfig = None
TextIteratorStreamer = None
process_vision_info = None

MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FULL_MODEL_DIR = os.path.join(BASE_DIR, "full_model")
LOCAL_MODEL_DIR = os.path.join(BASE_DIR, "quantized_model")
HF_CACHE_DIR = os.path.join(BASE_DIR, "hf_cache")
MODEL_OFFLOAD_TIMEOUT = 300

_model = None
_processor = None
_model_loading = False
_last_used_time = None
_offload_timer = None
_offload_lock = threading.Lock()


def _log(msg: str, status_callback: Optional[Callable[[str], None]] = None) -> None:
    print(msg)
    if status_callback:
        try:
            status_callback(msg)
        except Exception:
            pass


def _has_model_files(model_dir: str) -> bool:
    if not os.path.isdir(model_dir):
        return False

    config_path = os.path.join(model_dir, "config.json")
    if not os.path.isfile(config_path):
        return False

    p = Path(model_dir)
    weight_patterns = [
        "model*.safetensors",
        "model*.bin",
        "pytorch_model*.bin",
        "*.gguf",
    ]
    for pattern in weight_patterns:
        if any(p.glob(pattern)):
            return True
    return False


def has_quantized_model() -> bool:
    """有効な量子化モデルがローカルに存在するかを返す。"""
    return _has_model_files(LOCAL_MODEL_DIR)


def _offload_model():
    """モデルをオフロードしてGPUメモリを解放"""
    global _model, _processor, _last_used_time, _offload_timer
    
    with _offload_lock:
        if _model is None:
            return
        
        print("モデルを5分間未使用のためオフロードしています...")
        
        try:
            del _model
            del _processor
            _model = None
            _processor = None
            _last_used_time = None
            _offload_timer = None
            
            torch.cuda.empty_cache()
            gc.collect()
            
            print("モデルのオフロード完了（GPUメモリを解放しました）")
        except Exception as e:
            print(f"モデルオフロード中にエラー: {e}")


def _reset_offload_timer():
    """オフロードタイマーをリセット"""
    global _last_used_time, _offload_timer
    
    with _offload_lock:
        _last_used_time = time.time()
        if _offload_timer is not None:
            _offload_timer.cancel()
        _offload_timer = threading.Timer(MODEL_OFFLOAD_TIMEOUT, _offload_model)
        _offload_timer.daemon = True
        _offload_timer.start()


def _get_model(status_callback: Optional[Callable[[str], None]] = None):
    """モデルとプロセッサを取得（遅延初期化）"""
    global _model, _processor, _model_loading
    
    with _offload_lock:
        if _model is not None and _processor is not None:
            return _model, _processor
    
    if _model_loading:
        raise RuntimeError("モデルは現在ロード中です。しばらくお待ちください。")
    
    _model_loading = True
    try:
        from transformers import Qwen3VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
        from huggingface_hub import snapshot_download
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )

        _log("モデルをロード中...", status_callback)

        if _has_model_files(LOCAL_MODEL_DIR):
            try:
                _log(f"ローカル量子化モデルを読み込み中: {LOCAL_MODEL_DIR}", status_callback)
                _model = Qwen3VLForConditionalGeneration.from_pretrained(
                    LOCAL_MODEL_DIR,
                    device_map="auto",
                    trust_remote_code=True,
                    low_cpu_mem_usage=True
                )
                _processor = AutoProcessor.from_pretrained(
                    LOCAL_MODEL_DIR, trust_remote_code=True, fix_mistral_regex=True
                )
                _log("ローカル量子化モデルの読み込み完了", status_callback)
                _reset_offload_timer()
                return _model, _processor
            except Exception as e:
                _log(f"ローカル量子化モデル読み込み失敗。再構築します: {e}", status_callback)
                try:
                    shutil.rmtree(LOCAL_MODEL_DIR, ignore_errors=True)
                except Exception:
                    pass

        os.makedirs(HF_CACHE_DIR, exist_ok=True)
        os.makedirs(FULL_MODEL_DIR, exist_ok=True)

        _log(f"フルモデルをダウンロード中: {MODEL_ID}", status_callback)
        snapshot_download(
            repo_id=MODEL_ID,
            cache_dir=HF_CACHE_DIR,
            local_dir=FULL_MODEL_DIR,
            local_files_only=False,
            force_download=False,
        )
        _log(f"フルモデルのダウンロード完了: {FULL_MODEL_DIR}", status_callback)

        _log("フルモデルを量子化して読み込み中...", status_callback)
        quantized_model = Qwen3VLForConditionalGeneration.from_pretrained(
            FULL_MODEL_DIR,
            quantization_config=bnb_config,
            device_map="auto",
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )
        quantized_processor = AutoProcessor.from_pretrained(
            FULL_MODEL_DIR, trust_remote_code=True
        )

        tmp_quant_dir = f"{LOCAL_MODEL_DIR}_tmp"
        shutil.rmtree(tmp_quant_dir, ignore_errors=True)
        os.makedirs(tmp_quant_dir, exist_ok=True)

        _log(f"量子化モデルを保存中: {LOCAL_MODEL_DIR}", status_callback)
        quantized_model.save_pretrained(tmp_quant_dir)
        quantized_processor.save_pretrained(tmp_quant_dir)
        shutil.rmtree(LOCAL_MODEL_DIR, ignore_errors=True)
        os.replace(tmp_quant_dir, LOCAL_MODEL_DIR)
        _log("量子化モデル保存完了", status_callback)

        try:
            del quantized_model
            del quantized_processor
            torch.cuda.empty_cache()
            gc.collect()
        except Exception:
            pass

        _log(f"フルモデルを削除中: {FULL_MODEL_DIR}", status_callback)
        shutil.rmtree(FULL_MODEL_DIR, ignore_errors=True)
        _log("フルモデル削除完了", status_callback)

        _log("量子化モデルのみを再ロード中...", status_callback)
        _model = Qwen3VLForConditionalGeneration.from_pretrained(
            LOCAL_MODEL_DIR,
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        _processor = AutoProcessor.from_pretrained(
            LOCAL_MODEL_DIR, trust_remote_code=True, fix_mistral_regex=True
        )
        _log("量子化モデルのロード完了", status_callback)
        _reset_offload_timer()
        return _model, _processor
    finally:
        _model_loading = False



def check_vlm_connection(status_callback: Optional[Callable[[str], None]] = None) -> tuple:
    """
    モデルの利用可否を確認する
    
    Returns:
        tuple: (利用可能なら True, モデル名または空文字列)
    """
    global _model, _processor
    try:
        if _model is not None and _processor is not None:
            _reset_offload_timer()
            return (True, MODEL_ID)
        _get_model(status_callback)
        return (True, MODEL_ID)
    except Exception as e:
        return (False, str(e))



def _cv2_to_pil(image: np.ndarray) -> Image.Image:
    """OpenCV画像をPIL Imageに変換（リサイズ含む）"""
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb_image)
    if pil_image.height != 1080:
        aspect_ratio = pil_image.width / pil_image.height
        new_height = 1080
        new_width = int(new_height * aspect_ratio)
        pil_image = pil_image.resize((new_width, new_height))
    
    return pil_image


def _call_vlm(
    system_prompt: str, 
    user_prompt: str, 
    image: np.ndarray, 
    status_callback: Optional[Callable[[str], None]] = None,
    stream_callback: Optional[Callable[[str], None]] = None
) -> str:
    """VLMを呼び出してテキスト応答を取得"""
    from transformers import TextIteratorStreamer
    from qwen_vl_utils import process_vision_info
    model, processor = _get_model(status_callback)
    
    pil_image = _cv2_to_pil(image)
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": [
            {"type": "image", "image": pil_image},
            {"type": "text", "text": user_prompt}
        ]}
    ]
    
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to("cuda")
    
    if stream_callback:
        streamer = TextIteratorStreamer(processor, skip_prompt=True, skip_special_tokens=True)
        generation_kwargs = dict(
            **inputs,
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.8,
            top_k=20,
            do_sample=True,
            streamer=streamer,
        )
        
        thread = threading.Thread(target=model.generate, kwargs=generation_kwargs)
        thread.start()
        
        output_text = ""
        for new_text in streamer:
            output_text += new_text
            stream_callback(new_text)
            
        thread.join()
        
    else:
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.7,
                top_p=0.8,
                top_k=20,
                do_sample=True,
            )
        
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        del generated_ids

    del inputs, image_inputs, video_inputs
    torch.cuda.empty_cache()
    gc.collect()

    _reset_offload_timer()
    
    return output_text


def generate_response(
    user_prompt: str,
    system_prompt: str = "You are a helpful assistant.",
    image: Optional[np.ndarray] = None,
    history: Optional[List[dict]] = None,
    status_callback: Optional[Callable[[str], None]] = None,
    stream_callback: Optional[Callable[[str], None]] = None
) -> str:
    """
    汎用的なチャット応答を生成する
    
    Args:
        user_prompt: ユーザーからのメッセージ
        system_prompt: システムプロンプト
        image: オプションの画像 (BGR)
        history: 会話履歴 [{'role': 'user'|'assistant', 'content': str}]
        
    Returns:
        応答テキスト
    """
    from transformers import TextIteratorStreamer
    from qwen_vl_utils import process_vision_info
    if status_callback: status_callback("接続確認中...")
    connected, err = check_vlm_connection(status_callback)
    if not connected:
        return f"エラー: モデルの読み込みに失敗しました: {err}"

    model, processor = _get_model(status_callback)
    
    if status_callback: status_callback("回答を生成中...")
    content = []
    
    image_inputs = None
    video_inputs = None
    
    if image is not None:
        pil_image = _cv2_to_pil(image)
        content.append({"type": "image", "image": pil_image})
        
    content.append({"type": "text", "text": user_prompt})

    messages = [{"role": "system", "content": system_prompt}]

    if history:
        for msg in history:
            messages.append(msg)
            
    messages.append({"role": "user", "content": content})
    
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    if image is not None:
        image_inputs, video_inputs = process_vision_info(messages)
    
    inputs_args = {"text": [text], "padding": True, "return_tensors": "pt"}
    if image_inputs is not None:
        inputs_args["images"] = image_inputs
    if video_inputs is not None:
        inputs_args["videos"] = video_inputs
        
    inputs = processor(**inputs_args).to("cuda")

    if image is not None:
        pass
    
    if stream_callback:
        streamer = TextIteratorStreamer(processor, skip_prompt=True, skip_special_tokens=True)
        inputs_args["streamer"] = streamer
        
        generation_kwargs = dict(
            **inputs,
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.8,
            top_k=20,
            do_sample=True,
            streamer=streamer,
        )
        
        thread = threading.Thread(target=model.generate, kwargs=generation_kwargs)
        thread.start()
        
        output_text = ""
        for new_text in streamer:
            output_text += new_text
            stream_callback(new_text)
            
        thread.join()
        
    else:

        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.7,
                top_p=0.8,
                top_k=20,
                do_sample=True,
            )
        
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

    del inputs
    if "generated_ids" in locals(): del generated_ids
    if image_inputs is not None:
        del image_inputs
    if video_inputs is not None:
        del video_inputs
    torch.cuda.empty_cache()
    gc.collect()
    
    _reset_offload_timer()
    
    return output_text


def _parse_boxes(text: str) -> List[Tuple[int, int, int, int]]:
    """
    VLMレスポンスからボックス座標をパース
    
    Args:
        text: VLMからのレスポンステキスト
        
    Returns:
        ボックスのリスト [(x1,y1,x2,y2), ...]（0-1000正規化座標）
    
    対応形式:
        - (x1,y1,x2,y2) - 4つの数値
        - (x1,y1),(x2,y2) - 2つの座標ペア
        - (x1,y1),(x2,y2);(x1,y1),(x2,y2) - セミコロン区切りで複数
    """
    boxes = []

    if not text or "NONE" in text.upper():
        return boxes

    pattern_4nums = r"\((\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\)"
    matches_4nums = re.findall(pattern_4nums, text)
    
    for match in matches_4nums:
        try:
            x1, y1, x2, y2 = map(int, match)
            if all(0 <= v <= 1000 for v in [x1, y1, x2, y2]):
                boxes.append((x1, y1, x2, y2))
        except ValueError:
            continue

    if boxes:
        return boxes

    segments = text.split(';')
    pattern_pair = r"\((\d+)\s*,\s*(\d+)\)\s*,\s*\((\d+)\s*,\s*(\d+)\)"
    
    for segment in segments:
        match = re.search(pattern_pair, segment.strip())
        if match:
            try:
                x1, y1, x2, y2 = map(int, match.groups())
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

    mask = np.ones((height, width), dtype=np.uint8) * 255
    
    for (x1, y1, x2, y2) in boxes:
        px1 = int(x1 * width / 1000) - margin
        py1 = int(y1 * height / 1000) - margin
        px2 = int(x2 * width / 1000) + margin
        py2 = int(y2 * height / 1000) + margin

        px1 = max(0, px1)
        py1 = max(0, py1)
        px2 = min(width, px2)
        py2 = min(height, py2)

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

    connected, err = check_vlm_connection()
    if not connected:
        log(f"エラー: モデルの読み込みに失敗しました: {err}")
        return None
    
    log("モデル準備OK。明るいエリアの検出を実行中...")
    
    try:
        system_prompt = "Identify STATIC bright light sources (moon, streetlights). Ignore meteors/satellites."
        
        user_prompt = (
            "Detect STATIC bright areas. Ignore meteors/trails. "
            "Return bounding boxes (0-1000 norm coords): (x1,y1,x2,y2). "
            "Output format: (x1,y1,x2,y2);(x1,y1,x2,y2) "
            "If none, reply: NONE"
        )

        result_text = _call_vlm(system_prompt, user_prompt, image)
        log(f"VLM応答: {result_text[:100]}..." if len(result_text) > 100 else f"VLM応答: {result_text}")

        boxes = _parse_boxes(result_text)
        
        if not boxes:
            log("明るいエリアは検出されませんでした。")
            return None
        
        log(f"{len(boxes)}個の明るいエリアを検出しました。")

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
        print(f"[detect_bright_areas_with_boxes] {msg}")
        if progress_callback:
            progress_callback(msg)

    log("モデル接続を確認中...")
    connected, err = check_vlm_connection()
    if not connected:
        log(f"エラー: モデルの読み込みに失敗しました: {err}")
        return None
    
    log("モデル準備OK。明るいエリアの検出を実行中...")
    
    try:
        system_prompt = "Identify STATIC bright light sources (moon, planets, streetlights). Ignore meteors/satellites."
        
        user_prompt = (
            "Detect STATIC bright areas. Ignore meteors/trails. "
            "Return bounding boxes (0-1000 norm coords): (x1,y1,x2,y2). "
            "Output format: (x1,y1,x2,y2);(x1,y1,x2,y2) "
            "If none, reply: NONE"
        )

        result_text = _call_vlm(system_prompt, user_prompt, image)
        log(f"VLM応答: {result_text[:100]}..." if len(result_text) > 100 else f"VLM応答: {result_text}")

        boxes = _parse_boxes(result_text)
        
        if not boxes:
            log("明るいエリアは検出されませんでした。")
            return None
        
        log(f"{len(boxes)}個の明るいエリアを検出しました。")

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

    mask = np.zeros((height, width), dtype=np.uint8)
    
    for (x1, y1, x2, y2) in boxes:
        px1 = int(x1 * width / 1000) - margin
        py1 = int(y1 * height / 1000) - margin
        px2 = int(x2 * width / 1000) + margin
        py2 = int(y2 * height / 1000) + margin

        px1 = max(0, px1)
        py1 = max(0, py1)
        px2 = min(width, px2)
        py2 = min(height, py2)

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
        print(f"[detect_meteors_with_boxes] {msg}")
        if progress_callback:
            progress_callback(msg)

    log("モデル接続を確認中...")
    connected, err = check_vlm_connection()
    if not connected:
        log(f"エラー: モデルの読み込みに失敗しました: {err}")
        return None
    
    log("モデル準備OK。流星の検出を実行中...")
    
    try:
        system_prompt = "Identify meteors, shooting stars, and linear light trails. Ignore static lights like moon/streetlights."
        
        user_prompt = (
            "Detect meteors/shooting stars (linear light streaks). Ignore moon/stars/streetlights. "
            "Return bounding boxes (0-1000 norm coords): (x1,y1,x2,y2). "
            "Output format: (x1,y1,x2,y2);(x1,y1,x2,y2) "
            "If none, reply: NONE"
        )

        result_text = _call_vlm(system_prompt, user_prompt, image)
        log(f"VLM応答: {result_text[:100]}..." if len(result_text) > 100 else f"VLM応答: {result_text}")

        boxes = _parse_boxes(result_text)
        
        if not boxes:
            log("流星は検出されませんでした。")
            return None
        
        log(f"{len(boxes)}個の流星を検出しました。")

        height, width = image.shape[:2]
        mask = create_inclusion_mask_from_boxes((height, width), boxes, margin=margin)
        
        return (mask, boxes)
        
    except Exception as e:
        log(f"流星検出中にエラーが発生しました: {e}")
        return None

