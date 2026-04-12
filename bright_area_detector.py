"""
明るいエリア検出モジュール

Qwen3-VL を使用して画像内の明るいエリア（月、街灯など）を検出し、
マスク画像を生成する。

使用方法:
    import bright_area_detector
    mask = bright_area_detector.detect_bright_areas(image, progress_callback)
"""

import os
import base64
import io
import json
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
import requests

try:
    import config as app_config
except Exception:
    app_config = None

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

AI_BACKEND_LOCAL = getattr(app_config, "AI_VLM_BACKEND_LOCAL_QWEN3_VL_4B", "local_qwen3_vl_4b")
AI_BACKEND_LM_STUDIO = getattr(app_config, "AI_VLM_BACKEND_LM_STUDIO_QWEN35_2B", "lmstudio_qwen3_5_2b")
DEFAULT_AI_BACKEND = getattr(app_config, "DEFAULT_AI_VLM_BACKEND", AI_BACKEND_LOCAL)
DEFAULT_LM_STUDIO_URL = getattr(app_config, "DEFAULT_LM_STUDIO_VLM_URL", "http://localhost:1234/v1")
DEFAULT_LM_STUDIO_MODEL_ID = getattr(app_config, "DEFAULT_LM_STUDIO_VLM_MODEL_ID", "qwen3.5-2b")
DEFAULT_LM_STUDIO_API_KEY = getattr(app_config, "DEFAULT_LM_STUDIO_VLM_API_KEY", "lm-studio")

_model = None
_processor = None
_model_loading = False
_last_used_time = None
_offload_timer = None
_offload_lock = threading.Lock()
_ai_backend = DEFAULT_AI_BACKEND
_lm_studio_url = DEFAULT_LM_STUDIO_URL
_lm_studio_model_id = DEFAULT_LM_STUDIO_MODEL_ID
_lm_studio_api_key = DEFAULT_LM_STUDIO_API_KEY
_lm_studio_connection_cache_key = None


def _log(msg: str, status_callback: Optional[Callable[[str], None]] = None) -> None:
    print(msg)
    if status_callback:
        try:
            status_callback(msg)
        except Exception:
            pass


def normalize_lm_studio_url(raw_url: str) -> str:
    value = (raw_url or DEFAULT_LM_STUDIO_URL).strip().rstrip("/")
    if not value:
        value = DEFAULT_LM_STUDIO_URL.rstrip("/")
    if value.endswith("/api/v1"):
        value = value[: -len("/api/v1")]
    if not value.endswith("/v1"):
        value = f"{value}/v1"
    return value


def configure_ai_backend(
    backend: Optional[str] = None,
    lm_studio_url: Optional[str] = None,
    lm_studio_model_id: Optional[str] = None,
    lm_studio_api_key: Optional[str] = None,
) -> None:
    global _ai_backend, _lm_studio_url, _lm_studio_model_id, _lm_studio_api_key, _lm_studio_connection_cache_key
    old_key = (_ai_backend, normalize_lm_studio_url(_lm_studio_url), _lm_studio_model_id, _lm_studio_api_key)

    selected_backend = (backend or DEFAULT_AI_BACKEND).strip()
    if selected_backend not in {AI_BACKEND_LOCAL, AI_BACKEND_LM_STUDIO}:
        selected_backend = DEFAULT_AI_BACKEND

    _ai_backend = selected_backend
    if lm_studio_url is not None:
        _lm_studio_url = normalize_lm_studio_url(lm_studio_url)
    else:
        _lm_studio_url = normalize_lm_studio_url(_lm_studio_url)
    if lm_studio_model_id is not None:
        _lm_studio_model_id = (lm_studio_model_id or DEFAULT_LM_STUDIO_MODEL_ID).strip() or DEFAULT_LM_STUDIO_MODEL_ID
    if lm_studio_api_key is not None:
        _lm_studio_api_key = (lm_studio_api_key or "").strip()

    new_key = (_ai_backend, normalize_lm_studio_url(_lm_studio_url), _lm_studio_model_id, _lm_studio_api_key)
    if new_key != old_key:
        _lm_studio_connection_cache_key = None


def get_ai_backend_config() -> dict:
    return {
        "backend": _ai_backend,
        "lm_studio_url": _lm_studio_url,
        "lm_studio_model_id": _lm_studio_model_id,
        "lm_studio_api_key": _lm_studio_api_key,
    }


def uses_local_model() -> bool:
    return _ai_backend == AI_BACKEND_LOCAL


def get_active_model_name() -> str:
    if _ai_backend == AI_BACKEND_LM_STUDIO:
        return f"LM Studio: {_lm_studio_model_id}"
    return MODEL_ID


def _lm_studio_headers() -> dict:
    headers = {"Accept": "application/json", "Content-Type": "application/json"}
    if _lm_studio_api_key:
        headers["Authorization"] = f"Bearer {_lm_studio_api_key}"
    return headers


def _lm_studio_request(method: str, path: str, payload: Optional[dict] = None, timeout: int = 120) -> dict:
    url = f"{normalize_lm_studio_url(_lm_studio_url)}{path}"
    try:
        response = requests.request(
            method,
            url,
            headers=_lm_studio_headers(),
            json=payload,
            timeout=timeout,
        )
    except requests.RequestException as exc:
        raise RuntimeError(f"LM Studioに接続できません: {normalize_lm_studio_url(_lm_studio_url)}\n{exc}") from exc

    if not response.ok:
        try:
            detail = response.json()
        except ValueError:
            detail = response.text.strip()
        raise RuntimeError(f"LM Studio APIエラー ({response.status_code}): {detail}")

    if not response.content:
        return {}
    try:
        return response.json()
    except ValueError as exc:
        raise RuntimeError("LM Studio APIの応答がJSONではありません。") from exc


def _lm_studio_list_model_ids() -> List[str]:
    payload = _lm_studio_request("GET", "/models", timeout=20)
    items = payload.get("data") or payload.get("models") or []
    model_ids = []
    for item in items:
        if not isinstance(item, dict):
            continue
        model_id = item.get("id") or item.get("key") or item.get("model")
        if model_id:
            model_ids.append(str(model_id))
    return model_ids


def _image_to_data_uri(image: np.ndarray) -> str:
    pil_image = _cv2_to_pil(image)
    buffer = io.BytesIO()
    pil_image.save(buffer, format="JPEG", quality=85)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def _call_lm_studio_chat(
    messages: List[dict],
    status_callback: Optional[Callable[[str], None]] = None,
    stream_callback: Optional[Callable[[str], None]] = None,
    max_tokens: int = 512,
) -> str:
    if status_callback:
        status_callback(f"LM Studioへ送信中: {_lm_studio_model_id}")

    payload = {
        "model": _lm_studio_model_id,
        "messages": messages,
        "temperature": 0.1,
        "top_p": 0.9,
        "max_tokens": max_tokens,
        "stream": bool(stream_callback),
    }

    if stream_callback:
        url = f"{normalize_lm_studio_url(_lm_studio_url)}/chat/completions"
        try:
            response = requests.post(
                url,
                headers=_lm_studio_headers(),
                json=payload,
                timeout=300,
                stream=True,
            )
        except requests.RequestException as exc:
            raise RuntimeError(f"LM Studioに接続できません: {normalize_lm_studio_url(_lm_studio_url)}\n{exc}") from exc

        if not response.ok:
            try:
                detail = response.json()
            except ValueError:
                detail = response.text.strip()
            raise RuntimeError(f"LM Studio APIエラー ({response.status_code}): {detail}")

        output_text = ""
        for raw_line in response.iter_lines(decode_unicode=True):
            if not raw_line:
                continue
            line = raw_line.strip()
            if line.startswith("data:"):
                line = line[len("data:"):].strip()
            if not line or line == "[DONE]":
                continue
            try:
                chunk = json.loads(line)
            except ValueError:
                continue

            choices = chunk.get("choices") or []
            if not choices:
                continue
            delta = choices[0].get("delta") or {}
            content = delta.get("content")
            if isinstance(content, list):
                text = "".join(str(item.get("text", "")) for item in content if isinstance(item, dict))
            else:
                text = str(content or "")
            if text:
                output_text += text
                stream_callback(text)
        return output_text.strip()

    data = _lm_studio_request("POST", "/chat/completions", payload=payload, timeout=300)
    choices = data.get("choices") or []
    if not choices:
        raise RuntimeError("LM Studioから有効な応答が返りませんでした。")
    content = choices[0].get("message", {}).get("content", "")
    if isinstance(content, list):
        text_parts = []
        for item in content:
            if isinstance(item, dict) and item.get("text"):
                text_parts.append(str(item["text"]))
        output_text = "\n".join(text_parts).strip()
    else:
        output_text = str(content).strip()
    return output_text


STRICT_BOX_SYSTEM_PROMPT = (
    "You are a strict visual bounding-box detector. "
    "Return only the requested coordinates in the exact text format. "
    "Do not explain, do not use JSON, and do not use Markdown."
)


def _strict_box_user_prompt(target: str, ignore: str) -> str:
    return (
        f"Find only: {target}\n"
        f"Ignore: {ignore}\n"
        "Coordinates must be normalized integers from 0 to 1000.\n"
        "Output exactly one of these two forms:\n"
        "NONE\n"
        "(x1,y1,x2,y2);(x1,y1,x2,y2)\n"
        "Rules:\n"
        "- Use digits only inside each coordinate tuple.\n"
        "- Never write coordinate names such as x1, y1, xmin, ymin, bbox, bbox_2d.\n"
        "- Never prefix numbers with letters, for example do not write x648 or y239.\n"
        "- Never output JSON, arrays, labels, comments, prose, bullets, or code fences.\n"
        "- If there is no clear target, output exactly NONE.\n"
        "Valid example with one box: (648,239,705,263)\n"
        "Valid example with two boxes: (648,239,705,263);(120,300,180,360)\n"
        "Invalid examples: (x648,239),(705,263), (x1,y1,x2,y2), "
        "{\"bbox_2d\":[648,239,705,263]}\n"
        "Now inspect the image and output only the final answer."
    )


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


def _load_processor_compat(
    auto_processor_cls,
    model_dir: str,
    status_callback: Optional[Callable[[str], None]] = None,
):
    """AutoProcessorのAPI差分を吸収しつつ読み込む。"""
    try:
        return auto_processor_cls.from_pretrained(model_dir, trust_remote_code=True)
    except TypeError as e:
        msg = str(e)
        # transformers/tokenizersの組み合わせによりキーワード衝突が起きる環境向けフォールバック
        if "fix_mistral_regex" in msg or "trust_remote_code" in msg:
            _log("AutoProcessor互換フォールバックを適用します。", status_callback)
            return auto_processor_cls.from_pretrained(model_dir)
        raise


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
        if not torch.cuda.is_available():
            raise RuntimeError("ローカルLLM機能はNVIDIA GPU (CUDA) 環境でのみ利用できます。")

        from transformers import Qwen3VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
        from huggingface_hub import snapshot_download
        try:
            import transformers
            import tokenizers
            _log(
                f"Transformers={transformers.__version__}, tokenizers={tokenizers.__version__}",
                status_callback,
            )
        except Exception:
            pass
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
                _processor = _load_processor_compat(
                    AutoProcessor, LOCAL_MODEL_DIR, status_callback
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
        quantized_processor = _load_processor_compat(
            AutoProcessor, FULL_MODEL_DIR, status_callback
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
        _processor = _load_processor_compat(
            AutoProcessor, LOCAL_MODEL_DIR, status_callback
        )
        _log("量子化モデルのロード完了", status_callback)
        _reset_offload_timer()
        return _model, _processor
    finally:
        _model_loading = False



def check_vlm_connection(
    status_callback: Optional[Callable[[str], None]] = None,
    force: bool = False,
) -> tuple:
    """
    モデルの利用可否を確認する
    
    Returns:
        tuple: (利用可能なら True, モデル名または空文字列)
    """
    global _model, _processor, _lm_studio_connection_cache_key
    try:
        if _ai_backend == AI_BACKEND_LM_STUDIO:
            cache_key = (normalize_lm_studio_url(_lm_studio_url), _lm_studio_model_id, _lm_studio_api_key)
            if not force and _lm_studio_connection_cache_key == cache_key:
                return (True, get_active_model_name())

            model_ids = _lm_studio_list_model_ids()
            if model_ids and _lm_studio_model_id not in model_ids:
                return (
                    False,
                    f"LM Studioで指定モデルが見つかりません: {_lm_studio_model_id}\n"
                    f"利用可能: {', '.join(model_ids)}",
                )
            _lm_studio_connection_cache_key = cache_key
            return (True, get_active_model_name())

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
    if _ai_backend == AI_BACKEND_LM_STUDIO:
        data_uri = _image_to_data_uri(image)
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {"type": "image_url", "image_url": {"url": data_uri}},
                ],
            },
        ]
        return _call_lm_studio_chat(messages, status_callback, stream_callback, max_tokens=128)

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
    if _ai_backend == AI_BACKEND_LM_STUDIO:
        if status_callback:
            status_callback("LM Studio接続確認中...")
        connected, err = check_vlm_connection(status_callback)
        if not connected:
            return f"エラー: LM Studioモデルの確認に失敗しました: {err}"

        messages = [{"role": "system", "content": system_prompt}]
        if history:
            messages.extend(history)
        if image is not None:
            user_content = [
                {"type": "text", "text": user_prompt},
                {"type": "image_url", "image_url": {"url": _image_to_data_uri(image)}},
            ]
        else:
            user_content = user_prompt
        messages.append({"role": "user", "content": user_content})
        return _call_lm_studio_chat(messages, status_callback, stream_callback)

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
        system_prompt = STRICT_BOX_SYSTEM_PROMPT
        user_prompt = _strict_box_user_prompt(
            "static bright light sources such as the moon, planets, streetlights, lamps, or fixed glare",
            "meteors, shooting stars, satellites, aircraft trails, stars, clouds, noise, and moving light streaks",
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
        system_prompt = STRICT_BOX_SYSTEM_PROMPT
        user_prompt = _strict_box_user_prompt(
            "static bright light sources such as the moon, planets, streetlights, lamps, or fixed glare",
            "meteors, shooting stars, satellites, aircraft trails, stars, clouds, noise, and moving light streaks",
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
        system_prompt = STRICT_BOX_SYSTEM_PROMPT
        user_prompt = _strict_box_user_prompt(
            "meteors, shooting stars, or linear meteor-like light streaks",
            "moon, stars, planets, streetlights, clouds, static bright areas, aircraft, satellites, and noise",
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

