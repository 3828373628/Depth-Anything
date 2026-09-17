from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_ROOT / "models"
CACHE_DIR = PROJECT_ROOT / "cache"
TEMP_DIR = PROJECT_ROOT / "temp"
UPLOADS_DIR = PROJECT_ROOT / "uploads"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

MODEL_REPOS = {
    "Small": "depth-anything/Depth-Anything-V2-Small-hf",
    "Base": "depth-anything/Depth-Anything-V2-Base-hf",
}
MODEL_DIR_NAMES = {"Small": "small", "Base": "base"}

_MODEL_CACHE: dict[tuple[str, str], tuple[Any, Any]] = {}


def ensure_project_dirs() -> None:
    for path in (MODELS_DIR, CACHE_DIR, TEMP_DIR, UPLOADS_DIR, OUTPUTS_DIR):
        path.mkdir(parents=True, exist_ok=True)


def configure_project_caches() -> None:
    ensure_project_dirs()
    huggingface_cache = CACHE_DIR / "huggingface"
    torch_cache = CACHE_DIR / "torch"
    pip_cache = CACHE_DIR / "pip"
    inductor_cache = torch_cache / "inductor"
    gradio_temp = TEMP_DIR / "gradio"
    for path in (huggingface_cache, torch_cache, pip_cache, inductor_cache, gradio_temp):
        path.mkdir(parents=True, exist_ok=True)

    os.environ["HF_HOME"] = str(huggingface_cache)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TORCH_HOME"] = str(torch_cache)
    os.environ["PIP_CACHE_DIR"] = str(pip_cache)
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(inductor_cache)
    os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
    os.environ["GRADIO_TEMP_DIR"] = str(gradio_temp)
    os.environ["TEMP"] = str(TEMP_DIR)
    os.environ["TMP"] = str(TEMP_DIR)
    os.environ["TMPDIR"] = str(TEMP_DIR)
    for key in ("NO_PROXY", "no_proxy"):
        entries = [item.strip() for item in os.environ.get(key, "").split(",") if item.strip()]
        for host in ("127.0.0.1", "localhost"):
            if host not in entries:
                entries.append(host)
        os.environ[key] = ",".join(entries)
    tempfile.tempdir = str(TEMP_DIR)


def select_device_name(cuda_available: bool | None = None, mps_available: bool | None = None) -> str:
    if cuda_available is None or mps_available is None:
        import torch

        if cuda_available is None:
            cuda_available = torch.cuda.is_available()
        if mps_available is None:
            mps_available = bool(
                getattr(torch.backends, "mps", None)
                and torch.backends.mps.is_available()
            )

    if cuda_available:
        return "cuda"
    if mps_available:
        return "mps"
    return "cpu"


def _even(value: int) -> int:
    return max(2, value - (value % 2))


def calculate_output_size(width: int, height: int, resolution: str) -> tuple[int, int]:
    if width <= 0 or height <= 0:
        raise ValueError("视频尺寸无效。")

    if resolution == "Original":
        return _even(width), _even(height)

    targets = {"720p": 720, "1080p": 1080, "2160p": 2160}
    if resolution not in targets:
        raise ValueError(f"不支持的输出分辨率：{resolution}")

    short_side = targets[resolution]
    if width >= height:
        target_h = short_side
        target_w = round(width * target_h / height)
    else:
        target_w = short_side
        target_h = round(height * target_w / width)
    return _even(target_w), _even(target_h)


def normalize_depth(depth: np.ndarray, invert: bool = False) -> np.ndarray:
    array = np.asarray(depth, dtype=np.float32)
    minimum = float(np.nanmin(array))
    maximum = float(np.nanmax(array))
    span = maximum - minimum
    if not np.isfinite(span) or span <= 1e-8:
        normalized = np.zeros_like(array, dtype=np.uint8)
    else:
        normalized = np.clip((array - minimum) / span * 255.0, 0, 255).astype(np.uint8)
    if invert:
        normalized = 255 - normalized
    return normalized


def smooth_depth(
    current: np.ndarray,
    previous: np.ndarray | None,
    strength: float,
) -> np.ndarray:
    if not 0.0 <= float(strength) <= 0.95:
        raise ValueError("时间平滑强度必须在 0 到 0.95 之间。")
    current_float = np.asarray(current, dtype=np.float32)
    if previous is None or strength == 0:
        return current_float.copy()
    previous_float = np.asarray(previous, dtype=np.float32)
    return previous_float * float(strength) + current_float * (1.0 - float(strength))


def get_model_dir(model_size: str) -> Path:
    try:
        return MODELS_DIR / MODEL_DIR_NAMES[model_size]
    except KeyError as exc:
        raise ValueError(f"不支持的模型大小：{model_size}") from exc


def _make_output_path(source: Path, now: datetime | None = None) -> Path:
    timestamp = (now or datetime.now()).strftime("%Y%m%d_%H%M%S_%f")
    return OUTPUTS_DIR / f"{source.stem}_depth_{timestamp}.mp4"


def find_ffmpeg() -> Path:
    configured = os.environ.get("FFMPEG_PATH")
    if configured:
        candidate = Path(configured)
        if candidate.is_file():
            return candidate

    found = shutil.which("ffmpeg")
    if found:
        return Path(found)
    raise FileNotFoundError("未找到 FFmpeg。请先安装 FFmpeg 并加入 PATH，或设置 FFMPEG_PATH。")


def build_video_encode_command(
    ffmpeg_path: Path,
    width: int,
    height: int,
    fps: float,
    output_path: Path,
) -> list[str]:
    return [
        str(ffmpeg_path),
        "-y",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s:v",
        f"{width}x{height}",
        "-r",
        f"{fps:.6f}",
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output_path),
    ]


def _load_model(model_size: str, device_name: str) -> tuple[Any, Any]:
    cache_key = (model_size, device_name)
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]

    model_dir = get_model_dir(model_size)
    required_files = ("config.json", "preprocessor_config.json", "model.safetensors")
    missing_files = [name for name in required_files if not (model_dir / name).is_file()]
    if missing_files:
        raise FileNotFoundError(
            f"本地 {model_size} 模型缺少必要文件：{', '.join(missing_files)}。"
            f"请重新准备模型目录：{model_dir}"
        )

    import torch
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation

    processor = AutoImageProcessor.from_pretrained(model_dir, local_files_only=True)
    model = AutoModelForDepthEstimation.from_pretrained(model_dir, local_files_only=True)
    model.to(torch.device(device_name))
    model.eval()
    _MODEL_CACHE[cache_key] = (processor, model)
    return processor, model


def _infer_depth(frame_bgr: np.ndarray, model_size: str, device_name: str) -> np.ndarray:
    import cv2
    import torch
    from PIL import Image

    processor, model = _load_model(model_size, device_name)
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(rgb)
    inputs = processor(images=image, return_tensors="pt")
    inputs = {key: value.to(device_name) for key, value in inputs.items()}

    autocast_context = (
        torch.autocast(device_type="cuda", dtype=torch.float16)
        if device_name == "cuda"
        else nullcontext()
    )
    with torch.inference_mode(), autocast_context:
        outputs = model(**inputs)
        result = processor.post_process_depth_estimation(
            outputs,
            target_sizes=[(frame_bgr.shape[0], frame_bgr.shape[1])],
        )[0]["predicted_depth"]
    return result.detach().float().cpu().numpy()


def _mux_audio(ffmpeg_path: Path, video_path: Path, source_path: Path, output_path: Path) -> None:
    command = [
        str(ffmpeg_path),
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(video_path),
        "-i",
        str(source_path),
        "-map",
        "0:v:0",
        "-map",
        "1:a?",
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-shortest",
        "-movflags",
        "+faststart",
        str(output_path),
    ]
    result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if result.returncode != 0:
        raise RuntimeError(f"音频合并失败：{result.stderr.strip() or 'FFmpeg 未返回详细错误'}")


def convert_video(
    input_path: str,
    model_size: str,
    resolution: str,
    invert: bool,
    smoothing: float,
    preserve_audio: bool,
    progress: Any | None = None,
) -> tuple[str, str]:
    configure_project_caches()
    if not input_path:
        raise ValueError("请先上传 MP4 或 MOV 视频。")

    source = Path(input_path)
    if source.suffix.lower() not in {".mp4", ".mov"}:
        raise ValueError("仅支持 MP4 和 MOV 视频。")
    if not source.exists():
        raise FileNotFoundError(f"找不到输入视频：{source}")

    import cv2

    ffmpeg_path = find_ffmpeg()
    device_name = select_device_name()
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise RuntimeError("无法打开输入视频。")

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0 or not np.isfinite(fps):
        fps = 30.0
    output_w, output_h = calculate_output_size(width, height, resolution)

    job_dir = Path(tempfile.mkdtemp(prefix="depth_job_", dir=TEMP_DIR))
    silent_path = job_dir / "silent.mp4"
    final_path = _make_output_path(source)
    encoder: subprocess.Popen[bytes] | None = None

    try:
        command = build_video_encode_command(ffmpeg_path, output_w, output_h, fps, silent_path)
        encoder = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        if encoder.stdin is None:
            raise RuntimeError("无法打开 FFmpeg 视频输入管道。")

        previous: np.ndarray | None = None
        processed = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            depth = _infer_depth(frame, model_size, device_name)
            current = normalize_depth(depth, invert=False).astype(np.float32)
            smoothed = smooth_depth(current, previous, float(smoothing))
            previous = smoothed
            grayscale = np.clip(smoothed, 0, 255).astype(np.uint8)
            if invert:
                grayscale = 255 - grayscale
            if (grayscale.shape[1], grayscale.shape[0]) != (output_w, output_h):
                grayscale = cv2.resize(grayscale, (output_w, output_h), interpolation=cv2.INTER_LINEAR)
            bgr = cv2.cvtColor(grayscale, cv2.COLOR_GRAY2BGR)
            encoder.stdin.write(bgr.tobytes())

            processed += 1
            if progress is not None and total_frames > 0:
                progress(min(processed / total_frames, 1.0), desc=f"正在处理第 {processed}/{total_frames} 帧")

        encoder.stdin.close()
        stderr = encoder.stderr.read().decode("utf-8", errors="replace") if encoder.stderr else ""
        return_code = encoder.wait()
        encoder = None
        if return_code != 0:
            raise RuntimeError(f"FFmpeg 视频编码失败：{stderr.strip() or '未知错误'}")
        if processed == 0:
            raise RuntimeError("输入视频没有可读取的视频帧。")

        if preserve_audio:
            _mux_audio(ffmpeg_path, silent_path, source, final_path)
        else:
            shutil.move(str(silent_path), str(final_path))

        device_label = {"cuda": "NVIDIA CUDA", "mps": "Apple MPS", "cpu": "CPU"}[device_name]
        status = (
            f"完成：{processed} 帧 | {output_w}x{output_h} | {fps:.3f} FPS | "
            f"{model_size} | {device_label}"
        )
        return status, str(final_path)
    finally:
        capture.release()
        if encoder is not None:
            if encoder.stdin:
                try:
                    encoder.stdin.close()
                except OSError:
                    pass
            encoder.kill()
            encoder.wait()
        shutil.rmtree(job_dir, ignore_errors=True)


def build_ui() -> Any:
    configure_project_caches()
    import gradio as gr

    def run_conversion(
        video_path: str,
        model_size: str,
        resolution: str,
        invert: bool,
        smoothing: float,
        preserve_audio: bool,
        progress: gr.Progress = gr.Progress(track_tqdm=False),
    ) -> tuple[str, str | None]:
        try:
            return convert_video(
                video_path,
                model_size,
                resolution,
                invert,
                smoothing,
                preserve_audio,
                progress,
            )
        except Exception as exc:
            return f"失败：{exc}", None

    with gr.Blocks(title="Depth Anything V2 深度视频转换器") as demo:
        gr.Markdown("# Depth Anything V2 深度视频转换器\n本地逐帧生成灰度深度视频。")
        with gr.Row():
            with gr.Column():
                video_input = gr.File(label="上传 MP4 / MOV", file_types=[".mp4", ".mov"], type="filepath")
                model_size = gr.Radio(["Small", "Base"], value="Small", label="模型大小")
                resolution = gr.Dropdown(
                    ["Original", "720p", "1080p", "2160p"],
                    value="Original",
                    label="输出分辨率",
                )
                invert = gr.Checkbox(value=False, label="黑白反转")
                smoothing = gr.Slider(
                    minimum=0,
                    maximum=0.95,
                    value=0.6,
                    step=0.05,
                    label="时间平滑强度",
                    info="数值越大，帧间闪烁越少，但快速变化会更平滑。",
                )
                preserve_audio = gr.Checkbox(value=True, label="保留原始音频")
                convert_button = gr.Button("开始转换", variant="primary")
            with gr.Column():
                status = gr.Textbox(label="状态", interactive=False)
                output_video = gr.Video(label="输出 MP4", format="mp4")

        convert_button.click(
            run_conversion,
            inputs=[video_input, model_size, resolution, invert, smoothing, preserve_audio],
            outputs=[status, output_video],
        )
    return demo


if __name__ == "__main__":
    build_ui().queue(default_concurrency_limit=1).launch(inbrowser=True)
