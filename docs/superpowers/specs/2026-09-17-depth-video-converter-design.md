# Depth Video Converter Design

## Goal

Build a self-contained local Gradio application for Windows and macOS that converts MP4/MOV video frames into grayscale Depth Anything V2 depth video, with automatic CUDA/MPS/CPU selection, selectable model size, output resolution, optional inversion, temporal smoothing, optional source audio preservation, and H.264 MP4 output.

## Constraints

- Keep the project portable: runtime paths must be derived from the project directory rather than a fixed drive letter or absolute path.
- Keep the Python virtual environment, model snapshots, caches, temporary files, uploads, and outputs under the project root where practical.
- Use Hugging Face Transformers' native Depth Anything V2 support; do not copy source code from an existing Depth Anything project.
- Prepare Small and Base model snapshots locally. Large remains an optional later download.
- Reuse the existing FFmpeg available on PATH; do not install CUDA Toolkit or cuDNN separately.
- Do not dump individual video frames to disk. Stream processed frames directly to FFmpeg.

## Architecture

`app.py` contains the complete application. OpenCV reads input video frames. Transformers preprocesses each RGB frame and runs Depth Anything V2 on the selected PyTorch device. The resulting relative depth is normalized to 8-bit grayscale, optionally temporally smoothed and inverted, resized to the selected output resolution, then streamed as raw BGR frames into FFmpeg for H.264 encoding. If requested, a second FFmpeg mux step copies the encoded video and adds the source audio as AAC when audio is present.

Model snapshots are stored under `models/small` and `models/base`. Runtime model objects are cached in memory by `(model_size, device)` to avoid reloading between conversions.

## Device Selection

Priority is CUDA, then Apple MPS, then CPU. CUDA inference uses automatic mixed precision when available. MPS and CPU use float32.

## User Interface

The Gradio interface exposes: MP4/MOV upload; model size (Small/Base); output resolution (Original/720p/1080p/2160p); invert black/white; temporal smoothing strength from 0 to 0.95; preserve original audio; Convert button; status text; MP4 result.

## Error Handling

Validate the uploaded extension, video readability, frame dimensions/FPS, local model presence, and FFmpeg availability. Surface concise actionable exceptions in the Gradio status rather than silently failing. Always terminate or close FFmpeg pipes and remove per-job temporary directories.

## Verification

Unit tests cover output dimension calculation, depth normalization/inversion, temporal smoothing, and device-priority logic. Environment verification confirms imports, CUDA status, FFmpeg, local Small/Base model snapshots, a one-image model inference, and a synthetic short-video conversion.
