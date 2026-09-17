# Depth Video Converter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and verify a local cross-platform Depth Anything V2 video-to-depth converter with Gradio UI and H.264 MP4 output.

**Architecture:** OpenCV decodes frames, Transformers runs local Depth Anything V2 snapshots on CUDA/MPS/CPU, NumPy performs temporal smoothing/inversion, and FFmpeg receives raw frames for H.264 encoding and optional audio muxing. All project data and caches are kept under the project root.

**Tech Stack:** Python 3.13, PyTorch, torchvision, Transformers, OpenCV, NumPy, Pillow, Gradio, Hugging Face Hub, FFmpeg.

**Spec:** `docs/superpowers/specs/2026-09-17-depth-video-converter-design.md`

## Global Constraints

- The application must not depend on a fixed drive letter or absolute project path.
- Small and Base are prepared locally; Large is not downloaded now.
- Existing FFmpeg is reused.
- CUDA Toolkit is not installed separately.
- No image-sequence frame dump is used.

---

### Task 1: Pure video-processing helpers

**Files:**
- Create: `tests/test_app.py`
- Create: `app.py`

**Interfaces:**
- Produces: `select_device_name`, `calculate_output_size`, `normalize_depth`, `smooth_depth`.

- [ ] Write helper behavior tests first and verify they fail because `app.py` does not exist.
- [ ] Implement only the helpers required by those tests.
- [ ] Re-run the helper tests and verify they pass.

### Task 2: Local inference and streaming video conversion

**Files:**
- Modify: `app.py`
- Modify: `tests/test_app.py`

**Interfaces:**
- Produces: local model loading, frame inference, FFmpeg raw-frame encoding, optional audio muxing, and `convert_video`.

- [ ] Add tests for model path mapping and FFmpeg command construction.
- [ ] Implement local-only model loading and frame inference.
- [ ] Implement streaming H.264 encoding and optional AAC audio muxing.
- [ ] Run all unit tests.

### Task 3: Gradio UI, dependencies, and documentation

**Files:**
- Create: `requirements.txt`
- Create: `README.md`
- Modify: `app.py`

**Interfaces:**
- Produces: `build_ui()` and executable `python app.py` entry point.

- [ ] Add the Gradio UI matching the design.
- [ ] Document Windows and macOS setup/start commands and local cache/model behavior.
- [ ] Validate Python syntax and imports.

### Task 4: Install and verify the current Windows environment

**Files:**
- Runtime directories: `.venv`, `cache`, `models`, `temp`, `uploads`, `outputs`.

**Interfaces:**
- Consumes the finished application and creates a runnable local environment.

- [ ] Create a project-local `.venv` from a supported Python interpreter.
- [ ] Install a compatible CUDA PyTorch build and remaining requirements with caches under the project.
- [ ] Download Small and Base model snapshots to `models/small` and `models/base`.
- [ ] Confirm `torch.cuda.is_available()`, FFmpeg, and both local snapshots.
- [ ] Run one image inference and a short synthetic video conversion.
- [ ] Run the full unit test suite once more.
