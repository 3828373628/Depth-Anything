# Depth Anything V2 深度视频转换器

一个完全本地运行的 Python/Gradio 工具，将 MP4/MOV 视频逐帧转换为灰度 Depth Anything V2 深度视频，并导出兼容的 H.264 MP4。

## 功能

- MP4 / MOV 上传
- Depth Anything V2 Small / Base 本地模型切换
- Windows 自动优先 NVIDIA CUDA
- Apple Silicon macOS 自动优先 MPS
- 无 GPU 加速时自动回退 CPU
- Original / 720p / 1080p / 2160p 输出
- 黑白反转
- 时间指数平滑，减少帧间闪烁
- 可选保留原始音频
- FFmpeg H.264 + yuv420p 输出
- 不把每一帧保存为图片，避免临时空间爆炸

## 项目目录

```text
E:\VideoTool\shenduzhuanhuan\
├─ app.py
├─ requirements.txt
├─ README.md
├─ .venv\
├─ models\
│  ├─ small\
│  └─ base\
├─ cache\
├─ temp\
├─ uploads\
└─ outputs\
```

模型在程序运行时只从 `models` 目录读取；准备完成后可断网使用。

当前验证通过的核心组合为 Python 3.13、PyTorch 2.11.0、torchvision 0.26.0、Transformers 5.17.0、Gradio 6.27.0 和 OpenCV 5.0.0.93。`requirements.txt` 已固定这些版本，减少以后依赖升级造成的兼容性变化。

程序启动时会把 Hugging Face、PyTorch、TorchInductor、Gradio 和 Python 临时目录指向本项目的 `cache` / `temp` 目录，避免视频处理和模型缓存大量占用 C 盘。

模型准备完成后，程序运行时会启用 Hugging Face 离线模式并关闭 Gradio 匿名分析，因此正常转换不会主动下载模型或发送 Gradio 遥测。首次安装依赖和准备模型仍然需要网络连接。

## 环境要求与验证范围

- Python 3.10+；本项目实际验证使用 Python 3.13。
- Windows 当前已在 RTX 4060 Laptop GPU 上完成 CUDA、Small/Base 推理、Gradio 启动和带音频视频转换的实机验证。
- Apple Silicon macOS 代码会自动尝试 MPS，但当前没有 Mac 实机可用于最终验证；请以 `torch.backends.mps.is_available()` 的检查结果为准。
- 当前固定的 `opencv-python==5.0.0.93` Apple Silicon wheel 标记为 macOS 13.0+，因此当前依赖组合建议使用 macOS 13 或更新版本；为减少 MPS 兼容性问题，优先使用较新的 macOS。

## Windows 安装（NVIDIA GPU 推荐）

当前这台机器实际验证使用 `E:\Software\Python313\python.exe`。建议直接用它创建项目虚拟环境，避免系统中其他 Python/Conda 版本被误用。

CMD：

```bat
cd /d E:\VideoTool\shenduzhuanhuan
E:\Software\Python313\python.exe -m venv .venv
```

PowerShell：

```powershell
Set-Location E:\VideoTool\shenduzhuanhuan
E:\Software\Python313\python.exe -m venv .venv
```

安装 CUDA 12.8 版 PyTorch：

```bat
.venv\Scripts\python.exe -m pip install --isolated --cache-dir cache\pip torch==2.11.0+cu128 torchvision==0.26.0+cu128 --index-url https://download.pytorch.org/whl/cu128
```

下文所有以 `.venv\Scripts\python.exe` 开头的 CMD 命令，在 PowerShell 中把开头改为 `.\.venv\Scripts\python.exe`，其余参数相同。

安装其余依赖：

```bat
.venv\Scripts\python.exe -m pip install --isolated --cache-dir cache\pip -r requirements.txt
```

> 本机用户级 pip 配置包含全局 `target`，因此这里特意使用 `--isolated`，避免依赖被装到项目外。

检查 CUDA：

```bat
.venv\Scripts\python.exe -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

## macOS 安装（Apple Silicon）

```bash
cd /path/to/shenduzhuanhuan
python3 -m venv .venv
./.venv/bin/python -m pip install -r requirements.txt
```

检查 MPS：

```bash
./.venv/bin/python -c "import torch; print(torch.backends.mps.is_available())"
```

macOS 不需要 CUDA。Apple Silicon 上 PyTorch 会使用 MPS；不支持时程序自动回退 CPU。

## FFmpeg

程序要求 `ffmpeg` 可以通过 PATH 调用。也可以设置 `FFMPEG_PATH` 指向 FFmpeg 可执行文件。

Windows 示例：

```bat
set FFMPEG_PATH=E:\AI\Tool\ffmpeg\ffmpeg-9.0.1-essentials_build\bin\ffmpeg.exe
```

PowerShell 示例：

```powershell
$env:FFMPEG_PATH = "E:\AI\Tool\ffmpeg\ffmpeg-9.0.1-essentials_build\bin\ffmpeg.exe"
```

macOS 可使用 Homebrew：

```bash
brew install ffmpeg
```

## 下载本地模型

安装依赖后执行：

```bat
.venv\Scripts\python.exe -c "from huggingface_hub import snapshot_download; snapshot_download('depth-anything/Depth-Anything-V2-Small-hf', local_dir='models/small', allow_patterns=['config.json','preprocessor_config.json','model.safetensors'])"
.venv\Scripts\python.exe -c "from huggingface_hub import snapshot_download; snapshot_download('depth-anything/Depth-Anything-V2-Base-hf', local_dir='models/base', allow_patterns=['config.json','preprocessor_config.json','model.safetensors'])"
```

macOS 把 `.venv\Scripts\python.exe` 改成 `./.venv/bin/python`。

模型许可证需要单独注意：当前 Hugging Face 的 Depth Anything V2 Small-hf 标记为 Apache-2.0；Base-hf 标记为 CC-BY-NC-4.0（非商业许可）。如果项目用于商业用途，请在使用 Base 前自行确认许可证是否满足用途要求。

## 启动

### Windows 日常使用（推荐）

项目已经准备完成后，平时不需要重新安装环境或重新下载模型。

直接双击项目目录中的：

```text
启动深度转换器.bat
```

浏览器会自动打开 Gradio 页面。关闭时在启动窗口按 `Ctrl+C`，或直接关闭该窗口即可。

如需放到桌面，可右键 `启动深度转换器.bat` → “发送到” → “桌面快捷方式”。不要把 BAT 文件本身移动出项目目录。

也可以手动启动：

Windows：

```bat
cd /d E:\VideoTool\shenduzhuanhuan
.venv\Scripts\python.exe app.py
```

PowerShell：

```powershell
Set-Location E:\VideoTool\shenduzhuanhuan
.\.venv\Scripts\python.exe app.py
```

macOS：

```bash
cd /path/to/shenduzhuanhuan
./.venv/bin/python app.py
```

启动后浏览器会自动打开 Gradio 页面。

## 说明

时间平滑强度为 `0` 时不平滑；数值越接近 `0.95`，闪烁抑制越强，但快速运动的深度变化也会变慢。输出分辨率会保持原视频宽高比，并自动保证 H.264 所需的偶数宽高。

“保留原始音频”表示保留输入视频中的音频内容。为保证最终 MP4 的兼容性，音频会通过 FFmpeg 重新编码为 AAC 192 kbps，而不是对原音频码流做逐字节复制。

输入视频按 OpenCV 报告的 FPS 编码为恒定帧率（CFR）输出。普通 CFR 视频会保持正常时长；对于手机等设备生成的可变帧率（VFR）视频，程序不会保留每一帧原始时间戳，因此极端 VFR 素材可能出现轻微节奏差异。如果必须逐帧保留 VFR 时间戳，需要改为时间戳感知的 FFmpeg 解码/编码流程。

Base 模型比 Small 更慢、显存占用更高，但通常细节更好。当前项目不预下载 Large 模型。
