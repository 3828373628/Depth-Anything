import os
import tempfile
import unittest
from datetime import datetime
from unittest.mock import patch

import numpy as np

from pathlib import Path

from app import (
    CACHE_DIR,
    TEMP_DIR,
    _load_model,
    _make_output_path,
    build_video_encode_command,
    calculate_output_size,
    configure_project_caches,
    get_model_dir,
    normalize_depth,
    select_device_name,
    smooth_depth,
)


class DeviceSelectionTests(unittest.TestCase):
    def test_cuda_has_priority(self):
        self.assertEqual(select_device_name(cuda_available=True, mps_available=True), "cuda")

    def test_mps_is_second_choice(self):
        self.assertEqual(select_device_name(cuda_available=False, mps_available=True), "mps")

    def test_cpu_is_fallback(self):
        self.assertEqual(select_device_name(cuda_available=False, mps_available=False), "cpu")


class OutputSizeTests(unittest.TestCase):
    def test_original_size_is_even(self):
        self.assertEqual(calculate_output_size(1919, 1079, "Original"), (1918, 1078))

    def test_720p_landscape_preserves_aspect_ratio(self):
        self.assertEqual(calculate_output_size(1920, 1080, "720p"), (1280, 720))

    def test_720p_portrait_preserves_aspect_ratio(self):
        self.assertEqual(calculate_output_size(1080, 1920, "720p"), (720, 1280))


class DepthProcessingTests(unittest.TestCase):
    def test_normalize_depth_maps_range_to_uint8(self):
        depth = np.array([[2.0, 4.0], [6.0, 10.0]], dtype=np.float32)
        result = normalize_depth(depth, invert=False)
        self.assertEqual(result.dtype, np.uint8)
        self.assertEqual(int(result.min()), 0)
        self.assertEqual(int(result.max()), 255)

    def test_normalize_depth_can_invert(self):
        depth = np.array([[0.0, 1.0]], dtype=np.float32)
        normal = normalize_depth(depth, invert=False)
        inverted = normalize_depth(depth, invert=True)
        np.testing.assert_array_equal(inverted, 255 - normal)

    def test_smoothing_blends_previous_and_current(self):
        previous = np.full((2, 2), 100.0, dtype=np.float32)
        current = np.full((2, 2), 200.0, dtype=np.float32)
        result = smooth_depth(current, previous, 0.75)
        np.testing.assert_allclose(result, 125.0)

    def test_smoothing_without_previous_returns_current(self):
        current = np.full((2, 2), 50.0, dtype=np.float32)
        result = smooth_depth(current, None, 0.9)
        np.testing.assert_array_equal(result, current)


class ConfigurationTests(unittest.TestCase):
    def test_output_path_is_unique_within_same_second(self):
        source = Path("clip.mov")
        first = _make_output_path(source, datetime(2026, 9, 17, 19, 0, 0, 1))
        second = _make_output_path(source, datetime(2026, 9, 17, 19, 0, 0, 2))
        self.assertNotEqual(first, second)

    def test_model_loader_reports_missing_required_file(self):
        with tempfile.TemporaryDirectory() as directory:
            model_dir = Path(directory)
            (model_dir / "config.json").write_text("{}", encoding="utf-8")
            (model_dir / "preprocessor_config.json").write_text("{}", encoding="utf-8")
            with patch("app.get_model_dir", return_value=model_dir):
                with self.assertRaisesRegex(FileNotFoundError, "model.safetensors"):
                    _load_model("Small", "cpu")

    def test_localhost_bypasses_system_proxy(self):
        with patch.dict(os.environ, {"NO_PROXY": "example.com"}, clear=False):
            configure_project_caches()
            upper = {item.strip() for item in os.environ["NO_PROXY"].split(",")}
            lower = {item.strip() for item in os.environ["no_proxy"].split(",")}
            self.assertTrue({"127.0.0.1", "localhost"}.issubset(upper))
            self.assertTrue({"127.0.0.1", "localhost"}.issubset(lower))
            self.assertIn("example.com", upper | lower)

    def test_project_caches_and_temp_stay_under_project(self):
        keys = [
            "HF_HOME",
            "HF_HUB_OFFLINE",
            "TORCH_HOME",
            "PIP_CACHE_DIR",
            "TORCHINDUCTOR_CACHE_DIR",
            "GRADIO_ANALYTICS_ENABLED",
            "GRADIO_TEMP_DIR",
            "TEMP",
            "TMP",
            "TMPDIR",
        ]
        with patch.dict(os.environ, {}, clear=False):
            for key in keys:
                os.environ.pop(key, None)
            old_tempdir = tempfile.tempdir
            try:
                configure_project_caches()
                self.assertEqual(os.environ["HF_HOME"], str(CACHE_DIR / "huggingface"))
                self.assertEqual(os.environ["HF_HUB_OFFLINE"], "1")
                self.assertEqual(os.environ["TORCHINDUCTOR_CACHE_DIR"], str(CACHE_DIR / "torch" / "inductor"))
                self.assertEqual(os.environ["GRADIO_ANALYTICS_ENABLED"], "False")
                self.assertEqual(os.environ["GRADIO_TEMP_DIR"], str(TEMP_DIR / "gradio"))
                self.assertEqual(os.environ["TEMP"], str(TEMP_DIR))
                self.assertEqual(os.environ["TMP"], str(TEMP_DIR))
                self.assertEqual(os.environ["TMPDIR"], str(TEMP_DIR))
                self.assertEqual(tempfile.tempdir, str(TEMP_DIR))
            finally:
                tempfile.tempdir = old_tempdir

    def test_model_dir_maps_small_and_base(self):
        self.assertEqual(get_model_dir("Small").name, "small")
        self.assertEqual(get_model_dir("Base").name, "base")

    def test_encode_command_uses_h264_and_yuv420p(self):
        command = build_video_encode_command(
            ffmpeg_path=Path("ffmpeg"),
            width=1280,
            height=720,
            fps=30.0,
            output_path=Path("silent.mp4"),
        )
        self.assertIn("libx264", command)
        self.assertIn("yuv420p", command)
        self.assertIn("1280x720", command)


if __name__ == "__main__":
    unittest.main()
