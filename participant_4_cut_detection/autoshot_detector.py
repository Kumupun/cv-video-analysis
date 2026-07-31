from __future__ import annotations

from pathlib import Path
from typing import Any

from participant_4_cut_detection.autoshot_architecture import AutoShotSupernet


class AutoShotDetector:
    """Loads AutoShot once and returns local hard-cut candidates.

    Input is the backend contract tensor: RGB uint8 frames with shape
    ``[T, H, W, C]``. The detector returns local frame indices and confidence
    values; global-frame ownership is handled by the Redis worker adapter.
    """

    def __init__(
        self,
        *,
        weights_path: str | Path,
        architecture_dir: str | Path,
        threshold: float = 0.55,
        allow_download: bool = False,
        inference_window: int = 100,
        inference_overlap: int = 20,
    ) -> None:
        import torch

        if not 0.0 < threshold < 1.0:
            raise ValueError("threshold must be between 0 and 1")
        if inference_window < 8:
            raise ValueError("inference_window must be at least 8 frames")
        if not 0 <= inference_overlap < inference_window:
            raise ValueError("inference_overlap must be smaller than the window")

        self._torch = torch
        self.threshold = float(threshold)
        self.inference_window = int(inference_window)
        self.inference_overlap = int(inference_overlap)
        self.frame_width = 48
        self.frame_height = 27
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Retain the two arguments for compatibility with existing environment
        # files. The architecture is now vendored with the worker, so startup
        # never downloads or imports executable Python source from /models.
        del architecture_dir, allow_download
        self.model = AutoShotSupernet().eval()

        checkpoint_path = Path(weights_path).expanduser().resolve()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"AutoShot weights were not found at {checkpoint_path}. "
                "The real ML worker must never run with random weights."
            )

        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )
        state = checkpoint.get("net", checkpoint.get("state_dict", checkpoint))
        if not isinstance(state, dict):
            raise ValueError("AutoShot checkpoint does not contain a state dictionary")

        clean_state: dict[str, Any] = {}
        for key, value in state.items():
            clean_key = str(key)
            for prefix in ("module.", "model.", "net."):
                if clean_key.startswith(prefix):
                    clean_key = clean_key[len(prefix) :]
            clean_state[clean_key] = value

        try:
            self.model.load_state_dict(clean_state, strict=True)
        except RuntimeError as exc:
            raise ValueError(
                "AutoShot checkpoint does not exactly match the bundled architecture"
            ) from exc
        self.model = self.model.to(self.device)

    def _resize_rgb_frames(self, frames: Any) -> Any:
        """Downscale uint8 frames before padding or float conversion.

        Padding a short 1080p/4K chunk to the 100-frame model window at source
        resolution can allocate several gigabytes and make the native Ray actor
        disappear without a Python traceback. AutoShot only consumes 48x27
        inputs, so all temporal padding is performed after this bounded resize.
        """

        import cv2
        import numpy as np

        array = np.asarray(frames)
        if array.ndim != 4 or array.shape[-1] != 3:
            raise ValueError("AutoShot expects RGB frames with shape [T,H,W,C]")
        if len(array) == 0:
            raise ValueError("AutoShot received an empty frame tensor")
        if array.dtype != np.uint8:
            array = np.clip(array, 0, 255).astype(np.uint8, copy=False)

        resized = np.empty(
            (len(array), self.frame_height, self.frame_width, 3),
            dtype=np.uint8,
        )
        for index, frame in enumerate(array):
            interpolation = (
                cv2.INTER_AREA
                if frame.shape[0] >= self.frame_height
                and frame.shape[1] >= self.frame_width
                else cv2.INTER_LINEAR
            )
            resized[index] = cv2.resize(
                frame,
                (self.frame_width, self.frame_height),
                interpolation=interpolation,
            )
        return resized

    def _preprocess_resized(self, resized_frames: Any) -> Any:
        import numpy as np

        array = np.ascontiguousarray(resized_frames)
        tensor = self._torch.from_numpy(array).permute(0, 3, 1, 2)
        tensor = tensor.to(
            device=self.device,
            dtype=self._torch.float32,
            non_blocking=self.device.type == "cuda",
        )
        return tensor.permute(1, 0, 2, 3).unsqueeze(0)

    def _window_probabilities_resized(self, resized_frames: Any) -> Any:
        import numpy as np

        original_length = len(resized_frames)
        if original_length <= 0:
            raise ValueError("AutoShot received an empty frame tensor")

        if original_length < self.inference_window:
            pad_count = self.inference_window - original_length
            padding = np.repeat(resized_frames[-1:], pad_count, axis=0)
            model_frames = np.concatenate((resized_frames, padding), axis=0)
        else:
            model_frames = resized_frames[: self.inference_window]

        tensor = self._preprocess_resized(model_frames)
        with self._torch.inference_mode():
            output = self.model(tensor)
            if self.device.type == "cuda":
                self._torch.cuda.synchronize(self.device)
        logits = output[0] if isinstance(output, tuple) else output
        probabilities = self._torch.sigmoid(logits[0]).detach().cpu().numpy()
        return probabilities.reshape(-1)[:original_length]

    def predict_probabilities(self, frames: Any) -> Any:
        import numpy as np

        resized = self._resize_rgb_frames(frames)
        frame_count = len(resized)
        if frame_count <= self.inference_window:
            return self._window_probabilities_resized(resized)

        step = self.inference_window - self.inference_overlap
        score_sum = np.zeros(frame_count, dtype=np.float32)
        score_count = np.zeros(frame_count, dtype=np.float32)
        start = 0
        while start < frame_count:
            end = min(start + self.inference_window, frame_count)
            window_scores = self._window_probabilities_resized(resized[start:end])
            score_sum[start:end] += window_scores
            score_count[start:end] += 1.0
            if end == frame_count:
                break
            start += step
        return score_sum / np.maximum(score_count, 1.0)

    def detect_cuts(self, frames: Any) -> list[dict[str, float | int]]:
        import numpy as np

        probabilities = self.predict_probabilities(frames)
        mask = probabilities > self.threshold
        padded = np.pad(mask.astype(np.int8), (1, 1), constant_values=0)
        changes = np.diff(padded)
        starts = np.flatnonzero(changes == 1)
        ends_exclusive = np.flatnonzero(changes == -1)

        cuts: list[dict[str, float | int]] = []
        for start, end_exclusive in zip(starts, ends_exclusive, strict=True):
            peak = int(start + np.argmax(probabilities[start:end_exclusive]))
            cuts.append(
                {
                    "local_frame": peak,
                    "confidence": float(probabilities[peak]),
                }
            )
        return cuts
