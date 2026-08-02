from __future__ import annotations

from pathlib import Path
from typing import Any


class AutoShotDetector:
    """Loads AutoShot once and returns local hard/gradual transitions.

    Input is the backend contract tensor: RGB uint8 frames with shape
    ``[T, H, W, C]``. AutoShot exposes a one-hot boundary head and a many-hot
    transition-range head. Global-frame ownership is handled by the worker.
    """

    def __init__(
        self,
        *,
        weights_path: str | Path,
        architecture_dir: str | Path,
        threshold: float = 0.55,
        gradual_threshold: float = 0.75,
        gradual_boundary_threshold: float = 0.65,
        gradual_min_frames: int = 6,
        gradual_merge_gap_frames: int = 0,
        allow_download: bool = False,
        inference_window: int = 100,
        inference_overlap: int = 20,
    ) -> None:
        import torch

        from participant_4_cut_detection.autoshot_architecture import AutoShotSupernet

        if not 0.0 < threshold < 1.0:
            raise ValueError("threshold must be between 0 and 1")
        if not 0.0 < gradual_threshold < 1.0:
            raise ValueError("gradual_threshold must be between 0 and 1")
        if not 0.0 < gradual_boundary_threshold < 1.0:
            raise ValueError("gradual_boundary_threshold must be between 0 and 1")
        if gradual_min_frames < 2:
            raise ValueError("gradual_min_frames must be at least 2")
        if gradual_merge_gap_frames < 0:
            raise ValueError("gradual_merge_gap_frames cannot be negative")
        if inference_window < 8:
            raise ValueError("inference_window must be at least 8 frames")
        if not 0 <= inference_overlap < inference_window:
            raise ValueError("inference_overlap must be smaller than the window")

        self._torch = torch
        self.threshold = float(threshold)
        self.gradual_threshold = float(gradual_threshold)
        self.gradual_boundary_threshold = float(gradual_boundary_threshold)
        self.gradual_min_frames = int(gradual_min_frames)
        self.gradual_merge_gap_frames = int(gradual_merge_gap_frames)
        self.inference_window = int(inference_window)
        self.inference_overlap = int(inference_overlap)
        self.frame_width = 48
        self.frame_height = 27
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

    def _window_probabilities_resized(self, resized_frames: Any) -> tuple[Any, Any]:
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
        if not isinstance(output, tuple) or len(output) != 2:
            raise RuntimeError(
                "AutoShot gradual detection requires one-hot and many-hot heads"
            )
        boundary_logits, transition_logits = output
        boundary_probabilities = (
            self._torch.sigmoid(boundary_logits[0]).detach().cpu().numpy()
        )
        transition_probabilities = (
            self._torch.sigmoid(transition_logits[0]).detach().cpu().numpy()
        )
        return (
            boundary_probabilities.reshape(-1)[:original_length],
            transition_probabilities.reshape(-1)[:original_length],
        )

    def predict_transition_probabilities(self, frames: Any) -> tuple[Any, Any]:
        import numpy as np

        resized = self._resize_rgb_frames(frames)
        frame_count = len(resized)
        if frame_count <= self.inference_window:
            return self._window_probabilities_resized(resized)

        step = self.inference_window - self.inference_overlap
        boundary_score_sum = np.zeros(frame_count, dtype=np.float32)
        transition_score_sum = np.zeros(frame_count, dtype=np.float32)
        score_count = np.zeros(frame_count, dtype=np.float32)
        start = 0
        while start < frame_count:
            end = min(start + self.inference_window, frame_count)
            boundary_scores, transition_scores = self._window_probabilities_resized(
                resized[start:end]
            )
            boundary_score_sum[start:end] += boundary_scores
            transition_score_sum[start:end] += transition_scores
            score_count[start:end] += 1.0
            if end == frame_count:
                break
            start += step
        denominator = np.maximum(score_count, 1.0)
        return boundary_score_sum / denominator, transition_score_sum / denominator

    def predict_probabilities(self, frames: Any) -> Any:
        """Return the one-hot head for backwards-compatible diagnostics."""

        boundary_probabilities, _ = self.predict_transition_probabilities(frames)
        return boundary_probabilities

    @staticmethod
    def _positive_runs(mask: Any) -> list[tuple[int, int]]:
        import numpy as np

        padded = np.pad(np.asarray(mask, dtype=np.int8), (1, 1), constant_values=0)
        changes = np.diff(padded)
        starts = np.flatnonzero(changes == 1)
        ends_exclusive = np.flatnonzero(changes == -1)
        return [
            (int(start), int(end_exclusive - 1))
            for start, end_exclusive in zip(starts, ends_exclusive, strict=True)
        ]

    def _merge_runs(self, runs: list[tuple[int, int]]) -> list[tuple[int, int]]:
        merged: list[tuple[int, int]] = []
        for start, end in runs:
            if merged and start - merged[-1][1] - 1 <= self.gradual_merge_gap_frames:
                merged[-1] = (merged[-1][0], end)
            else:
                merged.append((start, end))
        return merged

    def detect_transitions(self, frames: Any) -> list[dict[str, float | int | str]]:
        import numpy as np

        boundary_probabilities, transition_probabilities = (
            self.predict_transition_probabilities(frames)
        )
        gradual_runs = self._merge_runs(
            self._positive_runs(transition_probabilities > self.gradual_threshold)
        )
        gradual_runs = [
            (start, end)
            for start, end in gradual_runs
            if end - start + 1 >= self.gradual_min_frames
        ]
        confirmed_gradual_runs: list[tuple[int, int, float]] = []
        for start, end in gradual_runs:
            boundary_segment = boundary_probabilities[start : end + 1]
            transition_segment = transition_probabilities[start : end + 1]
            boundary_peak_offset = int(np.argmax(boundary_segment))
            boundary_peak = float(boundary_segment[boundary_peak_offset])
            transition_peak = float(np.max(transition_segment))
            midpoint = (end - start) / 2.0
            midpoint_tolerance = max(1.0, (end - start) * 0.4)
            if boundary_peak < self.gradual_boundary_threshold:
                continue
            if abs(boundary_peak_offset - midpoint) > midpoint_tolerance:
                continue
            confirmed_gradual_runs.append(
                (start, end, float(np.sqrt(boundary_peak * transition_peak)))
            )

        transitions: list[dict[str, float | int | str]] = []
        for start, end, confidence in confirmed_gradual_runs:
            transitions.append(
                {
                    "type": "gradual_transition",
                    "local_start_frame": start,
                    "local_end_frame": end,
                    "confidence": confidence,
                }
            )

        for start, end in self._positive_runs(boundary_probabilities > self.threshold):
            peak = int(start + np.argmax(boundary_probabilities[start : end + 1]))
            if any(
                range_start <= peak <= range_end
                for range_start, range_end, _ in confirmed_gradual_runs
            ):
                continue
            transitions.append(
                {
                    "type": "hard_cut",
                    "local_frame": peak,
                    "confidence": float(boundary_probabilities[peak]),
                }
            )
        return sorted(
            transitions,
            key=lambda item: int(
                item.get("local_frame", item.get("local_end_frame", 0))
            ),
        )

    def detect_cuts(self, frames: Any) -> list[dict[str, float | int | str]]:
        """Compatibility alias for older actor code and local integrations."""

        return self.detect_transitions(frames)
