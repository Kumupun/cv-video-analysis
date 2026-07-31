from __future__ import annotations

import base64
import logging
from collections.abc import Iterable
from typing import Any

logger = logging.getLogger(__name__)


class YoloWorldTracker:
    """Stateful YOLO-World + ByteTrack adapter for one video task."""

    def __init__(
        self,
        *,
        model_id: str,
        classes: Iterable[str],
        confidence: float,
    ) -> None:
        import torch
        from ultralytics import YOLOWorld

        self.model = YOLOWorld(model_id)
        self.classes = tuple(classes)
        if not self.classes:
            raise ValueError("At least one YOLO-World class is required")
        self.model.set_classes(list(self.classes))
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.confidence = float(confidence)

        self.active_scene_id: str | None = None
        self.raw_to_scene_track_id: dict[int, int] = {}
        self.next_scene_track_id = 0
        self.expected_chunk_index: int | None = None
        self._pending_trackers: Any | None = None

    def _reset_for_scene(self, scene_id: str) -> None:
        self.active_scene_id = scene_id
        self.raw_to_scene_track_id.clear()
        self.next_scene_track_id = 0
        self._pending_trackers = None
        if getattr(self.model, "predictor", None) is not None:
            self.model.predictor = None

    @staticmethod
    def _to_bgr_frame(frame: Any) -> Any:
        import numpy as np

        array = np.asarray(frame)
        if array.ndim != 3 or array.shape[-1] != 3:
            raise ValueError("YOLO-World expects one RGB frame with shape [H,W,C]")
        return np.ascontiguousarray(array[..., ::-1])

    def _install_pending_trackers(self, first_frame: Any) -> None:
        if self._pending_trackers is None:
            return
        self.model.predict(source=[first_frame], verbose=False, device=self.device)
        predictor = getattr(self.model, "predictor", None)
        if predictor is None:
            raise RuntimeError("Ultralytics predictor was not initialized")
        predictor.trackers = self._pending_trackers
        self._pending_trackers = None

    def _map_track_id(self, raw_track_id: int) -> int:
        mapped = self.raw_to_scene_track_id.get(raw_track_id)
        if mapped is None:
            mapped = self.next_scene_track_id
            self.raw_to_scene_track_id[raw_track_id] = mapped
            self.next_scene_track_id += 1
        return mapped

    def process_interval(
        self,
        frames: Any,
        *,
        global_start_frame: int,
        scene_id: str,
    ) -> list[dict[str, Any]]:
        if scene_id != self.active_scene_id:
            self._reset_for_scene(scene_id)

        import numpy as np

        frame_array = np.asarray(frames)
        if frame_array.ndim != 4 or frame_array.shape[-1] != 3:
            raise ValueError("YOLO-World expects RGB frames with shape [T,H,W,C]")
        if len(frame_array) == 0:
            return []

        tracks: list[dict[str, Any]] = []
        for offset, rgb_frame in enumerate(frame_array):
            bgr_frame = self._to_bgr_frame(rgb_frame)
            if offset == 0:
                self._install_pending_trackers(bgr_frame)
            results = self.model.track(
                source=[bgr_frame],
                persist=True,
                tracker="bytetrack.yaml",
                verbose=False,
                conf=self.confidence,
                device=self.device,
            )
            if not results:
                continue
            result = results[0]
            boxes = result.boxes
            if boxes is None or boxes.id is None:
                continue
            xyxy = boxes.xyxy.detach().cpu().numpy()
            raw_ids = boxes.id.int().detach().cpu().numpy()
            confidences = boxes.conf.detach().cpu().numpy()
            class_ids = boxes.cls.int().detach().cpu().numpy()
            frame_height, frame_width = bgr_frame.shape[:2]

            for box, raw_id, confidence, class_id in zip(
                xyxy,
                raw_ids,
                confidences,
                class_ids,
                strict=True,
            ):
                x1, y1, x2, y2 = (float(value) for value in box)
                x1 = min(max(x1, 0.0), float(frame_width))
                y1 = min(max(y1, 0.0), float(frame_height))
                x2 = min(max(x2, x1), float(frame_width))
                y2 = min(max(y2, y1), float(frame_height))
                width = x2 - x1
                height = y2 - y1
                if width <= 0.0 or height <= 0.0:
                    continue

                names = self.model.names
                if isinstance(names, dict):
                    class_name = str(names[int(class_id)])
                else:
                    class_name = str(names[int(class_id)])
                tracks.append(
                    {
                        "frame": global_start_frame + offset,
                        "scene_id": scene_id,
                        "track_id": self._map_track_id(int(raw_id)),
                        "class_name": class_name,
                        "confidence": min(1.0, max(0.0, float(confidence))),
                        "bbox": {
                            "x": x1,
                            "y": y1,
                            "width": width,
                            "height": height,
                        },
                    }
                )
        return tracks

    def process_job(self, frames: Any, job: Any) -> list[dict[str, Any]]:
        import numpy as np

        array = np.asarray(frames)
        expected_count = job.context_end_frame - job.context_start_frame + 1
        if len(array) != expected_count:
            raise ValueError(
                "Ray frame tensor does not match tracking metadata: "
                f"expected {expected_count}, got {len(array)}"
            )

        if self.expected_chunk_index is None:
            self.expected_chunk_index = job.chunk_index
        if job.chunk_index != self.expected_chunk_index:
            raise RuntimeError(
                "Tracking chunks must be processed sequentially: "
                f"expected {self.expected_chunk_index}, got {job.chunk_index}"
            )

        tracks: list[dict[str, Any]] = []
        for scene in sorted(job.scenes, key=lambda item: item.start_frame):
            if not (
                job.valid_start_frame
                <= scene.start_frame
                <= scene.end_frame
                <= job.valid_end_frame
            ):
                raise ValueError("Tracking scene is outside the chunk valid range")
            local_start = scene.start_frame - job.context_start_frame
            local_end = scene.end_frame - job.context_start_frame + 1
            tracks.extend(
                self.process_interval(
                    array[local_start:local_end],
                    global_start_frame=scene.start_frame,
                    scene_id=scene.scene_id,
                )
            )

        self.expected_chunk_index += 1
        return tracks

    def export_state(self) -> str:
        import cloudpickle

        predictor = getattr(self.model, "predictor", None)
        trackers = getattr(predictor, "trackers", None) if predictor else None
        state = {
            "active_scene_id": self.active_scene_id,
            "raw_to_scene_track_id": self.raw_to_scene_track_id,
            "next_scene_track_id": self.next_scene_track_id,
            "expected_chunk_index": self.expected_chunk_index,
            "trackers": trackers,
        }
        try:
            payload = cloudpickle.dumps(state)
        except Exception:
            logger.warning(
                "ByteTrack internals could not be checkpointed; preserving IDs only",
                exc_info=True,
            )
            state["trackers"] = None
            payload = cloudpickle.dumps(state)
        return base64.b64encode(payload).decode("ascii")

    def restore_state(self, encoded_state: str) -> None:
        import cloudpickle

        state = cloudpickle.loads(base64.b64decode(encoded_state.encode("ascii")))
        self.active_scene_id = state.get("active_scene_id")
        self.raw_to_scene_track_id = {
            int(key): int(value)
            for key, value in state.get("raw_to_scene_track_id", {}).items()
        }
        self.next_scene_track_id = int(state.get("next_scene_track_id", 0))
        expected = state.get("expected_chunk_index")
        self.expected_chunk_index = int(expected) if expected is not None else None
        self._pending_trackers = state.get("trackers")
