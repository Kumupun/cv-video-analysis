from __future__ import annotations

import base64
import logging
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _DetectionBatch:
    """Minimal NumPy-backed Ultralytics Boxes interface used by BYTETracker."""

    data: Any

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: Any) -> _DetectionBatch:
        return _DetectionBatch(self.data[index])

    @property
    def xyxy(self) -> Any:
        return self.data[:, :4]

    @property
    def xywh(self) -> Any:
        import numpy as np

        result = np.asarray(self.xyxy, dtype=np.float32).copy()
        if len(result):
            result[:, 2:4] -= result[:, 0:2]
            result[:, 0:2] += result[:, 2:4] / 2
        return result

    @property
    def conf(self) -> Any:
        return self.data[:, 4]

    @property
    def cls(self) -> Any:
        return self.data[:, 5]


class YoloWorldTracker:
    """YOLO-World + optional YOLOE ensemble feeding one stateful ByteTrack."""

    def __init__(
        self,
        *,
        model_id: str,
        classes: Iterable[str],
        confidence: float,
        image_size: int,
        allow_download: bool,
        ensemble_model_id: str | None = None,
        ensemble_iou_threshold: float = 0.55,
        parallel_inference: bool = True,
        track_low_threshold: float = 0.10,
        new_track_threshold: float = 0.55,
        track_buffer: int = 45,
        match_threshold: float = 0.75,
        fuse_score: bool = True,
    ) -> None:
        import torch
        from ultralytics import YOLO

        resolved_model_id = self._resolve_model_id(
            model_id,
            allow_download,
            setting_name="TRACKING_MODEL_ID",
        )
        self.model = YOLO(resolved_model_id)
        loaded_architecture = type(getattr(self.model, "model", None)).__name__
        if loaded_architecture != "WorldModel":
            raise ValueError(
                "TRACKING_MODEL_ID must contain a YOLO-World WorldModel; "
                f"loaded architecture is {loaded_architecture!r}"
            )

        self.classes = tuple(
            class_name.strip() for class_name in classes if class_name.strip()
        )
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.classes:
            self.model.set_classes(list(self.classes))

            if hasattr(self.model.model, "clip_model"):
                self.model.model.clip_model = None
        self.model.to(self.device)
        self.model_names = self._normalize_names(self.model.names)

        normalized_ensemble_id = (ensemble_model_id or "").strip()
        self.ensemble_model = None
        self.ensemble_model_names: dict[int, str] = {}
        if normalized_ensemble_id:
            resolved_ensemble_id = self._resolve_model_id(
                normalized_ensemble_id,
                allow_download,
                setting_name="TRACKING_ENSEMBLE_MODEL_ID",
            )
            self.ensemble_model = YOLO(resolved_ensemble_id)
            ensemble_architecture = type(
                getattr(self.ensemble_model, "model", None)
            ).__name__
            if ensemble_architecture not in {
                "YOLOEModel",
                "YOLOESegModel",
                "WorldModel",
                "DetectionModel",
            }:
                raise ValueError(
                    "TRACKING_ENSEMBLE_MODEL_ID must contain an Ultralytics "
                    "detection checkpoint; loaded architecture is "
                    f"{ensemble_architecture!r}"
                )
            self.ensemble_model.to(self.device)
            self.ensemble_model_names = self._normalize_names(self.ensemble_model.names)

        self.output_names = self.model_names
        self._class_id_by_key = self._build_class_index(self.output_names)
        self.confidence = float(confidence)
        self.track_low_threshold = min(
            float(track_low_threshold),
            self.confidence,
        )
        self.new_track_threshold = max(
            float(new_track_threshold),
            self.confidence,
        )
        self.track_buffer = int(track_buffer)
        self.match_threshold = float(match_threshold)
        self.fuse_score = bool(fuse_score)
        self.image_size = int(image_size)
        self.ensemble_iou_threshold = float(ensemble_iou_threshold)
        self.parallel_inference = bool(parallel_inference and self.ensemble_model)
        self._inference_pool = (
            ThreadPoolExecutor(max_workers=2, thread_name_prefix="yolo-ensemble")
            if self.parallel_inference
            else None
        )
        self._tracker_args = SimpleNamespace(
            tracker_type="bytetrack",
            track_high_thresh=self.confidence,
            track_low_thresh=self.track_low_threshold,
            new_track_thresh=self.new_track_threshold,
            track_buffer=self.track_buffer,
            match_thresh=self.match_threshold,
            fuse_score=self.fuse_score,
        )

        self.active_scene_id: str | None = None
        self.raw_to_scene_track_id: dict[int, int] = {}
        self.next_scene_track_id = 0
        self.expected_chunk_index: int | None = None
        self.byte_tracker = self._new_byte_tracker()

        logger.info(
            "Tracking detector ensemble initialized",
            extra={
                "primary_model": str(resolved_model_id),
                "ensemble_model": (
                    str(resolved_ensemble_id) if self.ensemble_model else None
                ),
                "parallel_inference": self.parallel_inference,
                "class_count": len(self.output_names),
            },
        )

    @staticmethod
    def _resolve_model_id(
        model_id: str,
        allow_download: bool,
        *,
        setting_name: str,
    ) -> str:
        candidate = Path(model_id).expanduser()
        if candidate.is_file():
            return str(candidate)
        if allow_download:
            return model_id
        raise FileNotFoundError(
            "Tracking checkpoint is missing and runtime downloads are disabled: "
            f"{candidate}. Put the supplied checkpoint in models/ and set "
            f"{setting_name} to its /models path."
        )

    @staticmethod
    def _normalize_names(names: Any) -> dict[int, str]:
        if isinstance(names, dict):
            return {int(class_id): str(name) for class_id, name in names.items()}
        if isinstance(names, (list, tuple)):
            return {class_id: str(name) for class_id, name in enumerate(names)}
        raise TypeError("Ultralytics model did not expose a valid class-name mapping")

    @staticmethod
    def _class_key(name: str) -> str:
        return "_".join(name.strip().lower().replace("-", " ").split())

    @classmethod
    def _build_class_index(cls, names: dict[int, str]) -> dict[str, int]:
        result: dict[str, int] = {}
        for class_id, name in names.items():
            key = cls._class_key(name)
            previous = result.get(key)
            if previous is not None and previous != class_id:
                raise ValueError(f"Duplicate normalized tracking class name: {name!r}")
            result[key] = class_id
        return result

    def _new_byte_tracker(self) -> Any:
        from ultralytics.trackers.byte_tracker import BYTETracker

        return BYTETracker(self._tracker_args)

    def _inference_options(self) -> dict[str, Any]:
        return {
            "verbose": False,

            "conf": self.track_low_threshold,
            "device": self.device,
            "imgsz": self.image_size,
        }

    def _reset_for_scene(self, scene_id: str) -> None:
        self.active_scene_id = scene_id
        self.raw_to_scene_track_id.clear()
        self.next_scene_track_id = 0
        self.byte_tracker = self._new_byte_tracker()

    @staticmethod
    def _to_bgr_frame(frame: Any) -> Any:
        import numpy as np

        array = np.asarray(frame)
        if array.ndim != 3 or array.shape[-1] != 3:
            raise ValueError("Ultralytics expects one RGB frame with shape [H,W,C]")
        return np.ascontiguousarray(array[..., ::-1])

    @staticmethod
    def _to_numpy(value: Any) -> Any:
        import numpy as np

        if hasattr(value, "detach"):
            value = value.detach()
        if hasattr(value, "cpu"):
            value = value.cpu()
        if hasattr(value, "numpy"):
            value = value.numpy()
        return np.asarray(value)

    def _predict(self, model: Any, frame: Any) -> Any | None:
        results = model.predict(source=[frame], **self._inference_options())
        return results[0] if results else None

    def _predict_models(self, frame: Any) -> tuple[Any | None, Any | None]:
        if self.ensemble_model is None:
            return self._predict(self.model, frame), None
        if self._inference_pool is None:
            return (
                self._predict(self.model, frame),
                self._predict(self.ensemble_model, frame),
            )
        primary_future = self._inference_pool.submit(self._predict, self.model, frame)
        ensemble_future = self._inference_pool.submit(
            self._predict,
            self.ensemble_model,
            frame,
        )
        return primary_future.result(), ensemble_future.result()

    def _extract_detections(
        self,
        result: Any | None,
        fallback_names: dict[int, str],
    ) -> list[list[float]]:
        if result is None or getattr(result, "boxes", None) is None:
            return []
        boxes = result.boxes
        if len(boxes) == 0:
            return []
        result_names = self._normalize_names(getattr(result, "names", fallback_names))
        xyxy = self._to_numpy(boxes.xyxy)
        confidences = self._to_numpy(boxes.conf)
        class_ids = self._to_numpy(boxes.cls).astype(int)
        detections: list[list[float]] = []
        for box, confidence, source_class_id in zip(
            xyxy,
            confidences,
            class_ids,
            strict=True,
        ):
            source_name = result_names.get(int(source_class_id))
            if source_name is None:
                continue
            output_class_id = self._class_id_by_key.get(self._class_key(source_name))


            if output_class_id is None:
                continue
            detections.append(
                [
                    *(float(value) for value in box),
                    float(confidence),
                    float(output_class_id),
                ]
            )
        return detections

    @staticmethod
    def _iou(box: Any, other: Any) -> float:
        x1 = max(float(box[0]), float(other[0]))
        y1 = max(float(box[1]), float(other[1]))
        x2 = min(float(box[2]), float(other[2]))
        y2 = min(float(box[3]), float(other[3]))
        intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        if intersection <= 0.0:
            return 0.0
        box_area = max(0.0, float(box[2]) - float(box[0])) * max(
            0.0, float(box[3]) - float(box[1])
        )
        other_area = max(0.0, float(other[2]) - float(other[0])) * max(
            0.0, float(other[3]) - float(other[1])
        )
        union = box_area + other_area - intersection
        return intersection / union if union > 0.0 else 0.0

    def _fuse_detections(self, detections: list[list[float]]) -> _DetectionBatch:
        import numpy as np

        kept: list[list[float]] = []


        for candidate in sorted(detections, key=lambda item: item[4], reverse=True):
            if any(
                int(existing[5]) == int(candidate[5])
                and self._iou(existing, candidate) >= self.ensemble_iou_threshold
                for existing in kept
            ):
                continue
            kept.append(candidate)
        data = np.asarray(kept, dtype=np.float32)
        if data.size == 0:
            data = np.empty((0, 6), dtype=np.float32)
        return _DetectionBatch(data.reshape((-1, 6)))

    def _detect_frame(self, frame: Any) -> _DetectionBatch:
        primary_result, ensemble_result = self._predict_models(frame)
        detections = self._extract_detections(primary_result, self.model_names)
        detections.extend(
            self._extract_detections(
                ensemble_result,
                self.ensemble_model_names,
            )
        )
        return self._fuse_detections(detections)

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
        reset_tracker: bool = False,
    ) -> list[dict[str, Any]]:
        if reset_tracker or scene_id != self.active_scene_id:
            self._reset_for_scene(scene_id)

        import numpy as np

        frame_array = np.asarray(frames)
        if frame_array.ndim != 4 or frame_array.shape[-1] != 3:
            raise ValueError("Ultralytics expects RGB frames with shape [T,H,W,C]")
        if len(frame_array) == 0:
            return []

        tracks: list[dict[str, Any]] = []
        for offset, rgb_frame in enumerate(frame_array):
            bgr_frame = self._to_bgr_frame(rgb_frame)
            detections = self._detect_frame(bgr_frame)
            tracked_boxes = self.byte_tracker.update(detections, bgr_frame)
            frame_height, frame_width = bgr_frame.shape[:2]

            for tracked in tracked_boxes:
                if len(tracked) < 8:
                    raise RuntimeError("ByteTrack returned an unexpected result shape")
                confidence = float(tracked[5])

                if confidence < self.confidence:
                    continue
                class_id = int(tracked[6])
                class_name = self.output_names.get(class_id)
                if class_name is None:
                    continue
                x1, y1, x2, y2 = (float(value) for value in tracked[:4])
                x1 = min(max(x1, 0.0), float(frame_width))
                y1 = min(max(y1, 0.0), float(frame_height))
                x2 = min(max(x2, x1), float(frame_width))
                y2 = min(max(y2, y1), float(frame_height))
                width = x2 - x1
                height = y2 - y1
                if width <= 0.0 or height <= 0.0:
                    continue

                tracks.append(
                    {
                        "frame": global_start_frame + offset,
                        "scene_id": scene_id,
                        "track_id": self._map_track_id(int(tracked[4])),
                        "class_name": class_name,
                        "confidence": min(1.0, max(0.0, confidence)),
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

        reset_after_frames = {
            (transition.frame if transition.frame is not None else transition.end_frame)
            + 1
            for transition in job.transitions
            if transition.frame is not None or transition.end_frame is not None
        }
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
                    reset_tracker=scene.start_frame in reset_after_frames,
                )
            )

        self.expected_chunk_index += 1
        return tracks

    def export_state(self) -> str:
        import cloudpickle

        state = {
            "active_scene_id": self.active_scene_id,
            "raw_to_scene_track_id": self.raw_to_scene_track_id,
            "next_scene_track_id": self.next_scene_track_id,
            "expected_chunk_index": self.expected_chunk_index,
            "byte_tracker": self.byte_tracker,
        }
        try:
            payload = cloudpickle.dumps(state)
        except Exception:
            logger.warning(
                "ByteTrack internals could not be checkpointed; preserving IDs only",
                exc_info=True,
            )
            state["byte_tracker"] = None
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
        self.byte_tracker = state.get("byte_tracker") or self._new_byte_tracker()
