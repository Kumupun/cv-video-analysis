import os
import cv2
import json
import torch
from transnetv2_pytorch import TransNetV2

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[TransNetV2] Loading model onto {device}...")

try:
    cut_model = TransNetV2()
    cut_model.to(device) if hasattr(cut_model, "to") else None
    print("[TransNetV2] Model loaded successfully.")
except Exception as e:
    print(f"[TransNetV2] Error loading model: {e}")
    cut_model = None

CONFIDENCE_THRESHOLD = 0.5


def _run_inference(video_path: str):
    """
    predict_video повертає:
      video_frames     — вхідні кадри (не потрібні нам далі)
      single_frame_pred — per-frame confidence з "single-frame-per-transition" голови
      all_frame_pred    — per-frame confidence з "all-frames-per-transition" голови


    predictions_to_scenes() очікує numpy-масив (викликає .astype всередині),
    а модель повертає torch.Tensor
    """
    _video_frames, single_frame_pred, _all_frame_pred = cut_model.predict_video(video_path)
    if hasattr(single_frame_pred, "detach"):
        single_frame_pred = single_frame_pred.detach().cpu().numpy()
    return single_frame_pred


def process_video(video_path: str) -> list:
    """
    Receives an mp4 video path, processes it using TransNetV2,
    and returns a list of transitions formatted for JSON output.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    if cut_model is None:
        raise RuntimeError("TransNetV2 model is not loaded.")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV could not open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps != fps or fps <= 0:
        fps = 30.0
    cap.release()

    try:
        frame_scores = _run_inference(video_path)
    except Exception as e:
        raise RuntimeError(f"Inference failed: {e}")

    print(f"[TransNetV2] Got {len(frame_scores)} per-frame confidence scores "
          f"(sample: {frame_scores[:5] if len(frame_scores) else 'empty'})")

    scenes = cut_model.predictions_to_scenes(frame_scores, threshold=CONFIDENCE_THRESHOLD)

    results = []

    for i in range(len(scenes) - 1):

        shot1_end = int(scenes[i][1])
        shot2_start = int(scenes[i + 1][0])

        gap_scores = frame_scores[shot1_end:shot2_start + 1]
        confidence = float(max(gap_scores)) if len(gap_scores) else float(frame_scores[shot1_end])
        confidence = round(confidence, 4)

        if shot2_start - shot1_end <= 2:
            cut_frame = shot2_start
            results.append({
                "type": "hard_cut",
                "frame": cut_frame,
                "timestamp": round(cut_frame / fps, 2),
                "confidence": confidence,
            })

        else:
            results.append({
                "type": "gradual_transition",
                "start_frame": shot1_end,
                "end_frame": shot2_start,
                "start_timestamp": round(shot1_end / fps, 2),
                "end_timestamp": round(shot2_start / fps, 2),
                "confidence": confidence,
            })

    return results