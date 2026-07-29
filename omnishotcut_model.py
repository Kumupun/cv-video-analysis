import os
import cv2
import json
import torch
import omnishotcut

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[OmniShotCut] Loading model onto {device}...")

try:
    cut_model = omnishotcut.load("uva-cv-lab/OmniShotCut", filename="OmniShotCut_ckpt.pth")
    print("[OmniShotCut] Model loaded successfully.")
except Exception as e:
    print(f"[OmniShotCut] Error loading model: {e}")
    cut_model = None


def _run_inference(video_path: str):
    try:
        return cut_model.inference(video_path, mode="default", device=device)
    except TypeError:
        return cut_model.inference(video_path, mode="default")


def process_video(video_path: str) -> list:
    """
    Receives an mp4 video path, processes it using OmniShotCut,
    and returns a list of transitions formatted for JSON output.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    if cut_model is None:
        raise RuntimeError("OmniShotCut model is not loaded.")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV could not open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps != fps:  # покриває і 0, і NaN
        fps = 30.0  # Fallback
    cap.release()

    try:
        ranges, intra_labels, inter_labels = _run_inference(video_path)
    except Exception as e:
        raise RuntimeError(f"Inference failed: {e}")

    results = []

    for i in range(len(ranges) - 1):
        shot1_end = ranges[i][1]
        shot2_start = ranges[i + 1][0]

        confidence = None

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