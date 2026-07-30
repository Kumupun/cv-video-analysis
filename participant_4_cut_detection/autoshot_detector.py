import os
import sys
import urllib.request
import importlib
import numpy as np
import torch
import cv2

REPO_RAW_URL = "https://raw.githubusercontent.com/wentaozhu/AutoShot/master/"
REQUIRED_FILES = ["supernet_flattransf_3_8_8_8_13_12_0_16_60.py", "linear.py", "utils.py"]

def download_architecture_files(download_dir="autoshot_arch"):
    os.makedirs(download_dir, exist_ok=True)
    if download_dir not in sys.path:
        sys.path.insert(0, download_dir)
        
    for filename in REQUIRED_FILES:
        filepath = os.path.join(download_dir, filename)
        if not os.path.exists(filepath):
            url = REPO_RAW_URL + filename
            urllib.request.urlretrieve(url, filepath)
                
download_architecture_files()

module_name = "supernet_flattransf_3_8_8_8_13_12_0_16_60"
model_module = importlib.import_module(module_name)
TransNetV2Supernet = model_module.TransNetV2Supernet

class AutoShotDetector:
    def __init__(self, weights_path='weights_for_cut.pth'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.one_threshold = 0.55 
        self.frame_width = 48
        self.frame_height = 27
        self.model = TransNetV2Supernet().eval()
        
        try:
            checkpoint = torch.load(weights_path, map_location=self.device)
            state = checkpoint.get("net", checkpoint.get("state_dict", checkpoint))
            clean_state = {}
            for key, value in state.items():
                clean_key = str(key)
                for prefix in ("module.", "model.", "net."):
                    if clean_key.startswith(prefix):
                        clean_key = clean_key[len(prefix):]
                clean_state[clean_key] = value
            self.model.load_state_dict(clean_state, strict=False)
        except Exception:
            pass
            
        self.model = self.model.to(self.device)

    def preprocess(self, frames):
        resized_frames = [cv2.resize(f, (self.frame_width, self.frame_height)) for f in frames]
        batch_frames = np.stack(resized_frames, axis=0)
        return torch.from_numpy(batch_frames.transpose(3, 0, 1, 2))[None].float().to(self.device)

    @torch.inference_mode()
    def detect_scenes(self, frames):
        original_len = len(frames)
        pad_len = 100 - original_len if original_len < 100 else 0
        
        frames_to_process = np.concatenate([frames, [frames[-1]] * pad_len], axis=0) if pad_len > 0 else frames[:100]
        tensor = self.preprocess(frames_to_process)
        
        output = self.model(tensor)
        one_logits = output[0] if isinstance(output, tuple) else output
        one_prob = torch.sigmoid(one_logits[0]).cpu().numpy().reshape(-1)[:original_len]
        
        mask = one_prob > self.one_threshold
        padded = np.pad(mask.astype(np.int8), (1, 1), constant_values=0)
        changes = np.diff(padded)
        
        starts = np.flatnonzero(changes == 1)
        ends_ex = np.flatnonzero(changes == -1)
        
        cuts = []
        for start, end_ex in zip(starts, ends_ex):
            peak_idx = int(start + np.argmax(one_prob[start:end_ex]))
            cuts.append(peak_idx)
            
        return cuts
