import torch
from ultralytics import YOLOWorld

class YoloWorldTracker:
    def __init__(self, model_id='yolov8s-world.pt', classes=["person", "vehicle"]):
        self.model = YOLOWorld(model_id)
        self.model.set_classes(classes)
        
        # Примусово переносимо модель на GPU
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)
    def process_scene(self, frames, global_offset):
        """
        Обробляє одну чисту сцену (між склейками) і свідомо скидає 
        внутрішній стан ByteTrack, щоб ID об'єктів починалися заново.
        """
        # Форсуємо скидання стану трекера в Ultralytics
        if hasattr(self.model, 'predictor') and self.model.predictor:
            if hasattr(self.model.predictor, 'trackers'):
                self.model.predictor.trackers = [] 
                
        results = self.model.track(frames, persist=True, tracker="bytetrack.yaml", verbose=False)
        
        scene_results = []
        for frame_idx, r in enumerate(results):
            frame_data = {"frame_id": global_offset + frame_idx, "objects": []}
            if r.boxes is not None and r.boxes.id is not None:
                boxes = r.boxes.xywh.cpu().numpy()
                track_ids = r.boxes.id.int().cpu().numpy()
                confidences = r.boxes.conf.cpu().numpy()
                class_ids = r.boxes.cls.int().cpu().numpy()
                
                for box, track_id, conf, cls_id in zip(boxes, track_ids, confidences, class_ids):
                    frame_data["objects"].append({
                        "track_id": int(track_id),
                        "class": self.model.names[cls_id],
                        "bbox": [float(x) for x in box],
                        "confidence": float(conf)
                    })
            scene_results.append(frame_data)
        return scene_results
