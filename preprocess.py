import cv2
import numpy as np

def preprocess_video(video_path, target_fps = 25, target_size = (48, 27)):
    ''' Препроцессинг відео для TransNet.
        Відео ресемплиться до заданого FPS, змінюється розмір кадру до заданого, зберігаючи співвідношення сторін.
         Повертає:
            video_tensor: np.ndarray
            index_mapping: List[int] mapping від індексу кадру у відео до індексу у оригінальному відео
        !!! original_idx = int(resampled_idx * (orig_fps / target_fps)) !!!
            orig_fps: float
    '''
    cap = cv2.VideoCapture(video_path)
    
    orig_fps = cap.get(cv2.CAP_PROP_FPS)
    if orig_fps == 0 or np.isnan(orig_fps):
        orig_fps = 25.0 # Fallback для пошкоджених заголовків
        
    orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    time_ratio = orig_fps / target_fps
    
    frames_buffer = []
    index_mapping = [] 
    
    current_orig_frame = 0
    next_target_time = 0.0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        current_time = current_orig_frame / orig_fps

        if current_time >= next_target_time:
            target_aspect = target_size[0] / target_size[1]
            frame_aspect = orig_width / orig_height
            
            if frame_aspect > target_aspect:
                # Відео занадто широке
                new_h = int(orig_width / target_aspect)
                pad_h = (new_h - orig_height) // 2
                padded = cv2.copyMakeBorder(frame, pad_h, pad_h, 0, 0, cv2.BORDER_CONSTANT, value=(0,0,0))
            elif frame_aspect < target_aspect:
                # Відео занадто вузьке 
                new_w = int(orig_height * target_aspect)
                pad_w = (new_w - orig_width) // 2
                padded = cv2.copyMakeBorder(frame, 0, 0, pad_w, pad_w, cv2.BORDER_CONSTANT, value=(0,0,0))
            else:
                padded = frame

            resized = cv2.resize(padded, target_size, interpolation=cv2.INTER_AREA)
            rgb_frame = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            
            while current_time >= next_target_time:
                frames_buffer.append(rgb_frame)
                index_mapping.append(current_orig_frame)
            
                next_target_time += (1.0 / target_fps)
            
        current_orig_frame += 1
        
    cap.release()
    
    video_tensor = np.array(frames_buffer, dtype=np.uint8)

    return video_tensor, index_mapping, orig_fps


