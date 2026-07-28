import json
import cv2
import os

# ==========================================
# ЗАХАРДКОДЖЕНІ НАЛАШТУВАННЯ
# ==========================================
TARGET_FPS = 30.0                           # Бажана частота кадрів
VIDEOS_DIR = "D:/FourthSemestr/data_for_anotation/data/ClipShots/videos/only_gradual"     # Папка, де фізично лежать .mp4 файли
INPUT_JSON = "D:/FourthSemestr/data_for_anotation/data/ClipShots/annotations/fps_scale_test.json"            # Вхідний файл з оригінальною анотацією
OUTPUT_JSON = "D:/FourthSemestr/data_for_anotation/data/ClipShots/annotations/annotations_30fps.json"      # Файл для збереження результату

def get_video_fps(video_path):
    """
    Автоматично отримує оригінальний FPS відеофайлу за допомогою OpenCV.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Файл відео не знайдено: {video_path}")
    
    # Відкриваємо відеофайл та зчитуємо метадані
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    
    return fps

def normalize_annotations():
    print(f"[*] Запуск нормалізації анотацій до {TARGET_FPS} FPS...")
    
    # Завантажуємо оригінальний JSON
    with open(INPUT_JSON, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    new_data = {}
    processed_count = 0
    
    for video_name, annot in data.items():
        video_path = os.path.join(VIDEOS_DIR, video_name)
        
        # 1. Автоматично отримуємо оригінальний FPS
        try:
            orig_fps = get_video_fps(video_path)
            if orig_fps <= 0:
                print(f"[-] Помилка читання FPS для {video_name}. Пропускаємо.")
                continue
        except FileNotFoundError as e:
            print(f"[-] {e}. Пропускаємо.")
            continue
            
        # 2. Вираховуємо коефіцієнт зміни часу
        ratio = TARGET_FPS / orig_fps
        
        orig_transitions = annot.get("transitions", [])
        orig_frame_num = annot.get("frame_num", 0)
        
        # Перераховуємо загальну кількість кадрів для нового відео
        new_frame_num = round(orig_frame_num * ratio)
        new_transitions = []
        
        # 3. Перераховуємо кожну склейку
        for start, end in orig_transitions:
            # Перевіряємо, чи це жорстка склейка (довжина рівно 1 кадр)
            is_hard_cut = (end - start == 1)
            
            # Масштабуємо стартовий кадр у новий часовий вимір
            new_start = round(start * ratio)
            
            if is_hard_cut:
                # ГАРАНТІЯ ЖОРСТКОЇ СКЛЕЙКИ:
                # Навіть якщо ratio > 1 (наприклад, 15 FPS -> 30 FPS), 
                # примусово робимо різницю рівно 1 кадр, щоб модель не бачила плавного переходу.
                new_end = new_start + 1
            else:
                # Для плавних переходів (gradual) масштабуємо кінець природним чином
                new_end = round(end * ratio)
                
                # Захист при екстремальному даунскейлі (напр. 60 FPS -> 10 FPS)
                if new_end <= new_start:
                    new_end = new_start + 2
            
            # Захист від виходу індексів за межі нового відеоряду
            new_start = min(new_start, new_frame_num - 1)
            new_end = min(new_end, new_frame_num)
            
            new_transitions.append([new_start, new_end])
            
        # Зберігаємо оновлені дані для поточного відео
        new_data[video_name] = {
            "transitions": new_transitions,
            "frame_num": new_frame_num
        }
        
        processed_count += 1
        print(f"[+] {video_name}: {orig_fps:.2f} FPS -> {TARGET_FPS} FPS | Кадрів: {orig_frame_num} -> {new_frame_num}")

    # Зберігаємо фінальний результат у новий JSON
    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        # Використовуємо separators для компактного форматування без зайвих пробілів
        json.dump(new_data, f, separators=(',', ': '))
        
    print(f"\n[*] Успішно завершено! Оброблено відеофайлів: {processed_count}")
    print(f"[*] Оновлені анотації збережено у: {OUTPUT_JSON}")

if __name__ == "__main__":
    normalize_annotations()