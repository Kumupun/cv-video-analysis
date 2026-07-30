import ray
import redis
import json
import time
from yolo_world_tracker import YoloWorldTracker

REDIS_HOST = 'localhost'
REDIS_PORT = 6379
GROUP_NAME = 'yolo_tracking_group'
CONSUMER_NAME = 'yolo_worker_1'
STREAM_IN = 'video_chunks'
STREAM_OUT = 'tracking_results'

redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)

def init_redis_group():
    try:
        redis_client.xgroup_create(STREAM_IN, GROUP_NAME, id='0', mkstream=True)
    except redis.exceptions.ResponseError as e:
        if "BUSYGROUP" not in str(e):
            raise

def wait_for_cuts(task_id, batch_id, timeout=60):
    """
    Блокуюче очікування: чекаємо, поки AutoShot обробить цей батч 
    і запише індекси склейок у Redis Hash.
    """
    start_time = time.time()
    hash_key = f"cuts:{task_id}:{batch_id}"
    print(f"[{task_id} | Batch {batch_id}] Очікування результатів від AutoShot...")
    
    while time.time() - start_time < timeout:
        cuts_json = redis_client.hget(hash_key, "cuts")
        if cuts_json is not None:
            return json.loads(cuts_json)
        time.sleep(0.5) # Поллинг кожні півсекунди
        
    raise TimeoutError(f"Не дочекалися склейок для {task_id} batch {batch_id}")

def run_worker():
    tracker = YoloWorldTracker()
    init_redis_group()
    print("YOLO-World Tracker Worker запущено (Consumer Group mode).")
    
    while True:
        # Читаємо завдання з групи з підтвердженням (XACK)
        messages = redis_client.xreadgroup(GROUP_NAME, CONSUMER_NAME, {STREAM_IN: '>'}, count=1, block=2000)
        if not messages:
            continue
            
        for stream, message_list in messages:
            for message_id, message in message_list:
                task_id = message.get('task_id')
                batch_id = message.get('batch_id')
                object_ref_id = message.get('object_ref')
                start_frame = int(message.get('start_frame', 0))
                
                try:
                    # 1. Синхронізація: чекаємо склейки від Worker 4
                    cuts = wait_for_cuts(task_id, batch_id)
                    
                    # 2. Отримуємо кадри з пам'яті (Zero-Copy)
                    frames = ray.get(ray.ObjectRef(object_ref_id.encode('utf-8')))
                    
                    # 3. Обробляємо сцени ізольовано
                    batch_results = []
                    current_start = 0
                    
                    # Додаємо кінець батчу як останню "склейку" для зручності циклу
                    scene_boundaries = cuts + [len(frames)]
                    
                    for cut_idx in scene_boundaries:
                        if cut_idx <= current_start:
                            continue
                            
                        # Вирізаємо чисту сцену
                        scene_frames = frames[current_start:cut_idx]
                        
                        # Трекаємо сцену (всередині викликається reset для ByteTrack)
                        scene_res = tracker.process_scene(
                            scene_frames, 
                            global_offset=start_frame + current_start
                        )
                        batch_results.extend(scene_res)
                        current_start = cut_idx
                        
                    # 4. Відправляємо результат бекенду за новим контрактом
                    output_data = {
                        "task_id": task_id,
                        "batch_id": batch_id,
                        "frames": json.dumps(batch_results)
                    }
                    redis_client.xadd(STREAM_OUT, output_data)
                    
                    # 5. ПІДТВЕРДЖЕННЯ УСПІШНОЇ ОБРОБКИ
                    redis_client.xack(STREAM_IN, GROUP_NAME, message_id)
                    print(f"[{task_id} | Batch {batch_id}] Трекінг завершено. XACK відправлено.")
                    
                except TimeoutError as e:
                    print(f"Помилка синхронізації: {e}. Повідомлення залишається в PEL для повторної обробки.")
                    # Не робимо XACK, Бекенд перехопить його через DLQ (XPENDING)
                except Exception as e:
                    print(f"Помилка обробки {task_id} {batch_id}: {e}")

if __name__ == "__main__":
    ray.init(ignore_reinit_error=True)
    run_worker()
