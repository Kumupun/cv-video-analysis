import ray
import redis
import json
from autoshot_detector import AutoShotDetector

REDIS_HOST = 'localhost'
REDIS_PORT = 6379
GROUP_NAME = 'autoshot_cut_group'
CONSUMER_NAME = 'autoshot_worker_1'
STREAM_IN = 'video_chunks'
STREAM_OUT = 'scene_events'

redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)

def init_redis_group():
    try:
        redis_client.xgroup_create(STREAM_IN, GROUP_NAME, id='0', mkstream=True)
    except redis.exceptions.ResponseError as e:
        if "BUSYGROUP" not in str(e):
            raise

def run_worker():
    detector = AutoShotDetector()
    init_redis_group()
    print("AutoShot Cut Detection Worker запущено (Consumer Group mode).")
    
    while True:
        # Читаємо через Consumer Group
        messages = redis_client.xreadgroup(GROUP_NAME, CONSUMER_NAME, {STREAM_IN: '>'}, count=1, block=2000)
        if not messages:
            continue
            
        for stream, message_list in messages:
            for message_id, message in message_list:
                task_id = message.get('task_id')
                batch_id = message.get('batch_id')
                object_ref_id = message.get('object_ref')
                
                try:
                    frames = ray.get(ray.ObjectRef(object_ref_id.encode('utf-8')))
                    cuts = detector.detect_scenes(frames)
                    
                    # 1. Відправляємо JSON-результат у спільну чергу (для бекенду)
                    output_data = {
                        "task_id": task_id,
                        "batch_id": batch_id,
                        "cuts": json.dumps(cuts)
                    }
                    redis_client.xadd(STREAM_OUT, output_data)
                    
                    # 2. ОДНОЧАСНО записуємо в Hash для швидкої синхронізації з YOLO-World
                    hash_key = f"cuts:{task_id}:{batch_id}"
                    redis_client.hset(hash_key, "cuts", json.dumps(cuts))
                    redis_client.expire(hash_key, 3600) # Живе 1 годину, щоб не забити RAM
                    
                    # 3. ПІДТВЕРДЖЕННЯ УСПІШНОЇ ОБРОБКИ
                    redis_client.xack(STREAM_IN, GROUP_NAME, message_id)
                    print(f"[{task_id} | Batch {batch_id}] Знайдено склейок: {len(cuts)}. XACK відправлено.")
                    
                except Exception as e:
                    print(f"Помилка обробки {task_id} {batch_id}: {e}")
                    # Повідомлення залишається в PEL для обробки через DLQ

if __name__ == "__main__":
    ray.init(ignore_reinit_error=True)
    run_worker()
