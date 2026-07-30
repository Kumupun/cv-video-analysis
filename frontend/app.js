(function(){
  const DEFAULT_FPS = 30; // фолбек лише якщо бекенд не прислав fps.
  const dropzone = document.getElementById('dropzone');
  const fileInput = document.getElementById('file-input');
  const uploadView = document.getElementById('upload-view');
  const processingView = document.getElementById('processing-view');
  const resultView = document.getElementById('result-view');
  const statusSub = document.getElementById('status-sub');
  const resetBtn = document.getElementById('reset-btn');
  const video = document.getElementById('video');
  const overlay = document.getElementById('overlay');
  const videoWrap = document.getElementById('video-wrap');
  const timelineCanvas = document.getElementById('timeline-canvas');
  const consoleEl = document.getElementById('console');
  const stageTrack = document.getElementById('stage-track');

  let pipeline = null; // {scenes, cuts, tracks, duration}
  let visibleClasses = null; // Set<string> — які класи показувати в overlay (null = всі)
  let rafId = null;
  let currentTaskId = null;

  // ---------- Вибір класів детекції (до обробки) ----------
  // Базовий словник COCO-80 (те, на чому YOLO26 навчена за замовчуванням).
  const COCO_CLASSES = [
    'person','bicycle','car','motorcycle','airplane','bus','train','truck','boat',
    'traffic light','fire hydrant','stop sign','parking meter','bench','bird','cat',
    'dog','horse','sheep','cow','elephant','bear','zebra','giraffe','backpack',
    'umbrella','handbag','tie','suitcase','frisbee','skis','snowboard','sports ball',
    'kite','baseball bat','baseball glove','skateboard','surfboard','tennis racket',
    'bottle','wine glass','cup','fork','knife','spoon','bowl','banana','apple',
    'sandwich','orange','broccoli','carrot','hot dog','pizza','donut','cake','chair',
    'couch','potted plant','bed','dining table','toilet','tv','laptop','mouse',
    'remote','keyboard','cell phone','microwave','oven','toaster','sink',
    'refrigerator','book','clock','vase','scissors','teddy bear','hair drier',
    'toothbrush'
  ];

  // Додаткові класи з донавчання моделі (дрони, військова техніка).
  // ВАЖЛИВО: точні назви/написання (case, підкреслення) звірте з тим, як
  // саме названі класи у ваших training labels — keep_classes на бекенді
  // фільтрує по точному співпадінню рядка.
  const CUSTOM_CLASSES = [
    'drone', 'fpv_drone', 'tank', 'ifv', 'apc', 'artillery',
    'military_truck', 'helicopter', 'military_aircraft', 'anti_aircraft_system', 'mlrs'
  ];

  const CLASS_GROUPS = [
    { label: 'COCO-80', classes: COCO_CLASSES },
    { label: 'Донавчені класи', classes: CUSTOM_CLASSES },
  ];
  const CLASS_OPTIONS = [...COCO_CLASSES, ...CUSTOM_CLASSES];

  let selectedClasses = new Set(CLASS_OPTIONS); // за замовчуванням шукаємо все

  const classSelectContainer = document.getElementById('class-select-options');
  const classSearchInput = document.getElementById('class-search');

  classSelectContainer.innerHTML = CLASS_GROUPS.map(group => `
    <div class="class-group">
      <div class="class-group-label">${group.label}</div>
      <div class="class-group-items">
        ${group.classes.map(cls => `
          <label class="class-option" data-cls-name="${cls}">
            <input type="checkbox" value="${cls}" checked> ${cls}
          </label>`).join('')}
      </div>
    </div>`).join('');

  classSelectContainer.querySelectorAll('input[type="checkbox"]').forEach(cb => {
    cb.addEventListener('change', () => {
      if (cb.checked) selectedClasses.add(cb.value);
      else selectedClasses.delete(cb.value);
    });
  });

  classSearchInput.addEventListener('input', () => {
    const q = classSearchInput.value.trim().toLowerCase();
    classSelectContainer.querySelectorAll('.class-option').forEach(label => {
      const match = label.dataset.clsName.includes(q);
      label.classList.toggle('hidden-by-search', !match);
    });
  });

  document.getElementById('classes-select-all').addEventListener('click', () => {
    classSelectContainer.querySelectorAll('input[type="checkbox"]').forEach(cb => {
      cb.checked = true; selectedClasses.add(cb.value);
    });
  });
  document.getElementById('classes-select-none').addEventListener('click', () => {
    classSelectContainer.querySelectorAll('input[type="checkbox"]').forEach(cb => {
      cb.checked = false; selectedClasses.delete(cb.value);
    });
  });

  // DEMO_MODE = true  -> завжди локальна симуляція (generatePipeline), без мережі.
  // DEMO_MODE = false -> реальні виклики /process, /status, /results.
  //
  // API_BASE — лише префікс URL для цих викликів:
  //   '' (порожньо)            — same-origin, коли FastAPI роздає і фронтенд, і API
  //   'http://localhost:8000'  — якщо бекенд на іншому хості/порті (тоді потрібен CORS)
  const DEMO_MODE = true;
  const API_BASE = '';

  /**
   * ---------------------------------------------------------------------
   * normalizeBackendResponse — перетворює агрегований JSON бекенду
   * (кадри для склейок, пікселі + confidence для bbox, розбиття на батчі)
   * у внутрішній формат { duration, scenes, cuts, tracks }.
  
   */
  const CONFIG = {
    // 'xyxy' -> [x1,y1,x2,y2] (кути) | 'xywh' -> [x,y,ширина,висота]
    bboxFormat: 'xyxy',
    fields: {
      videoMeta: 'video',               // raw.video.{duration_sec, fps, width, height}
      batches: 'batches',                // raw.batches[]
      batchScenes: 'scene_boundaries',   // batch.scene_boundaries[]
      batchTracks: 'tracks',             // batch.tracks[]
      trackId: 'track_id',
      trackClass: 'class',
      detections: 'detections',          // track.detections[]
      frame: 'frame',
      bbox: 'bbox',
      confidence: 'confidence',
      cutStartFrame: 'start_frame',
      cutEndFrame: 'end_frame',
      cutType: 'type',                   // 'hard_cut' | 'gradual_transition'
    }
  };

  function normalizeBackendResponse(raw){
    const F = CONFIG.fields;
    const meta = raw[F.videoMeta] || {};
    const fps = meta.fps || DEFAULT_FPS;
    const width = meta.width || video.videoWidth || 1920;
    const height = meta.height || video.videoHeight || 1080;
    const duration = meta.duration_sec || video.duration;

    const batches = raw[F.batches] || [];

    // 1. Зібрати всі межі склейок з усіх батчів в один список (дедуп)
    const rawCuts = [];
    batches.forEach(batch => {
      (batch[F.batchScenes] || []).forEach(c => rawCuts.push({
        startFrame: c[F.cutStartFrame], endFrame: c[F.cutEndFrame], type: c[F.cutType]
      }));
    });
    const seen = new Set();
    const cuts = rawCuts.filter(c => {
      const key = `${c.startFrame}-${c.endFrame}-${c.type}`;
      if (seen.has(key)) return false;
      seen.add(key); return true;
    }).sort((a,b) => a.startFrame - b.startFrame).map(c => ({
      time: c.startFrame / fps, start: c.startFrame / fps, end: c.endFrame / fps, type: c.type
    }));

    // 2. Відновити неперервні сцени з меж склейок (потрібно для sceneIndex)
    const boundaryFrames = [0, ...cuts.map(c => c.start), duration].sort((a,b)=>a-b);
    const scenes = [];
    for (let i = 0; i < boundaryFrames.length - 1; i++){
      const start = boundaryFrames[i], end = boundaryFrames[i+1];
      if (end > start) scenes.push({ start, end, index: scenes.length });
    }
    const sceneIndexAtTime = t => {
      const s = scenes.find(s => t >= s.start && t <= s.end);
      return s ? s.index : Math.max(0, scenes.length - 1);
    };

    // 3. Зібрати треки з усіх батчів, злити детекції одного track_id
    const trackMap = new Map();
    batches.forEach(batch => {
      (batch[F.batchTracks] || []).forEach(t => {
        const id = t[F.trackId];
        if (!trackMap.has(id)) trackMap.set(id, { id, cls: t[F.trackClass], detections: [] });
        const entry = trackMap.get(id);
        (t[F.detections] || []).forEach(d => entry.detections.push({
          frame: d[F.frame], bbox: d[F.bbox], confidence: d[F.confidence]
        }));
      });
    });

    const tracks = Array.from(trackMap.values()).map(t => {
      t.detections.sort((a,b) => a.frame - b.frame);
      const keyframes = t.detections.map(d => {
        const [a,b,c,dd] = d.bbox;
        const px = CONFIG.bboxFormat === 'xyxy'
          ? { x: a, y: b, w: c - a, h: dd - b }
          : { x: a, y: b, w: c, h: dd };
        return {
          t: d.frame / fps,
          x: px.x / width, y: px.y / height, w: px.w / width, h: px.h / height,
          confidence: d.confidence
        };
      });
      const firstT = keyframes[0]?.t ?? 0;
      return {
        id: t.id, cls: t.cls,
        sceneIndex: sceneIndexAtTime(firstT),
        confidence: keyframes[0]?.confidence ?? 0.8,
        keyframes
      };
    });

    // Демо-заглушка 
    return { duration, fps: DEFAULT_FPS, scenes, cuts, tracks };
  }

  // ---------- Upload handling ----------
  dropzone.addEventListener('click', () => fileInput.click());
  ['dragenter','dragover'].forEach(ev => dropzone.addEventListener(ev, e => {
    e.preventDefault(); dropzone.classList.add('drag');
  }));
  ['dragleave','drop'].forEach(ev => dropzone.addEventListener(ev, e => {
    e.preventDefault(); dropzone.classList.remove('drag');
  }));
  dropzone.addEventListener('drop', e => {
    const f = e.dataTransfer.files[0];
    if (f) handleFile(f);
  });
  fileInput.addEventListener('change', e => {
    const f = e.target.files[0];
    if (f) handleFile(f);
  });

  function handleFile(file){
    const url = URL.createObjectURL(file);
    video.src = url;
    video.onloadedmetadata = async () => {
      uploadView.hidden = true;
      processingView.hidden = false;
      statusSub.textContent = 'обробка…';

      // ВАЖЛИВО: video.duration часто = Infinity одразу після loadedmetadata
      // для webm/деяких mp4-контейнерів (відомий баг/особливість Chrome) —
      // `video.duration || 12` це НЕ ловить, бо Infinity є truthy. Якщо таке
      // значення дійде до drawTimeline() (`for (let s=0; s<=dur; s++)`) —
      // це нескінченний цикл, який вішає вкладку. Тому дістаємо реальну
      // тривалість явно, з фолбеком на 12с лише якщо і після цього нічого
      // не вдалось отримати.
      const duration = await resolveVideoDuration(video);

      if (!DEMO_MODE) {
        startRealPipeline(file, duration);
      } else {
        // демо-режим: немає підключеного бекенду — симулюємо пайплайн локально
        runProcessing(duration);
      }
    };
  }

  function resolveVideoDuration(videoEl, fallback = 12){
    return new Promise(resolve => {
      if (Number.isFinite(videoEl.duration) && videoEl.duration > 0){
        resolve(videoEl.duration);
        return;
      }
      // Стандартний обхід Chrome-бага: перемотка у величезний timestamp
      // змушує браузер порахувати реальну тривалість.
      const onTimeUpdate = () => {
        videoEl.removeEventListener('timeupdate', onTimeUpdate);
        videoEl.currentTime = 0;
        resolve(Number.isFinite(videoEl.duration) && videoEl.duration > 0 ? videoEl.duration : fallback);
      };
      videoEl.addEventListener('timeupdate', onTimeUpdate);
      videoEl.currentTime = 1e101;

      // Захист про всяк випадок: якщо навіть цей трюк не спрацює (деякі
      // браузери/кодеки), не залишаємось висіти — фолбек через 2с.
      setTimeout(() => {
        videoEl.removeEventListener('timeupdate', onTimeUpdate);
        resolve(Number.isFinite(videoEl.duration) && videoEl.duration > 0 ? videoEl.duration : fallback);
      }, 2000);
    });
  }

  async function startRealPipeline(file, fallbackDuration){
    try {
      const form = new FormData();
      form.append('file', file);
      form.append('classes', JSON.stringify(Array.from(selectedClasses))); // -> keep_classes на бекенді
      const res = await fetch(`${API_BASE}/process`, { method:'POST', body: form });
      if (!res.ok) throw new Error(`POST /process → ${res.status}`);
      const { task_id } = await res.json();
      currentTaskId = task_id;
      pollStatus(task_id);
    } catch (err){
      console.warn('[backend недоступний, переходжу в демо-режим]', err);
      runProcessing(fallbackDuration);
    }
  }

  async function pollStatus(taskId){
    try {
      const res = await fetch(`${API_BASE}/status/${taskId}`);
      const data = await res.json(); // очікується { stage, progress, status }
      updateStageUI(data.stage, data.progress);

      if (data.status === 'done') {
        const resultsRes = await fetch(`${API_BASE}/results/${taskId}`);
        pipeline = normalizeBackendResponse(await resultsRes.json());
        processingView.hidden = true;
        resultView.hidden = false;
        resetBtn.style.display = 'inline-block';
        statusSub.textContent = 'готово';
        setupResultView();
      } else if (data.status === 'error') {
        statusSub.textContent = 'помилка обробки';
        consoleEl.innerHTML += `<div class="ln" style="color:var(--cut-hard)">$ ${data.message || 'unknown error'}</div>`;
      } else {
        setTimeout(() => pollStatus(taskId), 1000);
      }
    } catch (err){
      console.warn('[втрачено зв\'язок з бекендом]', err);
      setTimeout(() => pollStatus(taskId), 2000);
    }
  }

  // Прив'язує stage-key реального бекенду до DOM-нод, створених runProcessing()
  function updateStageUI(stageKey, progress){
    const idx = STAGES.findIndex(s => s.key === stageKey);
    if (idx === -1) return;
    if (!stageTrack.children.length) {
      stageTrack.innerHTML = STAGES.map(s => `
        <div class="stage" data-key="${s.key}">
          <div class="stage-line"></div>
          <div class="node">●</div>
          <div class="label">${s.label}</div>
          <div class="sublabel">${s.sub}${progress != null ? '' : ''}</div>
        </div>`).join('');
    }
    STAGES.forEach((s, i) => {
      const el = stageTrack.querySelector(`[data-key="${s.key}"]`);
      el.classList.remove('active','done');
      if (i < idx) el.classList.add('done');
      else if (i === idx) el.classList.add('active');
    });
  }

  resetBtn.addEventListener('click', () => {
    cancelAnimationFrame(rafId);
    video.pause();
    video.removeAttribute('src');
    video.load();
    resultView.hidden = true;
    processingView.hidden = true;
    uploadView.hidden = false;
    resetBtn.style.display = 'none';
    statusSub.textContent = 'очікування завантаження';
    fileInput.value = '';
  });

  // ---------- Stage / console simulation ----------
  const STAGES = [
    { key:'decode', label:'Decode', sub:'Decord / NVDEC', logs:[
      'POST /process → task_id=8f3a21c',
      '<span class="tag">Decord</span>: reading frame batch [0..64]',
      'NVDEC: hardware decode → PyTorch tensor',
      'ray.put(batch) → ObjectRef(0x7f2c…)'
    ]},
    { key:'cuts', label:'Cut Detection', sub:'TransNetV2', logs:[
      'XREAD video_chunks → task received',
      '<span class="tag">TransNetV2</span>: temporal window [0..100], BF16 autocast',
      'scipy.signal.find_peaks → 1D NMS',
      '<span class="warn">scene_events</span>: hard_cut @ frame 142',
      'scene_events: gradual_transition @ [388..410]'
    ]},
    { key:'tracking', label:'Tracking', sub:'Co-DETR + ByteTrack', logs:[
      'XREAD scene_events + video_chunks',
      '<span class="tag">Co-DETR</span>: BF16 inference on H200 Tensor Cores',
      'ByteTrack: Kalman filter update, Hungarian match',
      'reset Track IDs @ scene boundary',
      'tracking_results: XADD batch complete'
    ]},
    { key:'agg', label:'Aggregation', sub:'Backend', logs:[
      'Aggregator: collecting metadata from workers',
      'writing final JSON → PostgreSQL',
      'ray.internal.free() → ObjectRef released',
      '200 OK → /results/8f3a21c ready'
    ]}
  ];

  function runProcessing(duration){
    stageTrack.innerHTML = STAGES.map(s => `
      <div class="stage" data-key="${s.key}">
        <div class="stage-line"></div>
        <div class="node">●</div>
        <div class="label">${s.label}</div>
        <div class="sublabel">${s.sub}</div>
      </div>`).join('');
    consoleEl.innerHTML = '';

    let stageIdx = 0;
    let lineDelay = 0;

    function nextStage(){
      if (stageIdx > 0){
        const prev = stageTrack.querySelector(`[data-key="${STAGES[stageIdx-1].key}"]`);
        prev.classList.remove('active'); prev.classList.add('done');
      }
      if (stageIdx >= STAGES.length){
        setTimeout(() => finishProcessing(duration), 400);
        return;
      }
      const cur = STAGES[stageIdx];
      const el = stageTrack.querySelector(`[data-key="${cur.key}"]`);
      el.classList.add('active');

      cur.logs.forEach((line, i) => {
        setTimeout(() => {
          const ln = document.createElement('div');
          ln.className = 'ln';
          ln.innerHTML = '<span style="color:var(--text-dimmer)">$</span> ' + line;
          consoleEl.appendChild(ln);
          consoleEl.scrollTop = consoleEl.scrollHeight;
        }, i * 260);
      });

      const stageDuration = cur.logs.length * 260 + 500;
      setTimeout(() => { stageIdx++; nextStage(); }, stageDuration);
    }
    nextStage();
  }

  function finishProcessing(duration){
    pipeline = generatePipeline(duration);
    processingView.hidden = true;
    resultView.hidden = false;
    resetBtn.style.display = 'inline-block';
    statusSub.textContent = 'готово';
    setupResultView();
  }

  // ---------- Заглушка замість пайплайна (детермінована, без Math.random) ----------

  // Коли буде реальний бекенд aункція просто не викликається,
  // замість неї piepline = normalizeBackendResponse(await fetch(...)).

  const STUB_TEMPLATE = {
    // межі сцен як частки тривалості (0..1)
    sceneBoundaries: [0, 0.18, 0.34, 0.52, 0.71, 0.85, 1],
    // тип склейки на кожній внутрішній межі (довжина = sceneBoundaries.length - 2,
    // бо перша (0) і остання (1) межі відео — не склейки)
    cutTypes: ['hard_cut', 'gradual_transition', 'hard_cut', 'hard_cut', 'gradual_transition'],
    // фіксовані треки на сцену: клас + стартова/кінцева позиція (частки кадру)
    tracksPerScene: [
      [{ cls:'person',  from:{x:0.10,y:0.40}, to:{x:0.30,y:0.35}, w:0.14, h:0.30 }],
      [{ cls:'car', from:{x:0.05,y:0.55}, to:{x:0.45,y:0.50}, w:0.28, h:0.16 },
       { cls:'person',  from:{x:0.55,y:0.30}, to:{x:0.60,y:0.45}, w:0.12, h:0.28 }],
      [{ cls:'person',  from:{x:0.20,y:0.20}, to:{x:0.20,y:0.60}, w:0.13, h:0.32 }],
      [{ cls:'car', from:{x:0.60,y:0.45}, to:{x:0.10,y:0.48}, w:0.30, h:0.17 }],
      [{ cls:'person',  from:{x:0.40,y:0.35}, to:{x:0.42,y:0.36}, w:0.15, h:0.31 },
       { cls:'person',  from:{x:0.65,y:0.40}, to:{x:0.50,y:0.42}, w:0.14, h:0.29 }],
      [{ cls:'car', from:{x:0.15,y:0.50}, to:{x:0.55,y:0.47}, w:0.29, h:0.18 }],
    ],
  };

  function generatePipeline(duration){
    const boundaries = STUB_TEMPLATE.sceneBoundaries.map(f => f * duration);

    const scenes = [];
    for (let i=0;i<boundaries.length-1;i++){
      scenes.push({ start:boundaries[i], end:boundaries[i+1], index:scenes.length });
    }

    const cuts = [];
    for (let i=1;i<boundaries.length-1;i++){
      const boundary = boundaries[i];
      const type = STUB_TEMPLATE.cutTypes[i-1] || 'hard_cut';
      const span = type === 'hard_cut' ? 0 : 0.6; // фіксована ширина gradual-зони, сек
      cuts.push({ time:boundary, type, start:boundary-span/2, end:boundary+span/2 });
    }

    const tracks = [];
    scenes.forEach(scene => {
      const specs = (STUB_TEMPLATE.tracksPerScene[scene.index] || [])
        .filter(spec => selectedClasses.has(spec.cls)); // той самий фільтр, що keep_classes на бекенді
      specs.forEach((spec, i) => {
        const keyframes = [0, 0.5, 1].map(f => ({
          t: scene.start + (scene.end - scene.start) * f,
          x: spec.from.x + (spec.to.x - spec.from.x) * f,
          y: spec.from.y + (spec.to.y - spec.from.y) * f,
          w: spec.w, h: spec.h,
          confidence: 0.9, // фіксоване значення — не випадкове
        }));
        tracks.push({
          id: `${scene.index}_${i}`,
          cls: spec.cls,
          sceneIndex: scene.index,
          confidence: keyframes[0].confidence,
          keyframes,
        });
      });
    });

    return { duration, fps, scenes, cuts, tracks };
  }

  function detectionsAtTime(t){
    const scene = pipeline.scenes.find(s => t >= s.start && t <= s.end) ||
                  pipeline.scenes[pipeline.scenes.length-1];
    if (!scene) return [];
    const active = pipeline.tracks.filter(tr => tr.sceneIndex === scene.index);
    return active.map(tr => {
      const kfs = tr.keyframes;
      let a = kfs[0], b = kfs[kfs.length-1];
      for (let i=0;i<kfs.length-1;i++){
        if (t >= kfs[i].t && t <= kfs[i+1].t){ a = kfs[i]; b = kfs[i+1]; break; }
      }
      const span = (b.t - a.t) || 1;
      const f = Math.min(1, Math.max(0, (t - a.t) / span));
      const confA = a.confidence ?? tr.confidence, confB = b.confidence ?? tr.confidence;
      return {
        id: tr.id, cls: tr.cls, confidence: confA + (confB-confA)*f,
        x: a.x + (b.x-a.x)*f, y: a.y + (b.y-a.y)*f,
        w: a.w + (b.w-a.w)*f, h: a.h + (b.h-a.h)*f
      };
    });
  }

  // Кураторські кольори для найчастіших класів; решта з 80 —
  // детермінований хеш → HSL, щоб той самий клас завжди мав той самий колір,
  // але не довелось вручну прописувати 80 значень.
  const CURATED_CLASS_COLORS = {
    person: '#39e6b0',
    car: '#5b9dff',
    truck: '#5b9dff',
    bus: '#5b9dff',
    motorcycle: '#5b9dff',
    bicycle: '#5b9dff',
    // Донавчені класи — навмисно "тривожна" червоно-оранжева гама,
    // щоб вирізнялись на overlay серед побутових об'єктів COCO.
    drone: '#ff4f64',
    fpv_drone: '#ff4f64',
    tank: '#ff8a3d',
    ifv: '#ff8a3d',
    apc: '#ff8a3d',
    artillery: '#ff8a3d',
    military_truck: '#ff8a3d',
    helicopter: '#ff4f64',
    military_aircraft: '#ff4f64',
    anti_aircraft_system: '#ff8a3d',
    mlrs: '#ff8a3d',
  };
  function classColor(cls){
    if (CURATED_CLASS_COLORS[cls]) return CURATED_CLASS_COLORS[cls];
    let hash = 0;
    for (let i = 0; i < cls.length; i++) hash = (hash * 31 + cls.charCodeAt(i)) >>> 0;
    const hue = hash % 360;
    return `hsl(${hue}, 70%, 60%)`;
  }

  // ---------- Result view setup ----------
  function setupResultView(){
    document.getElementById('stat-duration').textContent = formatTime(pipeline.duration);
    document.getElementById('stat-fps').textContent = pipeline.fps ?? '—';
    document.getElementById('stat-tracks').textContent = pipeline.tracks.length;
    document.getElementById('scene-count').textContent = pipeline.cuts.length;

    const classCounts = {};
    pipeline.tracks.forEach(t => classCounts[t.cls] = (classCounts[t.cls]||0)+1);

    // шукати перед обробкою (selectedClasses); тут просто ховаємо/показуємо
    // вже отримані детекції в overlay.
    visibleClasses = new Set(Object.keys(classCounts));

    document.getElementById('class-legend').innerHTML = Object.entries(classCounts).map(([cls,count]) => `
      <div class="class-row" data-cls="${cls}">
        <label class="name">
          <input type="checkbox" checked data-cls-toggle="${cls}">
          <span class="swatch" style="background:${classColor(cls)}"></span>${cls}
        </label>
        <span class="count">${count}</span>
      </div>`).join('');

    document.querySelectorAll('[data-cls-toggle]').forEach(cb => {
      cb.addEventListener('change', () => {
        const cls = cb.dataset.clsToggle;
        if (cb.checked) visibleClasses.add(cls); else visibleClasses.delete(cls);
        cb.closest('.class-row').classList.toggle('dim', !cb.checked);
        drawOverlay(); // миттєвий фідбек, навіть якщо відео на паузі
      });
    });

    document.getElementById('scene-list').innerHTML = pipeline.cuts.map(c => `
      <div class="scene-row" data-t="${c.time}">
        <span>${formatTime(c.time)}</span>
        <span class="badge ${c.type==='hard_cut'?'hard':'gradual'}">${c.type}</span>
      </div>`).join('');
    document.querySelectorAll('.scene-row').forEach(row => {
      row.addEventListener('click', () => { video.currentTime = parseFloat(row.dataset.t); });
    });

    resizeCanvases();
    drawTimeline();
    drawOverlay();

    initPlaybackListenersOnce();

    video.controls = true;
    video.play().catch(()=>{});
  }

  // ВИПРАВЛЕНО: раніше ці 5 listener-ів навішувались всередині
  // setupResultView() — а вона викликається на КОЖНЕ оброблене відео.
  // video/window — persistent-об'єкти (не пересоздаються між відео,
  // міняється лише video.src), тому з кожним новим відео/reset додавався
  // ще один повний набір listener-ів поверх старих. Через кілька відео
  // за сесію timeupdate/resize/seeked виконувались N разів на кожну подію
  // (N = скільки відео вже обробили) — звідси прогресуюче гальмування
  // і "video freezing" при повторному використанні.
  let _playbackListenersAttached = false;
  function initPlaybackListenersOnce(){
    if (_playbackListenersAttached) return;
    _playbackListenersAttached = true;

    video.addEventListener('play', () => { cancelAnimationFrame(rafId); loop(); });
    video.addEventListener('pause', () => cancelAnimationFrame(rafId));
    video.addEventListener('seeked', () => { drawOverlay(); drawTimeline(); });
    video.addEventListener('timeupdate', drawTimeline);
    window.addEventListener('resize', () => { resizeCanvases(); drawOverlay(); drawTimeline(); });
  }

  function loop(){
    drawOverlay();
    drawTimeline();
    if (!video.paused && !video.ended) rafId = requestAnimationFrame(loop);
  }

  function resizeCanvases(){
    const rect = videoWrap.getBoundingClientRect();
    overlay.width = rect.width; overlay.height = rect.height;
    const tRect = timelineCanvas.getBoundingClientRect();
    timelineCanvas.width = tRect.width; timelineCanvas.height = 64;
  }

  function drawOverlay(){
    if (!pipeline) return;
    const ctx = overlay.getContext('2d');
    ctx.clearRect(0,0,overlay.width,overlay.height);
    const dets = detectionsAtTime(video.currentTime)
      .filter(d => !visibleClasses || visibleClasses.has(d.cls));
    dets.forEach(d => {
      const x = d.x*overlay.width, y = d.y*overlay.height;
      const w = d.w*overlay.width, h = d.h*overlay.height;
      const color = classColor(d.cls);
      const cl = Math.min(14, w*0.35, h*0.35);
      ctx.strokeStyle = color; ctx.lineWidth = 2;
      ctx.beginPath();
      // corner brackets (HUD style)
      [[x,y,1,1],[x+w,y,-1,1],[x,y+h,1,-1],[x+w,y+h,-1,-1]].forEach(([cx,cy,sx,sy])=>{
        ctx.moveTo(cx, cy+cl*sy);
        ctx.lineTo(cx, cy);
        ctx.lineTo(cx+cl*sx, cy);
      });
      ctx.stroke();

      const label = `${d.cls} #${d.id} ${d.confidence.toFixed(2)}`;
      ctx.font = '600 11px "IBM Plex Mono", monospace';
      const tw = ctx.measureText(label).width;
      ctx.fillStyle = 'rgba(16,22,31,0.9)';
      ctx.fillRect(x, y-18, tw+10, 18);
      ctx.fillStyle = color;
      ctx.fillRect(x, y-18, 3, 18);
      ctx.fillStyle = '#dce4ec';
      ctx.fillText(label, x+8, y-5);
    });
  }

  function drawTimeline(){
    if (!pipeline) return;
    const ctx = timelineCanvas.getContext('2d');
    const W = timelineCanvas.width, H = timelineCanvas.height;
    ctx.clearRect(0,0,W,H);
    const dur = pipeline.duration;
    if (!Number.isFinite(dur) || dur <= 0){
      // Захист: без цього `for (let s=0; s<=dur; s++)` нижче з
      // dur=Infinity/NaN підвісив би вкладку намертво.
      console.warn('[drawTimeline] некоректна тривалість, пропускаю рендер:', dur);
      return;
    }
    const xOf = t => (t/dur) * W;

    // second ticks
    ctx.strokeStyle = '#1a222d'; ctx.lineWidth = 1;
    ctx.font = '10px "IBM Plex Mono", monospace';
    ctx.fillStyle = '#3d4856';
    for (let s=0; s<=dur; s++){
      const x = xOf(s);
      const major = s % 5 === 0;
      ctx.strokeStyle = major ? '#212b38' : '#161d26';
      ctx.beginPath(); ctx.moveTo(x, H-10); ctx.lineTo(x, major?H-24:H-16); ctx.stroke();
      if (major) ctx.fillText(formatTime(s), x+3, H-2);
    }

    // baseline
    ctx.strokeStyle = '#212b38';
    ctx.beginPath(); ctx.moveTo(0,H-24); ctx.lineTo(W,H-24); ctx.stroke();

    // gradual transitions (hatched amber block)
    pipeline.cuts.filter(c=>c.type==='gradual_transition').forEach(c=>{
      const x1=xOf(c.start), x2=xOf(c.end);
      ctx.fillStyle = 'rgba(255,182,72,0.18)';
      ctx.fillRect(x1, 8, x2-x1, H-32);
      ctx.strokeStyle = 'rgba(255,182,72,0.5)';
      ctx.lineWidth = 1;
      for (let hx=x1; hx<x2; hx+=5){
        ctx.beginPath(); ctx.moveTo(hx,H-24); ctx.lineTo(hx-6,8); ctx.stroke();
      }
    });
    // hard cuts (red spike)
    pipeline.cuts.filter(c=>c.type==='hard_cut').forEach(c=>{
      const x = xOf(c.time);
      ctx.strokeStyle = '#ff4f64'; ctx.lineWidth = 2;
      ctx.beginPath(); ctx.moveTo(x,H-24); ctx.lineTo(x,6); ctx.stroke();
      ctx.fillStyle = '#ff4f64';
      ctx.beginPath(); ctx.moveTo(x-4,6); ctx.lineTo(x+4,6); ctx.lineTo(x,12); ctx.closePath(); ctx.fill();
    });

    // playhead
    const px = xOf(video.currentTime || 0);
    ctx.strokeStyle = '#39e6b0'; ctx.lineWidth = 2;
    ctx.beginPath(); ctx.moveTo(px,0); ctx.lineTo(px,H-24); ctx.stroke();
    ctx.fillStyle = '#39e6b0';
    ctx.beginPath(); ctx.moveTo(px-5,0); ctx.lineTo(px+5,0); ctx.lineTo(px,7); ctx.closePath(); ctx.fill();
  }

  timelineCanvas.addEventListener('click', e => {
    const rect = timelineCanvas.getBoundingClientRect();
    const frac = (e.clientX - rect.left) / rect.width;
    video.currentTime = frac * pipeline.duration;
  });

  function formatTime(t){
    const m = Math.floor(t/60), s = Math.floor(t%60);
    return `${m}:${s.toString().padStart(2,'0')}`;
  }
})();