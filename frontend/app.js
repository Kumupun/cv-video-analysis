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

  const CUSTOM_CLASSES = [
    'camouflage_soldier', 'weapon', 'military_tank', 'military_truck', 
    'military_vehicle', 'soldier', 'civilian_vehicle', 'military_artillery', 
    'military_aircraft', 'military_warship'
  ];

  const CLASS_GROUPS = [
    { label: 'COCO-80', classes: COCO_CLASSES },
    { label: 'Донавчені класи', classes: CUSTOM_CLASSES },
  ];
  const CLASS_OPTIONS = [...COCO_CLASSES, ...CUSTOM_CLASSES];

  let selectedClasses = new Set(CLASS_OPTIONS); 

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

  const API_BASE = 'http://127.0.0.1:8000/api/v1';

  function normalizeBackendResponse(raw){
    if (!raw || typeof raw !== 'object') {
      return { duration: 0, fps: DEFAULT_FPS, scenes: [], cuts: [], tracks: [] };
    }

    const meta = raw.video || {};
    const fps = meta.fps || DEFAULT_FPS;
    const width = meta.width || video.videoWidth || 1920;
    const height = meta.height || video.videoHeight || 1080;

    const transitions = (Array.isArray(raw.transitions) ? raw.transitions : []).map(t => {
      const time = t.timestamp ?? (t.frame != null ? t.frame / fps : 0);
      const hasRange = t.start_frame != null && t.end_frame != null;
      return {
        time,
        start: hasRange ? (t.start_timestamp ?? t.start_frame / fps) : time,
        end: hasRange ? (t.end_timestamp ?? t.end_frame / fps) : time,
        type: t.type,
      };
    }).sort((a,b) => a.time - b.time);

    const trackMap = new Map();
    (Array.isArray(raw.tracks) ? raw.tracks : []).forEach(d => {
      const sceneId = d.scene_id ?? d.sceneId;
      if (!sceneId) return;
      const key = `${sceneId}:${d.track_id}`;
      if (!trackMap.has(key)) {
        trackMap.set(key, { id: key, cls: d.class_name, sceneIndex: sceneId, detections: [] });
      }
      trackMap.get(key).detections.push(d);
    });

    const tracks = Array.from(trackMap.values()).map(t => {
      t.detections.sort((a,b) => a.frame - b.frame);
      const keyframes = t.detections.map(d => ({
        t: d.frame / fps,
        x: d.bbox.x / width,
        y: d.bbox.y / height,
        w: d.bbox.width / width,
        h: d.bbox.height / height,
        confidence: d.confidence,
      }));
      return {
        id: t.id,
        cls: t.cls,
        sceneIndex: t.sceneIndex,
        confidence: keyframes[0]?.confidence ?? 0.8,
        keyframes,
      };
    });

    const scenesById = new Map();
    tracks.forEach(tr => {
      tr.keyframes.forEach(frame => {
        const scene = scenesById.get(tr.sceneIndex) || { id: tr.sceneIndex, start_frame: frame.t * fps, end_frame: frame.t * fps };
        scene.start_frame = Math.min(scene.start_frame, frame.t * fps);
        scene.end_frame = Math.max(scene.end_frame, frame.t * fps);
        scenesById.set(tr.sceneIndex, scene);
      });
    });

    let scenes = Array.from(scenesById.values())
      .sort((a,b) => a.start_frame - b.start_frame)
      .map(s => ({ start: s.start_frame / fps, end: s.end_frame / fps, index: s.id }));

    const duration = meta.duration_seconds || raw.duration || video.duration || Math.max(
      scenes.length ? scenes[scenes.length-1].end : 0,
      transitions.length ? transitions[transitions.length-1].end : 0,
      tracks.length ? Math.max(...tracks.flatMap(t => t.keyframes.map(k => k.t))) : 0,
      DEFAULT_FPS,
    );

    if (!scenes.length && transitions.length) {
      const boundaries = [0, ...transitions.map(c => c.start), duration].filter((v, i, arr) => arr.indexOf(v) === i).sort((a,b) => a - b);
      scenes = boundaries.slice(0, -1).map((start, index) => ({ start, end: boundaries[index + 1], index: String(index) }));
    }

    return { duration, fps, scenes, cuts: transitions, tracks };
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

      const duration = await resolveVideoDuration(video);
      startRealPipeline(file, duration);
    };
  }

  function resolveVideoDuration(videoEl, fallback = 12){
    return new Promise(resolve => {
      if (Number.isFinite(videoEl.duration) && videoEl.duration > 0){
        resolve(videoEl.duration);
        return;
      }
      
      const onTimeUpdate = () => {
        videoEl.removeEventListener('timeupdate', onTimeUpdate);
        videoEl.currentTime = 0;
        resolve(Number.isFinite(videoEl.duration) && videoEl.duration > 0 ? videoEl.duration : fallback);
      };
      videoEl.addEventListener('timeupdate', onTimeUpdate);
      videoEl.currentTime = 1e101;

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
      if (!res.ok) {
        const errorText = await res.text().catch(() => null);
        throw new Error(`POST /process → ${res.status}${errorText ? `: ${errorText}` : ''}`);
      }
      
      const { task_id } = await res.json();
      currentTaskId = task_id;
      pollStatus(task_id);
    } catch (err){
      console.error('[Помилка запуску обробки]', err);
      statusSub.textContent = err.message || 'помилка запуску обробки';
      consoleEl.innerHTML += `<div class="ln" style="color:var(--cut-hard)">$ ${err.message || 'unknown error'}</div>`;
    }
  }

  async function pollStatus(taskId){
    try {
      const res = await fetch(`${API_BASE}/status/${taskId}`);
      if (!res.ok) {
        const errorText = await res.text().catch(() => null);
        throw new Error(`GET /status → ${res.status}${errorText ? `: ${errorText}` : ''}`);
      }
      const data = await res.json();
      updateStageUI(data.stage, data.progress);
      statusSub.textContent = data.message || data.stage || 'обробка…';

      if (data.stage === 'completed') {
        const resultsRes = await fetch(`${API_BASE}/results/${taskId}`);
        if (resultsRes.status === 202) {
          setTimeout(() => pollStatus(taskId), 1000);
          return;
        }
        if (!resultsRes.ok) {
          const errorData = await resultsRes.json().catch(() => null);
          const errorMessage = errorData?.detail?.message || errorData?.detail || `GET /results → ${resultsRes.status}`;
          statusSub.textContent = 'помилка обробки';
          consoleEl.innerHTML += `<div class="ln" style="color:var(--cut-hard)">$ ${errorMessage}</div>`;
          return;
        }
        pipeline = normalizeBackendResponse(await resultsRes.json());
        processingView.hidden = true;
        resultView.hidden = false;
        resetBtn.style.display = 'inline-block';
        statusSub.textContent = 'готово';
        setupResultView();
      } else if (data.stage === 'failed') {
        statusSub.textContent = 'помилка обробки';
        consoleEl.innerHTML += `<div class="ln" style="color:var(--cut-hard)">$ ${data.error_detail || data.message || 'unknown error'}</div>`;
      } else {
        setTimeout(() => pollStatus(taskId), 1000);
      }
    } catch (err){
      console.warn('[втрачено зв\'язок з бекендом]', err);
      statusSub.textContent = err.message || 'очікування відповіді сервера...';
      setTimeout(() => pollStatus(taskId), 2000);
    }
  }

  const STAGES = [
    { key:'queued', label:'Queued', sub:'Waiting for processing' },
    { key:'downloading', label:'Downloading', sub:'Ingesting video' },
    { key:'decoding', label:'Decoding', sub:'Video decode' },
    { key:'cut_detection', label:'Cut Detection', sub:'Scene boundary detection' },
    { key:'tracking', label:'Tracking', sub:'Object tracking' },
    { key:'aggregating', label:'Aggregating', sub:'Finalizing result' },
    { key:'completed', label:'Completed', sub:'Ready to view' },
    { key:'failed', label:'Failed', sub:'Processing error' },
  ];

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

  const CURATED_CLASS_COLORS = {
    person: '#39e6b0',
    car: '#5b9dff',
    truck: '#5b9dff',
    bus: '#5b9dff',
    motorcycle: '#5b9dff',
    bicycle: '#5b9dff',

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
        drawOverlay(); 
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
      console.warn('[drawTimeline] некоректна тривалість, пропускаю рендер:', dur);
      return;
    }
    const xOf = t => (t/dur) * W;

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

    ctx.strokeStyle = '#212b38';
    ctx.beginPath(); ctx.moveTo(0,H-24); ctx.lineTo(W,H-24); ctx.stroke();

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
    
    pipeline.cuts.filter(c=>c.type==='hard_cut').forEach(c=>{
      const x = xOf(c.time);
      ctx.strokeStyle = '#ff4f64'; ctx.lineWidth = 2;
      ctx.beginPath(); ctx.moveTo(x,H-24); ctx.lineTo(x,6); ctx.stroke();
      ctx.fillStyle = '#ff4f64';
      ctx.beginPath(); ctx.moveTo(x-4,6); ctx.lineTo(x+4,6); ctx.lineTo(x,12); ctx.closePath(); ctx.fill();
    });

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
