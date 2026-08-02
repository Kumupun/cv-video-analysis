(function(){
  const DEFAULT_FPS = 30;
  const dropzone = document.getElementById('dropzone');
  const fileInput = document.getElementById('file-input');
  const uploadView = document.getElementById('upload-view');
  const processingView = document.getElementById('processing-view');
  const resultView = document.getElementById('result-view');
  const archiveResultView = document.getElementById('archive-result-view');
  const archiveSummary = document.getElementById('archive-summary');
  const archiveResults = document.getElementById('archive-results');
  const statusSub = document.getElementById('status-sub');
  const resetBtn = document.getElementById('reset-btn');
  const video = document.getElementById('video');
  const overlay = document.getElementById('overlay');
  const videoWrap = document.getElementById('video-wrap');
  const videoRestoreNote = document.getElementById('video-restore-note');
  const videoRestoreText = document.getElementById('video-restore-text');
  const archiveBackBtn = document.getElementById('archive-back-btn');
  const timelineCanvas = document.getElementById('timeline-canvas');
  const consoleEl = document.getElementById('console');
  const stageTrack = document.getElementById('stage-track');
  const statusDetail = document.getElementById('status-detail');

  let pipeline = null;
  let visibleClasses = null;
  let rafId = null;
  let currentTaskId = null;
  let currentVideoUrl = null;
  let reattachVideoOnly = false;
  let archivePollToken = 0;
  let archiveStages = new Map();
  let lastSingleStatusSignature = null;
  let archiveOutcomes = [];
  let activeArchiveTask = null;

  const STORAGE_KEYS = {
    activeSession: 'cv-video-analysis:active-session:v1',
    selectedClasses: 'cv-video-analysis:selected-classes:v1',
  };

  function readStoredJson(key){
    try {
      const value = localStorage.getItem(key);
      return value ? JSON.parse(value) : null;
    } catch (err) {
      console.warn(`[localStorage] не вдалося прочитати ${key}`, err);
      return null;
    }
  }

  function writeStoredJson(key, value){
    try {
      localStorage.setItem(key, JSON.stringify(value));
    } catch (err) {
      console.warn(`[localStorage] не вдалося зберегти ${key}`, err);
    }
  }

  function removeStoredValue(key){
    try {
      localStorage.removeItem(key);
    } catch (err) {
      console.warn(`[localStorage] не вдалося видалити ${key}`, err);
    }
  }


  const CLASS_GROUPS = [
    { label: 'Військові класи', classes: [
      'camouflage_soldier', 'weapon', 'military_tank', 'military_truck',
      'military_vehicle', 'civilian', 'soldier', 'civilian_vehicle',
      'military_artillery', 'trench', 'military_aircraft', 'military_warship'
    ] },
    { label: 'Люди й транспорт', classes: [
      'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
      'truck', 'boat'
    ] },
    { label: 'Вулиця', classes: [
      'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench'
    ] },
    { label: 'Тварини', classes: [
      'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
      'zebra', 'giraffe'
    ] },
    { label: 'Особисті речі та спорт', classes: [
      'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
      'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
      'skateboard', 'surfboard', 'tennis racket'
    ] },
    { label: 'Їжа й посуд', classes: [
      'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana',
      'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
      'donut', 'cake'
    ] },
    { label: 'Дім та електроніка', classes: [
      'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
      'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
      'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
      'scissors', 'teddy bear', 'hair drier', 'toothbrush'
    ] },
  ];
  const CLASS_OPTIONS = CLASS_GROUPS.flatMap(group => group.classes);

  const storedClasses = readStoredJson(STORAGE_KEYS.selectedClasses);
  let selectedClasses = new Set(
    Array.isArray(storedClasses)
      ? storedClasses.filter(className => CLASS_OPTIONS.includes(className))
      : CLASS_OPTIONS
  );

  function persistSelectedClasses(){
    writeStoredJson(
      STORAGE_KEYS.selectedClasses,
      CLASS_OPTIONS.filter(className => selectedClasses.has(className)),
    );
  }

  const classSelectContainer = document.getElementById('class-select-options');
  const classSearchInput = document.getElementById('class-search');

  classSelectContainer.innerHTML = CLASS_GROUPS.map(group => `
    <div class="class-group">
      <div class="class-group-label">${group.label}</div>
      <div class="class-group-items">
        ${group.classes.map(cls => `
          <label class="class-option" data-cls-name="${cls}">
            <input type="checkbox" value="${cls}"${selectedClasses.has(cls) ? ' checked' : ''}> ${cls}
          </label>`).join('')}
      </div>
    </div>`).join('');

  classSelectContainer.querySelectorAll('input[type="checkbox"]').forEach(cb => {
    cb.addEventListener('change', () => {
      if (cb.checked) selectedClasses.add(cb.value);
      else selectedClasses.delete(cb.value);
      persistSelectedClasses();
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
    persistSelectedClasses();
  });
  document.getElementById('classes-select-none').addEventListener('click', () => {
    classSelectContainer.querySelectorAll('input[type="checkbox"]').forEach(cb => {
      cb.checked = false; selectedClasses.delete(cb.value);
    });
    persistSelectedClasses();
  });

  const API_BASE = '/api/v1';

  function persistActiveSession(session){
    writeStoredJson(STORAGE_KEYS.activeSession, {
      version: 1,
      createdAt: new Date().toISOString(),
      ...session,
    });
  }

  function clearActiveSession(){
    removeStoredValue(STORAGE_KEYS.activeSession);
  }

  function readActiveSession(){
    const session = readStoredJson(STORAGE_KEYS.activeSession);
    if (!session || session.version !== 1) return null;
    if (session.type === 'single' && typeof session.taskId === 'string') {
      return session;
    }
    if (
      session.type === 'archive' &&
      Array.isArray(session.tasks) &&
      session.tasks.length > 0 &&
      session.tasks.every(task =>
        typeof task.task_id === 'string' && typeof task.filename === 'string')
    ) {
      return session;
    }
    clearActiveSession();
    return null;
  }





  function normalizeBackendResponse(raw){
    const meta = raw.video || {};
    const fps = meta.fps || DEFAULT_FPS;
    const width = meta.width || video.videoWidth || 1920;
    const height = meta.height || video.videoHeight || 1080;

    const scenes = (raw.scenes || [])
      .sort((a,b) => a.start_frame - b.start_frame)
      .map(s => ({
        start: s.start_frame / fps,
        end: s.end_frame / fps,
        index: s.scene_id,
      }));

    const cuts = (raw.transitions || []).map(t => {
      const isGradual = t.type === 'gradual_transition';
      const hardTime = t.timestamp ?? ((t.frame ?? 0) / fps);
      const start = isGradual
        ? (t.start_timestamp ?? ((t.start_frame ?? 0) / fps))
        : hardTime;
      const end = isGradual
        ? (t.end_timestamp ?? ((t.end_frame ?? t.start_frame ?? 0) / fps))
        : hardTime;
      return {
        time: isGradual ? start : hardTime,
        start,
        end,
        type: t.type,
        confidence: Number(t.confidence) || 0,
      };
    });

    const trackMap = new Map();
    (raw.tracks || []).forEach(d => {
      const key = `${d.scene_id}:${d.track_id}`;
      if (!trackMap.has(key)) {
        trackMap.set(key, { id: key, cls: d.class_name, sceneIndex: d.scene_id, detections: [] });
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

    const inferredDuration = Math.max(
      0,
      ...scenes.map(s => s.end),
      ...tracks.flatMap(t => t.keyframes.map(k => k.t)),
    );
    const duration = meta.duration_seconds || video.duration || inferredDuration || 12;

    return { duration, fps, scenes, cuts, tracks };
  }


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
    if (reattachVideoOnly) {
      reattachVideoOnly = false;
      if (f) attachVideoForPlayback(f);
      fileInput.value = '';
      return;
    }
    if (f) handleFile(f);
  });

  document.getElementById('reattach-video-btn').addEventListener('click', () => {
    reattachVideoOnly = true;
    fileInput.click();
  });

  function isZipArchive(file){
    return file.name.toLowerCase().endsWith('.zip') ||
      ['application/zip', 'application/x-zip-compressed'].includes(file.type);
  }

  function showProcessing(){
    uploadView.hidden = true;
    processingView.hidden = false;
    resultView.hidden = true;
    archiveResultView.hidden = true;
    statusSub.textContent = 'обробка…';
    statusDetail.textContent = 'Очікуємо відповідь backend…';
  }

  function handleFile(file){
    reattachVideoOnly = false;
    archivePollToken += 1;
    archiveStages.clear();
    currentTaskId = null;
    lastSingleStatusSignature = null;
    archiveOutcomes = [];
    activeArchiveTask = null;
    archiveBackBtn.hidden = true;
    clearActiveSession();
    consoleEl.replaceChildren();
    videoRestoreNote.hidden = true;

    if (isZipArchive(file)) {
      showProcessing();
      startArchivePipeline(file);
      return;
    }

    if (!file.type.startsWith('video/') && !file.name.toLowerCase().endsWith('.mkv')) {
      appendLog('Оберіть відеофайл або ZIP-архів', true);
      return;
    }

    if (currentVideoUrl) URL.revokeObjectURL(currentVideoUrl);
    currentVideoUrl = URL.createObjectURL(file);
    video.src = currentVideoUrl;
    video.onloadedmetadata = () => {
      showProcessing();
      startRealPipeline(file);
    };
  }

  function attachVideoForPlayback(file){
    if (!file.type.startsWith('video/') && !file.name.toLowerCase().endsWith('.mkv')) {
      appendLog('Для playback потрібно вибрати відеофайл', true);
      return;
    }
    if (activeArchiveTask) {
      const expectedName = activeArchiveTask.filename.split(/[\\/]/).pop();
      if (file.name.toLowerCase() !== expectedName.toLowerCase()) {
        videoRestoreText.textContent =
          `Потрібен файл ${expectedName}. Вибрано ${file.name}.`;
        appendLog(`Очікувався ${expectedName}, вибрано ${file.name}`, true);
        return;
      }
    }
    if (currentVideoUrl) URL.revokeObjectURL(currentVideoUrl);
    currentVideoUrl = URL.createObjectURL(file);
    video.src = currentVideoUrl;
    video.onloadedmetadata = () => {
      videoRestoreNote.hidden = true;
      resizeCanvases();
      drawOverlay();
      drawTimeline();
      video.controls = true;
      video.play().catch(() => {});
    };
  }

  function clearVideoPlayback(){
    cancelAnimationFrame(rafId);
    video.pause();
    video.removeAttribute('src');
    video.load();
    if (currentVideoUrl) {
      URL.revokeObjectURL(currentVideoUrl);
      currentVideoUrl = null;
    }
  }

  function fileBasename(filename){
    return String(filename || 'video').split(/[\\/]/).pop();
  }

  async function startRealPipeline(file){
    try {
      const form = new FormData();
      form.append('file', file);
      form.append('classes', JSON.stringify(Array.from(selectedClasses)));

      const { task_id } = await fetchJson(`${API_BASE}/process`, {
        method:'POST',
        body: form,
      });
      currentTaskId = task_id;
      persistActiveSession({
        type: 'single',
        taskId: task_id,
        filename: file.name,
      });
      pollStatus(task_id);
    } catch (err){
      console.error('[Помилка запуску обробки]', err);
      statusSub.textContent = 'помилка запуску обробки';
      statusDetail.textContent = err.message || 'Невідома помилка backend';
      appendLog(err.message || 'unknown error', true);
    }
  }

  async function startArchivePipeline(file){
    const token = archivePollToken;
    try {
      const form = new FormData();
      form.append('file', file);
      form.append('classes', JSON.stringify(Array.from(selectedClasses)));

      const response = await fetchJson(`${API_BASE}/process/archive`, {
        method: 'POST',
        body: form,
      });
      appendLog(`Архів прийнято: ${response.accepted_count} відео`);
      response.skipped_files.forEach(item => {
        appendLog(`Пропущено ${item.filename}: ${item.reason}`, true);
      });
      persistActiveSession({
        type: 'archive',
        archiveFilename: response.archive_filename,
        tasks: response.tasks.map(task => ({
          task_id: task.task_id,
          filename: task.filename,
        })),
      });
      pollArchiveStatuses(response.tasks, token);
    } catch (err) {
      console.error('[Помилка запуску архіву]', err);
      statusSub.textContent = 'помилка запуску архіву';
      statusDetail.textContent = err.message || 'Невідома помилка backend';
      appendLog(err.message || 'unknown error', true);
    }
  }

  async function fetchJson(url, options = {}){


    const response = await fetch(url, {
      cache: 'no-store',
      ...options,
    });
    let payload = null;
    try {
      payload = await response.json();
    } catch (_err) {
      if (!response.ok) {
        const error = new Error(`${response.status} ${response.statusText}`);
        error.status = response.status;
        throw error;
      }
      throw new Error(`Сервер повернув невалідний JSON для ${url}`);
    }
    if (!response.ok) {
      const detail = payload?.detail;
      const message = typeof detail === 'string'
        ? detail
        : detail?.message || `${response.status} ${response.statusText}`;
      const error = new Error(message);
      error.status = response.status;
      throw error;
    }
    return payload;
  }

  function appendLog(message, isError = false){
    const line = document.createElement('div');
    line.className = 'ln';
    if (isError) line.style.color = 'var(--cut-hard)';
    line.textContent = `$ ${message}`;
    consoleEl.appendChild(line);
  }

  async function pollStatus(taskId){
    if (taskId !== currentTaskId) return;
    try {
      const data = await fetchJson(`${API_BASE}/status/${taskId}`);

      if (taskId !== currentTaskId) return;
      updateStageUI(data.stage, data.progress, data.message, data);

      const statusSignature = [
        data.stage,
        data.progress,
        data.cut_completed_chunks,
        data.tracking_completed_chunks,
      ].join(':');
      if (statusSignature !== lastSingleStatusSignature) {
        lastSingleStatusSignature = statusSignature;
        appendLog(formatStatusDetail(data));
      }

      if (data.stage === 'completed') {
        const results = await fetchJson(`${API_BASE}/results/${taskId}`);
        if (taskId !== currentTaskId) return;
        pipeline = normalizeBackendResponse(results);
        processingView.hidden = true;
        resultView.hidden = false;
        resetBtn.style.display = 'inline-block';
        statusSub.textContent = 'готово';
        videoRestoreNote.hidden = Boolean(currentVideoUrl);
        setupResultView();
      } else if (data.stage === 'failed') {
        statusSub.textContent = 'помилка обробки';
        appendLog(data.error_detail || data.message || 'unknown error', true);
        resetBtn.style.display = 'inline-block';
      } else {
        setTimeout(() => pollStatus(taskId), 1000);
      }
    } catch (err){
      console.warn('[втрачено зв\'язок з бекендом]', err);
      if (err.status === 404) {
        expireRestoredSession();
        return;
      }
      statusDetail.textContent = `Backend тимчасово недоступний: ${err.message || 'помилка мережі'}`;
      setTimeout(() => pollStatus(taskId), 2000);
    }
  }

  async function pollArchiveStatuses(tasks, token){
    if (token !== archivePollToken) return;
    try {
      const statuses = await Promise.all(tasks.map(async task => ({
        task,
        status: await fetchJson(`${API_BASE}/status/${task.task_id}`),
      })));

      statuses.forEach(({task, status}) => {
        const previousStage = archiveStages.get(task.task_id);
        if (previousStage !== status.stage) {
          archiveStages.set(task.task_id, status.stage);
          appendLog(`${task.filename}: ${status.stage}`,
            status.stage === 'failed');
        }
      });

      const finished = statuses.filter(({status}) =>
        ['completed', 'failed'].includes(status.stage)).length;
      const averageProgress = statuses.reduce(
        (total, {status}) => total + (Number(status.progress) || 0), 0
      ) / statuses.length;
      const active = statuses.find(({status}) =>
        !['completed', 'failed'].includes(status.stage));
      updateStageUI(
        active?.status.stage || 'completed',
        averageProgress,
        active?.status.message || '',
        active?.status || null,
      );
      statusSub.textContent = `архів: ${finished}/${statuses.length} · ${Math.round(averageProgress)}%`;

      if (finished === statuses.length) {
        const outcomes = await Promise.all(statuses.map(async ({task, status}) => {
          if (status.stage === 'failed') return {task, status, result: null};
          const result = await fetchJson(`${API_BASE}/results/${task.task_id}`);
          return {task, status, result};
        }));
        if (token === archivePollToken) showArchiveResults(outcomes);
        return;
      }
      setTimeout(() => pollArchiveStatuses(tasks, token), 1000);
    } catch (err) {
      console.warn('[втрачено зв\'язок з бекендом]', err);
      if (token === archivePollToken) {
        if (err.status === 404) {
          expireRestoredSession();
          return;
        }
        setTimeout(() => pollArchiveStatuses(tasks, token), 2000);
      }
    }
  }

  function showArchiveResults(outcomes){
    const completed = outcomes.filter(item => item.result).length;
    const failed = outcomes.length - completed;
    archiveOutcomes = outcomes;
    activeArchiveTask = null;
    processingView.hidden = true;
    resultView.hidden = true;
    archiveResultView.hidden = false;
    archiveBackBtn.hidden = true;
    videoRestoreNote.hidden = true;
    resetBtn.style.display = 'inline-block';
    statusSub.textContent = failed ?
      `архів готово: ${completed}/${outcomes.length}` : 'архів готово';
    archiveSummary.textContent =
      `${completed} завершено · ${failed} з помилкою · JSON для кожного відео окремо`;
    archiveResults.replaceChildren();

    outcomes.forEach(outcome => {
      const {task, status, result} = outcome;
      const card = document.createElement('div');
      card.className = `archive-result-card${result ? '' : ' failed'}`;
      const details = document.createElement('div');
      const name = document.createElement('div');
      name.className = 'archive-result-name';
      name.textContent = task.filename;
      const meta = document.createElement('div');
      meta.className = `archive-result-meta${result ? '' : ' archive-result-error'}`;
      meta.textContent = result
        ? `${result.scenes?.length || 0} сцен · ${result.transitions?.length || 0} переходів · ${result.tracks?.length || 0} детекцій`
        : status.error_detail || status.message || 'Помилка обробки';
      details.append(name, meta);
      card.appendChild(details);

      if (result) {
        const actions = document.createElement('div');
        actions.className = 'archive-result-actions';
        const open = document.createElement('button');
        open.type = 'button';
        open.className = 'mini-btn';
        open.textContent = 'Відкрити результат';
        open.addEventListener('click', () => openArchiveResult(outcome));

        const download = document.createElement('button');
        download.type = 'button';
        download.className = 'mini-btn';
        download.textContent = 'Завантажити JSON';
        download.addEventListener('click', () => {
          const blob = new Blob([JSON.stringify(result, null, 2)], {
            type: 'application/json',
          });
          const url = URL.createObjectURL(blob);
          const link = document.createElement('a');
          link.href = url;
          link.download = `${fileBasename(task.filename).replace(/\.[^.]+$/, '')}.json`;
          link.click();
          URL.revokeObjectURL(url);
        });
        actions.append(open, download);
        card.appendChild(actions);
      }
      archiveResults.appendChild(card);
    });
  }

  function openArchiveResult(outcome){
    if (!outcome?.result) return;
    clearVideoPlayback();
    activeArchiveTask = outcome.task;
    pipeline = normalizeBackendResponse(outcome.result);
    processingView.hidden = true;
    archiveResultView.hidden = true;
    resultView.hidden = false;
    archiveBackBtn.hidden = false;
    resetBtn.style.display = 'inline-block';
    const expectedName = fileBasename(outcome.task.filename);
    statusSub.textContent = `готово: ${expectedName}`;
    statusDetail.textContent = '';
    videoRestoreText.textContent =
      `Детекції та timeline завантажено. Для playback розпакуйте ZIP і виберіть файл ${expectedName}. Повторна обробка не запускається.`;
    videoRestoreNote.hidden = false;
    setupResultView();
  }

  archiveBackBtn.addEventListener('click', () => {
    if (!archiveOutcomes.length) return;
    clearVideoPlayback();
    activeArchiveTask = null;
    pipeline = null;
    resultView.hidden = true;
    archiveResultView.hidden = false;
    archiveBackBtn.hidden = true;
    videoRestoreNote.hidden = true;
    const completed = archiveOutcomes.filter(item => item.result).length;
    const failed = archiveOutcomes.length - completed;
    statusSub.textContent = failed
      ? `архів готово: ${completed}/${archiveOutcomes.length}`
      : 'архів готово';
  });

  const STAGES = [
    { key:'decoding', label:'Decode', sub:'FFmpeg / Ray' },
    { key:'cut_detection', label:'Cut Detection', sub:'AutoShot' },
    { key:'tracking', label:'Tracking', sub:'YOLO + ByteTrack' },
    { key:'aggregating', label:'Aggregation', sub:'Backend' }
  ];

  function formatStatusDetail(status){
    const parts = [];
    if (status?.message) parts.push(status.message);
    const total = Number(status?.total_chunks) || 0;
    if (total > 0) {
      const cut = Number(status?.cut_completed_chunks) || 0;
      const tracking = Number(status?.tracking_completed_chunks) || 0;
      parts.push(`AutoShot ${cut}/${total} · Tracking ${tracking}/${total}`);
    }
    return parts.join(' · ') || 'Backend обробляє задачу';
  }

  function updateStageUI(stageKey, progress, message = '', counters = null){
    const stageAliases = {
      queued: 'decoding',
      downloading: 'decoding',
      completed: 'aggregating',
    };
    const normalizedStage = stageAliases[stageKey] || stageKey;
    const idx = STAGES.findIndex(s => s.key === normalizedStage);
    if (idx !== -1 && !stageTrack.children.length) {
      stageTrack.innerHTML = STAGES.map(s => `
        <div class="stage" data-key="${s.key}">
          <div class="stage-line"></div>
          <div class="node">●</div>
          <div class="label">${s.label}</div>
          <div class="sublabel">${s.sub}</div>
        </div>`).join('');
    }
    if (idx !== -1) {
      STAGES.forEach((s, i) => {
        const el = stageTrack.querySelector(`[data-key="${s.key}"]`);
        el.classList.remove('active','done');
        if (i < idx) el.classList.add('done');
        else if (i === idx) el.classList.add('active');
      });
    }
    const numericProgress = Number(progress);
    if (Number.isFinite(numericProgress)) {
      statusSub.textContent = `обробка… ${Math.round(numericProgress)}%`;
    }
    statusDetail.textContent = formatStatusDetail({
      ...(counters || {}),
      message,
    });
  }

  function expireRestoredSession(){
    archivePollToken += 1;
    currentTaskId = null;
    reattachVideoOnly = false;
    clearActiveSession();
    processingView.hidden = true;
    resultView.hidden = true;
    archiveResultView.hidden = true;
    uploadView.hidden = false;
    resetBtn.style.display = 'none';
    statusSub.textContent = 'задачу вже видалено з сервера';
    statusDetail.textContent = '';
    appendLog('Збережена задача більше не існує. Завантажте відео знову.', true);
  }

  function restoreActiveSession(){
    const session = readActiveSession();
    if (!session) return;

    showProcessing();
    resetBtn.style.display = 'inline-block';
    appendLog('Стан відновлено після оновлення сторінки');

    if (session.type === 'single') {
      currentTaskId = session.taskId;
      lastSingleStatusSignature = null;
      const filename = session.filename || 'відео';
      statusSub.textContent = `відновлення: ${filename}`;
      appendLog(`Продовжуємо стежити за ${filename}`);
      pollStatus(session.taskId);
      return;
    }

    const token = ++archivePollToken;
    archiveStages.clear();
    statusSub.textContent = `відновлення архіву: ${session.tasks.length} відео`;
    appendLog(`Продовжуємо стежити за ${session.tasks.length} задачами архіву`);
    pollArchiveStatuses(session.tasks, token);
  }

  resetBtn.addEventListener('click', () => {
    archivePollToken += 1;
    archiveStages.clear();
    clearVideoPlayback();
    resultView.hidden = true;
    archiveResultView.hidden = true;
    processingView.hidden = true;
    uploadView.hidden = false;
    resetBtn.style.display = 'none';
    statusSub.textContent = 'очікування завантаження';
    statusDetail.textContent = '';
    fileInput.value = '';
    currentTaskId = null;
    lastSingleStatusSignature = null;
    archiveOutcomes = [];
    activeArchiveTask = null;
    archiveBackBtn.hidden = true;
    clearActiveSession();
    pipeline = null;
    archiveSummary.textContent = '';
    archiveResults.replaceChildren();
    consoleEl.replaceChildren();
    videoRestoreNote.hidden = true;
    videoRestoreText.textContent =
      'Результат відновлено з бекенда. Браузер не зберігає доступ до локального відеофайлу після F5.';
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
      <div class="scene-row" data-t="${c.start}" title="confidence ${c.confidence.toFixed(3)}">
        <span>${c.type === 'gradual_transition'
          ? `${formatTime(c.start)}–${formatTime(c.end)}`
          : formatTime(c.time)}</span>
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
      ctx.fillRect(x1, 8, Math.max(1, x2-x1), H-32);
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

  restoreActiveSession();
})();
