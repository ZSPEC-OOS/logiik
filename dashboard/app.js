/* ─── Config ─────────────────────────────────────────────────── */
const API = window.location.origin;
const WS  = `${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/ws/training`;

/* ─── Firebase ───────────────────────────────────────────────── */
const FIREBASE_CONFIG = {
  apiKey:            'AIzaSyDkbAhy7PlrYzHR5F-EDBquUtZ9fwLsyHg',
  authDomain:        'logiik.firebaseapp.com',
  projectId:         'logiik',
  storageBucket:     'logiik.firebasestorage.app',
  messagingSenderId: '835296209489',
  appId:             '1:835296209489:web:830f7cc853e729ecc42670',
};

let db            = null;
let _writeTimer   = null;

function initFirebase() {
  try {
    if (!firebase.apps.length) firebase.initializeApp(FIREBASE_CONFIG);
    db = firebase.firestore();
  } catch (e) {
    console.warn('Firebase init failed:', e);
  }
}

async function _fetchPINFromFirestore() {
  if (!db) return null;
  try {
    const snap = await db.collection('app_config').doc('pin').get();
    if (snap.exists) return snap.data().value;
    await db.collection('app_config').doc('pin').set({ value: '5522' });
    return '5522';
  } catch (_) { return null; }
}

function _scheduleFirestoreWrite() {
  if (!db) return;
  clearTimeout(_writeTimer);
  _writeTimer = setTimeout(async () => {
    try {
      const key = state.persistenceKey || 'default';
      await db.collection('sessions').doc(key).set({
        steps:     state.steps,
        trainLoss: state.trainLoss,
        valSteps:  state.valSteps,
        valLoss:   state.valLoss,
        valPpl:    state.valPpl,
        phase:     state.phase,
        updatedAt: firebase.firestore.FieldValue.serverTimestamp(),
      });
    } catch (_) {}
  }, 2000);
}

async function _restoreFromFirestore() {
  if (!db) return;
  try {
    const key  = state.persistenceKey || 'default';
    const snap = await db.collection('sessions').doc(key).get();
    if (!snap.exists) return;
    const d = snap.data();
    if (Array.isArray(d.steps) && d.steps.length) {
      state.steps     = d.steps;
      state.trainLoss = Array.isArray(d.trainLoss) ? d.trainLoss : [];
      state.valSteps  = Array.isArray(d.valSteps)  ? d.valSteps  : [];
      state.valLoss   = Array.isArray(d.valLoss)   ? d.valLoss   : [];
      state.valPpl    = Array.isArray(d.valPpl)    ? d.valPpl    : [];
      state.phase     = d.phase || null;
      addLog('system', 'State restored from Firestore');
    }
  } catch (_) {}
}

async function _writePhaseEvent(phase, step) {
  if (!db) return;
  try {
    await db.collection('phase_events').add({
      phase,
      step:      step ?? null,
      timestamp: firebase.firestore.FieldValue.serverTimestamp(),
    });
  } catch (_) {}
}

async function _writeQAEvent(bankCount, step) {
  if (!db) return;
  try {
    await db.collection('qa_events').add({
      bank_count: bankCount,
      step:       step ?? null,
      timestamp:  firebase.firestore.FieldValue.serverTimestamp(),
    });
  } catch (_) {}
}

/* ─── PIN Lock Screen ────────────────────────────────────────── */
let _resolvedPIN = '5522';

async function initPINScreen() {
  if (sessionStorage.getItem('nero-pin-verified') === '1') {
    _hidePINScreen();
    return;
  }
  // Screen is already visible from HTML default; fetch remote PIN asynchronously
  const remote = await _fetchPINFromFirestore();
  if (remote) _resolvedPIN = remote;
  document.getElementById('pin-input').focus();
}

function verifyPIN() {
  const entered = document.getElementById('pin-input').value;
  const errEl   = document.getElementById('pin-error');
  if (entered === _resolvedPIN) {
    sessionStorage.setItem('nero-pin-verified', '1');
    errEl.style.visibility = 'hidden';
    _hidePINScreen();
  } else {
    errEl.style.visibility = 'visible';
    document.getElementById('pin-input').value = '';
    document.getElementById('pin-input').focus();
  }
}

function _hidePINScreen() {
  const el = document.getElementById('pin-screen');
  if (el) el.style.display = 'none';
}

/* ─── AI Model Configuration ─────────────────────────────────── */
const modelConfig = {
  apiKey:      '',
  baseUrl:     '',
  modelId:     '',
  initialized: false,
};

function _loadModelConfigFromStorage() {
  const apiKey  = localStorage.getItem('nero-mc-apikey');
  const baseUrl = localStorage.getItem('nero-mc-baseurl');
  const modelId = localStorage.getItem('nero-mc-model');
  if (apiKey)  { const el = document.getElementById('mc-apikey');  if (el) el.value = apiKey;  modelConfig.apiKey  = apiKey;  }
  if (baseUrl) { const el = document.getElementById('mc-baseurl'); if (el) el.value = baseUrl; modelConfig.baseUrl = baseUrl; }
  if (modelId) { const el = document.getElementById('mc-model');   if (el) el.value = modelId; modelConfig.modelId = modelId; }
  if (apiKey && baseUrl && modelId) {
    modelConfig.initialized = true;
    _syncModelConfigToLegacy();
    _setMCStatus(true, modelId);
  }
}

function _syncModelConfigToLegacy() {
  const fApiKey  = document.getElementById('f-apikey');
  const fBaseUrl = document.getElementById('f-baseurl');
  const fModel   = document.getElementById('f-model');
  if (fApiKey)  fApiKey.value  = modelConfig.apiKey;
  if (fBaseUrl) fBaseUrl.value = modelConfig.baseUrl;
  if (fModel)   fModel.value   = modelConfig.modelId;
}

function _setMCStatus(live, label) {
  const dot  = document.getElementById('mc-dot');
  const text = document.getElementById('mc-status-text');
  if (dot)  dot.className  = live ? 'mc-dot live' : 'mc-dot';
  if (text) text.textContent = live ? `Connected · ${label}` : 'Not configured';
}

async function initModelConfig() {
  const apiKey  = document.getElementById('mc-apikey').value.trim();
  const baseUrl = document.getElementById('mc-baseurl').value.trim();
  const modelId = document.getElementById('mc-model').value.trim();
  const msg     = document.getElementById('mc-msg');

  if (!apiKey || !baseUrl || !modelId) {
    msg.textContent = 'All three fields are required.';
    msg.className   = 'init-msg error';
    return;
  }

  msg.textContent = 'Validating…';
  msg.className   = 'init-msg pending';

  modelConfig.apiKey      = apiKey;
  modelConfig.baseUrl     = baseUrl;
  modelConfig.modelId     = modelId;
  modelConfig.initialized = true;

  localStorage.setItem('nero-mc-apikey',  apiKey);
  localStorage.setItem('nero-mc-baseurl', baseUrl);
  localStorage.setItem('nero-mc-model',   modelId);

  _syncModelConfigToLegacy();

  try {
    const res = await fetch(`${API}/health`);
    if (res.ok) {
      msg.textContent = 'Config saved — connection active.';
      msg.className   = 'init-msg success';
      _setMCStatus(true, modelId);
    } else {
      throw new Error();
    }
  } catch {
    msg.textContent = 'Config saved — server unreachable, will connect when available.';
    msg.className   = 'init-msg pending';
    _setMCStatus(false, '');
  }
}

/* ─── State ──────────────────────────────────────────────────── */
const state = {
  initialized:      false,
  training:         false,
  trainingComplete: false,
  phase:            null,
  steps:            [],
  trainLoss:        [],
  valLoss:          [],
  valPpl:           [],
  valSteps:         [],
  logEntries:       [],
  persistenceKey:   'nero-session-default',
  historyLoaded:    false,
  _lastBankCount:   null,
};

/* ─── Phase definitions ──────────────────────────────────────── */
const PHASE_DEFS = [
  { key: 'Memorization', label: 'Phase 1 · Memorization', desc: 'Learning Q+A structure from Kimi K2.5', color: '#4CAF50' },
  { key: 'Generation',   label: 'Phase 2 · Generation',   desc: 'Creating original answers',              color: '#667eea' },
  { key: 'Abstraction',  label: 'Phase 3 · Abstraction',  desc: 'Cross-domain knowledge synthesis',       color: '#f093fb' },
];

/* ─── Tab switching ──────────────────────────────────────────── */
function showTab(id, btn) {
  document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.tab').forEach(b => b.classList.remove('active'));
  document.getElementById(`tab-${id}`).classList.add('active');
  btn.classList.add('active');
  if (id === 'monitor')   { renderPhases(); renderCharts(); }
  if (id === 'knowledge') loadKBFolders();
  if (id === 'logs')      loadLogs();
}

/* ─── Status badge ───────────────────────────────────────────── */
function setStatus(label, cls) {
  const el = document.getElementById('status-badge');
  el.textContent = `● ${label}`;
  el.className   = `status-badge ${cls}`;
}

/* ─── Metric helpers ─────────────────────────────────────────── */
function setMetric(id, value, delta) {
  const el = document.getElementById(id);
  if (el) el.textContent = value ?? '—';
  const dd = document.getElementById(`${id}-d`);
  if (dd && delta !== undefined) dd.textContent = delta;
}

function fmt(n, decimals = 4) {
  return n == null ? '—' : Number(n).toFixed(decimals);
}

function _sessionKey(kbPath = '') {
  const suffix = (kbPath || 'default').replace(/[^\w.-]+/g, '_').slice(-80);
  return `nero-session-${suffix}`;
}

function _persistState() {
  try {
    const payload = {
      steps: state.steps,
      trainLoss: state.trainLoss,
      valSteps: state.valSteps,
      valLoss: state.valLoss,
      valPpl: state.valPpl,
      phase: state.phase,
    };
    localStorage.setItem(state.persistenceKey, JSON.stringify(payload));
  } catch (_) {}
}

function _restoreState() {
  try {
    const raw = localStorage.getItem(state.persistenceKey);
    if (!raw) return;
    const saved = JSON.parse(raw);
    state.steps = Array.isArray(saved.steps) ? saved.steps : [];
    state.trainLoss = Array.isArray(saved.trainLoss) ? saved.trainLoss : [];
    state.valSteps = Array.isArray(saved.valSteps) ? saved.valSteps : [];
    state.valLoss = Array.isArray(saved.valLoss) ? saved.valLoss : [];
    state.valPpl = Array.isArray(saved.valPpl) ? saved.valPpl : [];
    state.phase = saved.phase || null;
  } catch (_) {}
}

/* ─── Initialize system ──────────────────────────────────────── */
async function browseFolder() {
  try {
    const res  = await fetch(`${API}/browse-folder`);
    const data = await res.json();
    if (data.path) {
      document.getElementById('f-kbpath').value = data.path;
      localStorage.setItem('nero-kbpath', data.path);
    }
    document.getElementById('browse-hint').style.display = 'none';
  } catch {
    // Server not running — reveal the hint and let the user type the path manually
    document.getElementById('browse-hint').style.display = 'block';
  }
}

/* ─── Cache Mode ─────────────────────────────────────────────── */
const cacheConfig = {
  enabled:    false,
  ttlSeconds: 1800, // 30 minutes default
};

function _loadCacheConfigFromStorage() {
  const enabled = localStorage.getItem('nero-cache-enabled') === '1';
  const ttl     = parseInt(localStorage.getItem('nero-cache-ttl')) || 1800;
  cacheConfig.enabled    = enabled;
  cacheConfig.ttlSeconds = ttl;
  if (enabled) {
    const cb = document.getElementById('cache-enabled');
    if (cb) cb.checked = true;
    onCacheToggle(true, /*silent*/ true);
    _applyTTLToInputs(ttl);
    updateCacheTTLDisplay();
  }
}

function onCacheToggle(enabled, silent = false) {
  cacheConfig.enabled = enabled;
  localStorage.setItem('nero-cache-enabled', enabled ? '1' : '0');

  const settings    = document.getElementById('cache-settings');
  const disabledHint = document.getElementById('cache-disabled-hint');
  if (settings)     settings.style.display     = enabled ? 'block' : 'none';
  if (disabledHint) disabledHint.style.display  = enabled ? 'none'  : 'block';

  if (!silent) addLog('system', `Cache mode ${enabled ? 'enabled' : 'disabled'}`);
}

function updateCacheTTLDisplay() {
  const val    = parseInt(document.getElementById('cache-ttl-value')?.value) || 30;
  const unit   = parseInt(document.getElementById('cache-ttl-unit')?.value)  || 60;
  const totalS = val * unit;
  cacheConfig.ttlSeconds = totalS;
  localStorage.setItem('nero-cache-ttl', totalS);

  const el = document.getElementById('cache-ttl-display');
  if (el) el.textContent = `= ${_fmtTTL(totalS)}`;
}

function _fmtTTL(s) {
  if (s < 60)   return `${s}s`;
  if (s < 3600) return `${Math.round(s / 60)} min`;
  return `${(s / 3600 % 1 === 0 ? s / 3600 : (s / 3600).toFixed(1))} hr`;
}

function _applyTTLToInputs(ttlSeconds) {
  let val, unit;
  if (ttlSeconds % 3600 === 0 && ttlSeconds >= 3600) { val = ttlSeconds / 3600; unit = '3600'; }
  else if (ttlSeconds % 60 === 0 && ttlSeconds >= 60)  { val = ttlSeconds / 60;   unit = '60';   }
  else                                                  { val = ttlSeconds;        unit = '1';    }
  const vEl = document.getElementById('cache-ttl-value');
  const uEl = document.getElementById('cache-ttl-unit');
  if (vEl) vEl.value = val;
  if (uEl) uEl.value = unit;
}

/* ─── Timing Test ────────────────────────────────────────────── */
const TEST_PROMPTS = [
  'What is supervised learning?',
  'Explain gradient descent briefly.',
  'Define overfitting in machine learning.',
  'What is a neural network activation function?',
  'Describe backpropagation in one sentence.',
];

async function runCacheTimingTest() {
  const btn     = document.getElementById('btn-timing-test');
  const results = document.getElementById('cache-test-results');
  results.style.display = 'block';

  if (btn) { btn.disabled = true; btn.textContent = '⏳ Testing…'; }

  const statusEl = document.getElementById('ctr-status');
  const timesEl  = document.getElementById('ctr-times');
  const avgEl    = document.getElementById('ctr-avg');
  const recEl    = document.getElementById('ctr-recommendation');
  const drEl     = document.getElementById('ctr-deep-review');

  timesEl.innerHTML = '';
  avgEl.textContent = '';
  recEl.textContent = '';
  recEl.className   = 'ctr-recommendation';
  drEl.innerHTML    = '';

  const times = [];

  for (let i = 0; i < TEST_PROMPTS.length; i++) {
    statusEl.textContent = `Running test ${i + 1} / ${TEST_PROMPTS.length}…`;
    const t0 = performance.now();
    let ok = false;

    try {
      const res = await fetch(`${API}/ask`, {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body:    JSON.stringify({ question: TEST_PROMPTS[i], require_original: false }),
      });
      ok = res.ok;
    } catch (_) { ok = false; }

    const ms = performance.now() - t0;

    if (ok) {
      times.push(ms);
      timesEl.innerHTML +=
        `<div class="ctr-time-row">
           <span>Test ${i + 1} — <em style="color:var(--muted);font-size:.78rem;">${TEST_PROMPTS[i]}</em></span>
           <span class="ctr-time-val">${(ms / 1000).toFixed(2)}s</span>
         </div>`;
    } else {
      timesEl.innerHTML +=
        `<div class="ctr-time-row error">
           <span>Test ${i + 1}</span>
           <span class="ctr-time-val">API unreachable</span>
         </div>`;
    }

    await new Promise(r => setTimeout(r, 400));
  }

  if (times.length === 0) {
    statusEl.textContent = 'Could not reach the API — is the server running and initialized?';
    if (btn) { btn.disabled = false; btn.textContent = '↻ Retry Test'; }
    return;
  }

  const avg = times.reduce((a, b) => a + b, 0) / times.length;
  const min = Math.min(...times);
  const max = Math.max(...times);
  const std = Math.sqrt(times.map(t => (t - avg) ** 2).reduce((a, b) => a + b, 0) / times.length);

  statusEl.textContent = `Done — ${times.length}/${TEST_PROMPTS.length} succeeded`;
  avgEl.textContent    =
    `Avg ${(avg / 1000).toFixed(2)}s  ·  min ${(min / 1000).toFixed(2)}s  ·  max ${(max / 1000).toFixed(2)}s  ·  ±${(std / 1000).toFixed(2)}s`;

  const rec = _buildRecommendation(avg, min, max, std, times.length);

  recEl.textContent = `Recommended TTL: ${_fmtTTL(rec.ttlSeconds)}  —  ${rec.tier}`;
  recEl.className   = `ctr-recommendation ctr-rec-${rec.cssKey}`;
  drEl.innerHTML    = _renderDeepReview(rec, avg, min, max, std, times.length);

  if (btn) { btn.disabled = false; btn.textContent = '↻ Re-run Test'; }
}

/* ─── Recommendation Engine ──────────────────────────────────── */
function _buildRecommendation(avgMs, minMs, maxMs, stdMs, n) {
  const avgS = avgMs / 1000;
  const stdS = stdMs / 1000;
  const cv   = stdS / avgS; // coefficient of variation

  let ttlSeconds, tier, cssKey, rationale, advice;

  if (avgS < 1) {
    ttlSeconds = 300; tier = 'Low Value'; cssKey = 'low-value';
    rationale = 'Responses are very fast (< 1 s). The overhead of a cache lookup may rival the savings. A short TTL keeps things tidy without much benefit.';
    advice    = 'Consider skipping caching entirely, or set a short TTL as insurance against rare slow responses.';
  } else if (avgS < 3) {
    ttlSeconds = 900; tier = 'Moderate'; cssKey = 'moderate';
    rationale = 'Responses take 1–3 s. Caching gives a noticeable speedup for repeated or similar questions during a training phase.';
    advice    = 'A 15-minute TTL covers most intra-phase repetition without stale-response risk.';
  } else if (avgS < 8) {
    ttlSeconds = 1800; tier = 'High Value'; cssKey = 'high-value';
    rationale = 'Responses take 3–8 s each. Caching meaningfully reduces training wall-time and API costs.';
    advice    = '30 minutes is a solid default — covers an entire phase with room for model warm-up variance.';
  } else if (avgS < 20) {
    ttlSeconds = 3600; tier = 'Critical'; cssKey = 'critical';
    rationale = 'Responses are slow (8–20 s). Without caching, repeated similar questions burn significant time. Caching is strongly recommended.';
    advice    = '1 hour ensures cached answers survive across phase restarts and brief pauses.';
  } else {
    ttlSeconds = 7200; tier = 'Essential'; cssKey = 'essential';
    rationale = 'Responses exceed 20 s. Training without a cache will be severely bottlenecked by API latency.';
    advice    = '2 hours is the minimum effective TTL. Consider 4–8 hours if your training sessions run overnight.';
  }

  // Variance penalty — high variability → extend TTL for stability
  if (cv > 0.4) {
    ttlSeconds = Math.round(ttlSeconds * 1.5);
    rationale += ` High response variability (CV=${cv.toFixed(2)}) detected — TTL extended by 50 % for stability.`;
  }

  return { ttlSeconds, tier, cssKey, rationale, advice, cv, avgS, stdS };
}

function _renderDeepReview(rec, avgMs, minMs, maxMs, stdMs, n) {
  const { ttlSeconds, tier, rationale, advice, cv, avgS, stdS } = rec;
  const costLabel = avgS > 15 ? 'Very High' : avgS > 7 ? 'High' : avgS > 2 ? 'Medium' : 'Low';
  const cvLabel   = cv < 0.15 ? 'Very stable' : cv < 0.30 ? 'Stable' : cv < 0.50 ? 'Moderate variance' : 'High variance';

  const varNote = cv > 0.4
    ? `<li>High variability (CV ${cv.toFixed(2)}) — responses fluctuate strongly; a longer TTL guards against slow outliers.</li>`
    : '';
  const costNote = avgS > 5
    ? `<li>At ${avgS.toFixed(1)} s/prompt, uncached training would cost ~<strong>${Math.round(3600 / avgS)} API calls/hr</strong>.</li>`
    : '';

  return `
    <div class="deep-review-box">
      <div class="dr-title">Deep Review</div>

      <div class="dr-grid">
        <div class="dr-stat">
          <div class="dr-label">Avg Latency</div>
          <div class="dr-val">${(avgMs / 1000).toFixed(2)} s</div>
        </div>
        <div class="dr-stat">
          <div class="dr-label">Std Dev</div>
          <div class="dr-val">±${(stdMs / 1000).toFixed(2)} s  <span style="font-size:.75rem;color:var(--muted)">${cvLabel}</span></div>
        </div>
        <div class="dr-stat">
          <div class="dr-label">API Cost Impact</div>
          <div class="dr-val">${costLabel}</div>
        </div>
        <div class="dr-stat">
          <div class="dr-label">Samples</div>
          <div class="dr-val">${n} / ${TEST_PROMPTS.length}</div>
        </div>
      </div>

      <div class="dr-rationale">${rationale}</div>

      <div class="dr-breakdown-title">Recommendation Reasoning</div>
      <ul class="dr-list">
        <li>Each prompt takes ~<strong>${(avgMs / 1000).toFixed(2)} s</strong> — caching eliminates this cost on repeated questions.</li>
        <li>Range: <strong>${(minMs / 1000).toFixed(2)} s – ${(maxMs / 1000).toFixed(2)} s</strong> across ${n} tests.</li>
        ${varNote}
        ${costNote}
        <li>${advice}</li>
      </ul>

      <div class="dr-actions">
        <button class="btn btn-sm btn-primary"
                onclick="_applyTTLToInputs(${ttlSeconds}); updateCacheTTLDisplay();">
          Apply ${_fmtTTL(ttlSeconds)} Recommendation
        </button>
      </div>
    </div>`;
}

async function initializeSystem() {
  // Prefer model config state; fall back to hidden legacy fields
  const apiKey  = modelConfig.apiKey  || document.getElementById('f-apikey').value.trim();
  const baseUrl = modelConfig.baseUrl || document.getElementById('f-baseurl').value.trim();
  const model   = modelConfig.modelId || document.getElementById('f-model').value.trim();
  const topicsRaw = document.getElementById('f-topics').value.trim();
  const kbPath    = document.getElementById('f-kbpath').value.trim() || './knowledge_base';
  const threshold = parseInt(document.getElementById('f-threshold').value, 10);
  state.persistenceKey = _sessionKey(kbPath);

  const msg = document.getElementById('init-msg');

  if (!apiKey || !baseUrl || !model) {
    msg.textContent = 'API Key, Base URL, and Model ID are required.';
    msg.className   = 'init-msg error';
    return;
  }

  msg.textContent = 'Initializing…';
  msg.className   = 'init-msg pending';

  const payload = {
    teacher_api_key:           apiKey,
    teacher_base_url:          baseUrl,
    teacher_model:             model,
    topics_description:        topicsRaw.trim(),
    question_repeat_threshold: isNaN(threshold) ? 75 : threshold,
    knowledge_base_path:       kbPath,
  };

  try {
    const res  = await fetch(`${API}/initialize`, {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify(payload),
    });
    const data = await res.json();

    if (!res.ok) {
      msg.textContent = data.detail || 'Initialization failed.';
      msg.className   = 'init-msg error';
      return;
    }

    state.initialized = true;
    msg.textContent   = `Initialized — ${data.knowledge_base?.checkpoints_count ?? 0} existing checkpoint(s) loaded.`;
    msg.className     = 'init-msg success';

    document.getElementById('init-state').textContent = 'Initialized';
    document.getElementById('init-state').className   = 'init-state ready';
    document.getElementById('btn-start').disabled     = false;
    document.getElementById('btn-stop').disabled      = false;

    setStatus('Ready', 'idle');
    await loadTrainingHistory();
    refreshMetrics();
  } catch (e) {
    msg.textContent = 'Cannot reach NERO API. Is the server running?';
    msg.className   = 'init-msg error';
  }
}

async function loadTrainingHistory() {
  try {
    const res = await fetch(`${API}/training/history`);
    if (!res.ok) return;
    const data = await res.json();
    if (!Array.isArray(data.entries) || data.entries.length === 0) return;

    state.steps = [];
    state.trainLoss = [];
    state.valSteps = [];
    state.valLoss = [];
    state.valPpl = [];

    data.entries.forEach(entry => {
      if (entry.step == null) return;
      state.steps.push(entry.step);
      state.trainLoss.push(entry.loss ?? null);
      if (entry.val_loss != null) {
        state.valSteps.push(entry.step);
        state.valLoss.push(entry.val_loss);
        state.valPpl.push(entry.val_perplexity);
      }
      if (entry.phase) state.phase = entry.phase;
    });
    state.historyLoaded = true;
    renderPhases();
    renderCharts();
    _persistState();
  } catch (_) {}
}

/* ─── Training controls ──────────────────────────────────────── */
async function startTraining() {
  try {
    const res  = await fetch(`${API}/train/start`, { method: 'POST' });
    const data = await res.json();
    if (res.ok) {
      state.training = true;
      setStatus('Training', 'active');
      addLog('system', 'Training started');
    } else {
      addLog('error', data.detail || 'Could not start training.');
    }
  } catch (_) {
    addLog('error', 'API unreachable.');
  }
}

async function pauseTraining() {
  try {
    await fetch(`${API}/train/stop`, { method: 'POST' });
  } catch (_) {}
  state.training = false;
  setStatus('Paused', 'paused');
  // Preserve full state to both localStorage and Firestore before halting
  _persistState();
  _scheduleFirestoreWrite();
  addLog('system', 'Training paused — state preserved');
}

// Keep old name as alias so any other callers still work
const stopTraining = pauseTraining;

/* ─── Download phase output ──────────────────────────────────── */
async function downloadPhaseOutput() {
  try {
    let report = null;
    try {
      const r = await fetch(`${API}/logs/report`);
      if (r.ok) report = await r.json();
    } catch (_) {}

    const output = {
      phase:             state.phase,
      training_complete: state.trainingComplete,
      final_metrics: {
        total_steps:   state.steps.length,
        last_step:     state.steps.at(-1)    ?? null,
        last_loss:     state.trainLoss.at(-1) ?? null,
        last_val_loss: state.valLoss.at(-1)   ?? null,
        last_val_ppl:  state.valPpl.at(-1)    ?? null,
      },
      report:      report,
      exported_at: new Date().toISOString(),
    };

    const blob = new Blob([JSON.stringify(output, null, 2)], { type: 'application/json' });
    const url  = URL.createObjectURL(blob);
    const a    = document.createElement('a');
    a.href     = url;
    a.download = `nero-phase-output-${Date.now()}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
    addLog('system', 'Phase output downloaded as JSON');
  } catch (e) {
    addLog('error', 'Download failed: ' + e.message);
  }
}

async function exportKnowledge() {
  try {
    const res  = await fetch(`${API}/knowledge/export`, { method: 'POST' });
    const data = await res.json();
    addLog(res.ok ? 'system' : 'error', res.ok ? `Knowledge exported to: ${data.path}` : 'Export failed.');
  } catch (_) {
    addLog('error', 'Export request failed.');
  }
}

/* ─── Phases ─────────────────────────────────────────────────── */
function renderPhases() {
  const container = document.getElementById('phases');
  container.innerHTML = PHASE_DEFS.map(p => {
    const isActive  = state.phase === p.key;
    const isDone    = state.phase && PHASE_DEFS.findIndex(x => x.key === state.phase) >
                      PHASE_DEFS.findIndex(x => x.key === p.key);
    const statusCls = isActive ? 'phase-active' : isDone ? 'phase-done' : 'phase-pending';
    const badge     = isActive ? 'Active' : isDone ? 'Done' : 'Pending';

    return `
      <div class="phase ${statusCls}" style="border-left-color:${p.color}">
        <div class="phase-badge" style="background:${p.color}22; color:${p.color}">${badge}</div>
        <h4 style="color:${p.color}">${p.label}</h4>
        <p>${p.desc}</p>
      </div>`;
  }).join('');
}

/* ─── Charts ─────────────────────────────────────────────────── */
let chartsReady = { loss: false, ppl: false };

function renderCharts() {
  const base = {
    paper_bgcolor: 'transparent',
    plot_bgcolor:  'transparent',
    font:          { family: '-apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif', size: 12 },
    margin:        { l: 48, r: 16, t: 16, b: 40 },
    showlegend:    true,
    legend:        { x: 0, y: 1, orientation: 'h', bgcolor: 'transparent' },
  };

  const xaxis = { title: 'Optimizer Step', gridcolor: '#e4e6ef', zeroline: false };
  const yaxis = { gridcolor: '#e4e6ef', zeroline: false };

  // Loss chart
  Plotly.newPlot('chart-loss', [
    {
      x: state.steps, y: state.trainLoss,
      name: 'Train Loss', type: 'scatter', mode: 'lines',
      line: { color: '#FF6B6B', width: 2 },
      fill: 'tozeroy', fillcolor: 'rgba(255,107,107,0.08)',
    },
    {
      x: state.valSteps, y: state.valLoss,
      name: 'Val Loss', type: 'scatter', mode: 'lines+markers',
      line: { color: '#4ECDC4', width: 2, dash: 'dot' },
      marker: { size: 5 },
    },
  ], {
    ...base, xaxis: { ...xaxis }, yaxis: { ...yaxis, title: 'Cross-Entropy Loss' },
    height: 260,
  }, { responsive: true });

  // Perplexity chart
  Plotly.newPlot('chart-ppl', [
    {
      x: state.valSteps, y: state.valPpl,
      name: 'Val Perplexity', type: 'scatter', mode: 'lines+markers',
      line: { color: '#667eea', width: 2 },
      fill: 'tozeroy', fillcolor: 'rgba(102,126,234,0.08)',
      marker: { size: 5 },
    },
  ], {
    ...base, xaxis: { ...xaxis }, yaxis: { ...yaxis, title: 'Perplexity' },
    height: 220,
  }, { responsive: true });

  chartsReady = { loss: true, ppl: true };
}

function extendCharts(step, trainLoss, valLoss, valPpl) {
  if (!chartsReady.loss) return;

  // Always extend train loss
  Plotly.extendTraces('chart-loss', {
    x: [[step]], y: [[trainLoss]],
  }, [0]);

  // Val metrics only arrive on validation steps
  if (valLoss != null) {
    Plotly.extendTraces('chart-loss', { x: [[step]], y: [[valLoss]] }, [1]);
    Plotly.extendTraces('chart-ppl',  { x: [[step]], y: [[valPpl]]  }, [0]);
  }

  // Prune to last 200 points
  const MAX = 200;
  if (state.steps.length > MAX) {
    Plotly.relayout('chart-loss', {
      'xaxis.range': [state.steps[state.steps.length - MAX], state.steps.at(-1)],
    });
  }
}

/* ─── Live training log ──────────────────────────────────────── */
function addLog(type, message) {
  const ts    = new Date().toLocaleTimeString();
  const entry = { ts, type, message };
  state.logEntries.push(entry);
  if (state.logEntries.length > 200) state.logEntries.shift();

  const panel = document.getElementById('log-panel');
  const div   = document.createElement('div');
  div.className = `log-entry log-${type}`;
  div.innerHTML = `<span class="log-ts">${ts}</span><span class="log-msg">${message}</span>`;
  panel.appendChild(div);
  panel.scrollTop = panel.scrollHeight;
}

function clearLog() {
  state.logEntries = [];
  document.getElementById('log-panel').innerHTML = '';
}

/* ─── WebSocket ──────────────────────────────────────────────── */
function connectWS() {
  const ws = new WebSocket(WS);

  ws.onopen  = () => addLog('system', 'WebSocket connected');

  ws.onmessage = e => {
    const m = JSON.parse(e.data);

    // Update state arrays
    if (m.step != null) {
      state.steps.push(m.step);
      state.trainLoss.push(m.loss ?? null);
      if (m.val_loss != null) {
        state.valSteps.push(m.step);
        state.valLoss.push(m.val_loss);
        state.valPpl.push(m.val_perplexity);
      }
      _persistState();
      // Throttled Firestore write (debounced 2 s)
      _scheduleFirestoreWrite();
    }

    // Phase tracking
    if (m.phase && m.phase !== state.phase) {
      state.phase = m.phase;
      renderPhases();
      addLog('phase', `Phase changed → ${m.phase}`);
      _writePhaseEvent(m.phase, m.step);
    }

    // Q&A bank events — write snapshot whenever bank_count changes
    if (m.bank_count != null && m.bank_count !== state._lastBankCount) {
      state._lastBankCount = m.bank_count;
      _writeQAEvent(m.bank_count, m.step);
    }

    // Metric cards
    if (m.step      != null) setMetric('m-step',  m.step);
    if (m.loss      != null) setMetric('m-loss',  fmt(m.loss, 4));
    if (m.val_loss  != null) {
      setMetric('m-vloss', fmt(m.val_loss, 4));
      setMetric('m-ppl',   fmt(m.val_perplexity, 1));
    }
    if (m.bank_count != null) {
      setMetric('m-bank', m.bank_count);
      setMetric('m-toss', m.toss_count);
      const pct = m.repeat_threshold > 0
        ? Math.min(100, (m.toss_count / m.repeat_threshold) * 100).toFixed(1)
        : 0;
      setMetric('m-toss-d', `${pct}% of threshold (${m.repeat_threshold})`);
      // Update Logs summary bar
      _updateLogsSummary(m.bank_count, m.toss_count, m.repeat_threshold, m.training_complete);
    }

    // Training status (generating / training on batch)
    if (m.training_status && m.training_status !== state.lastTrainingStatus) {
      state.lastTrainingStatus = m.training_status;
      addLog('system', m.training_status);
    }

    // Current topic banner
    const topicEl = document.getElementById('topic-text');
    if (topicEl) {
      if (m.current_topics && m.current_topics.length) {
        topicEl.textContent = m.current_topics.join(' · ');
      } else if (m.training_status) {
        topicEl.textContent = m.training_status;
      }
    }

    // Training error
    if (m.training_error && m.training_error !== state.lastTrainingError) {
      state.lastTrainingError = m.training_error;
      addLog('error', `Training crashed: ${m.training_error.split('\n')[0]}`);
      setStatus('Error', 'error');
    }

    // Training complete
    if (m.training_complete && !state.trainingComplete) {
      state.trainingComplete = true;
      state.training = false;
      setStatus('Complete', 'complete');
      document.getElementById('complete-banner').style.display = 'flex';
      // Show conditional Download button now that phase is complete
      const dlBtn = document.getElementById('btn-download');
      if (dlBtn) dlBtn.style.display = 'inline-block';
      addLog('phase', 'Training complete — repeat threshold reached');
      loadReport();
      // Final Firestore persist on completion
      _scheduleFirestoreWrite();
    }

    // Extend charts if monitor tab visible
    if (document.getElementById('tab-monitor').classList.contains('active')) {
      extendCharts(m.step, m.loss, m.val_loss, m.val_perplexity);
    }

    // Log val events
    if (m.val_loss != null) {
      addLog('val', `step ${m.step} — val_loss ${fmt(m.val_loss, 4)} · ppl ${fmt(m.val_perplexity, 1)}`);
    }

    document.getElementById('last-updated').textContent = `Live · ${new Date().toLocaleTimeString()}`;
  };

  ws.onclose = () => {
    addLog('system', 'WebSocket disconnected — reconnecting in 3 s…');
    setTimeout(connectWS, 3000);
  };

  ws.onerror = () => ws.close();
}

/* ─── Refresh static metrics ─────────────────────────────────── */
async function refreshMetrics() {
  try {
    const [hRes, kbRes] = await Promise.all([
      fetch(`${API}/health`),
      fetch(`${API}/knowledge/summary`).catch(() => null),
    ]);

    const health = await hRes.json();

    if (!state.initialized && (health.brain_loaded || health.teacher_loaded)) {
      // Server was already initialized (e.g. page reload)
      state.initialized = true;
      document.getElementById('init-state').textContent = 'Initialized';
      document.getElementById('init-state').className   = 'init-state ready';
      document.getElementById('btn-start').disabled     = false;
      document.getElementById('btn-stop').disabled      = false;
      await loadTrainingHistory();
    }

    state.training = health.training_active;
    setStatus(
      health.training_active ? 'Training' :
      health.brain_loaded    ? 'Ready'    : 'Not initialized',
      health.training_active ? 'active'   : 'idle'
    );

    if (kbRes && kbRes.ok) {
      const kb = await kbRes.json();
      setMetric('m-ckpt', kb.checkpoints_count ?? 0,
        kb.latest_checkpoint ? `Latest: ${kb.latest_checkpoint.name}` : '');
    }

    document.getElementById('last-updated').textContent =
      `Updated ${new Date().toLocaleTimeString()}`;
  } catch (_) { /* server offline — keep stale values */ }
}

/* ─── Knowledge tab ──────────────────────────────────────────── */
async function loadKBFolders() {
  try {
    const res = await fetch(`${API}/knowledge/summary`);
    if (!res.ok) throw new Error();
    const kb = await res.json();

    document.getElementById('kb-folders').innerHTML = [
      { icon: '📁', name: 'checkpoints',   detail: `${kb.checkpoints_count ?? 0} files` },
      { icon: '📁', name: 'embeddings',    detail: `${kb.embeddings_count  ?? 0} sets`  },
      { icon: '📁', name: 'training_data', detail: `${kb.training_sessions ?? 0} sessions` },
      { icon: '📁', name: 'metadata',      detail: `${(kb.total_size_mb ?? 0).toFixed(2)} MB total` },
    ].map(f => `
      <div class="folder-item">
        ${f.icon} <strong>${f.name}</strong>
        <div class="folder-meta">${f.detail}</div>
      </div>
    `).join('');

    const latest = kb.latest_checkpoint;
    document.getElementById('kb-latest').innerHTML = latest ? `
      <div class="ckpt-card">
        <div class="ckpt-name">${latest.name ?? 'checkpoint'}</div>
        <div class="ckpt-meta">Loss: ${fmt(latest.loss, 4)}</div>
        <div class="ckpt-meta">Val Loss: ${fmt(latest.val_loss, 4)}</div>
        <div class="ckpt-meta">Val Perplexity: ${fmt(latest.val_perplexity, 1)}</div>
        <div class="ckpt-meta">Phase: ${latest.phase ?? '—'}</div>
        <div class="ckpt-meta">Step: ${latest.step ?? '—'}</div>
      </div>
    ` : '<p class="muted">No checkpoints yet.</p>';

  } catch (_) {
    document.getElementById('kb-folders').innerHTML =
      '<p class="muted">Knowledge manager not initialized.</p>';
  }
}

/* ─── Logs tab ───────────────────────────────────────────────── */

function _ts(epoch) {
  if (!epoch) return '—';
  return new Date(epoch * 1000).toLocaleTimeString();
}

function _truncate(text, max = 80) {
  return text && text.length > max ? text.slice(0, max) + '…' : (text || '—');
}

function _updateLogsSummary(bankCount, tossCount, threshold, complete) {
  const el = id => document.getElementById(id);
  if (el('ls-bank'))      el('ls-bank').textContent      = bankCount ?? '—';
  if (el('ls-toss'))      el('ls-toss').textContent      = tossCount ?? '—';
  if (el('ls-threshold')) el('ls-threshold').textContent = threshold ?? '—';
  if (el('ls-status'))    el('ls-status').textContent    = complete ? 'Complete' : (state.training ? 'Training' : 'Paused');

  if (threshold > 0 && tossCount != null) {
    const pct  = Math.min(100, (tossCount / threshold) * 100);
    const fill = document.getElementById('toss-fill');
    const lbl  = document.getElementById('toss-pct');
    if (fill) {
      fill.style.width = pct + '%';
      fill.className   = 'toss-fill' + (pct >= 100 ? ' toss-fill-done' : pct >= 75 ? ' toss-fill-warn' : '');
    }
    if (lbl) lbl.textContent = `${tossCount} / ${threshold} (${pct.toFixed(1)}%)`;
  }
}

async function loadLogs() {
  try {
    const [bankRes, tossRes] = await Promise.all([
      fetch(`${API}/logs/question-bank`),
      fetch(`${API}/logs/toss-log`),
    ]);

    if (!bankRes.ok || !tossRes.ok) {
      addLog('error', 'Could not load logs — system may not be initialized.');
      return;
    }

    const bank = await bankRes.json();
    const toss = await tossRes.json();

    _updateLogsSummary(bank.count, toss.count, toss.threshold, state.trainingComplete);

    document.getElementById('bank-count-lbl').textContent = bank.count;
    document.getElementById('toss-count-lbl').textContent = toss.count;

    // Question Bank table — most recent first, cap 500
    const bankRows = [...bank.entries].reverse().slice(0, 500);
    document.getElementById('bank-tbody').innerHTML = bankRows.length
      ? bankRows.map(e => `
          <tr>
            <td><code>${e.id}</code></td>
            <td title="${e.question}">${_truncate(e.question)}</td>
            <td>${_ts(e.timestamp)}</td>
          </tr>`).join('')
      : '<tr><td colspan="3" class="muted" style="text-align:center;padding:.75rem;">Empty</td></tr>';

    // Toss Log table — most recent first, cap 500
    const tossRows = [...toss.entries].reverse().slice(0, 500);
    document.getElementById('toss-tbody').innerHTML = tossRows.length
      ? tossRows.map(e => `
          <tr>
            <td title="${e.question}">${_truncate(e.question)}</td>
            <td><code>${e.matched_id}</code></td>
            <td>${_ts(e.timestamp)}</td>
          </tr>`).join('')
      : '<tr><td colspan="3" class="muted" style="text-align:center;padding:.75rem;">Empty</td></tr>';

  } catch (_) {
    addLog('error', 'Failed to fetch logs.');
  }
}

async function loadReport() {
  try {
    const res  = await fetch(`${API}/logs/report`);
    const data = await res.json();
    if (!res.ok || data.status === 'in_progress') return;

    const card = document.getElementById('final-report-card');
    const body = document.getElementById('final-report-body');
    if (!card || !body) return;

    const topicsHtml = data.topics_covered
      ? Object.entries(data.topics_covered)
          .map(([t, n]) => `
            <div class="report-topic">
              <span>${_truncate(t, 60)}</span>
              <strong>${n}</strong>
            </div>`).join('')
      : '<p class="muted">No topic data.</p>';

    body.innerHTML = `
      <div class="report-stats">
        <div class="report-stat">
          <div class="rs-value">${data.questions_asked}</div>
          <div class="rs-label">Questions Asked</div>
        </div>
        <div class="report-stat">
          <div class="rs-value">${data.questions_tossed}</div>
          <div class="rs-label">Questions Tossed</div>
        </div>
        <div class="report-stat">
          <div class="rs-value">${Object.keys(data.topics_covered || {}).length}</div>
          <div class="rs-label">Domains Covered</div>
        </div>
      </div>
      <div style="margin-top:1.1rem;">
        <div class="report-section-title">Topics Breakdown</div>
        ${topicsHtml}
      </div>
      <div style="margin-top:1rem; font-size:.8rem; color:var(--muted);">
        Generated: ${data.generated_at ? new Date(data.generated_at * 1000).toLocaleString() : '—'}
      </div>`;

    card.style.display = 'block';
  } catch (_) {}
}

/* ─── Test AI ────────────────────────────────────────────────── */
async function askAI() {
  const q    = document.getElementById('question-input').value.trim();
  const orig = document.getElementById('require-original').checked;
  if (!q) return;

  const area = document.getElementById('response-area');
  area.style.display = 'block';
  document.getElementById('response-text').textContent = 'Thinking…';
  ['r-conf','r-orig','r-tok'].forEach(id => document.getElementById(id).textContent = '—');

  try {
    const res  = await fetch(`${API}/ask`, {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({ question: q, require_original: orig }),
    });

    if (!res.ok) {
      const err = await res.json();
      document.getElementById('response-text').textContent = err.detail || 'API error.';
      return;
    }

    const data = await res.json();
    document.getElementById('response-text').textContent = data.answer || '(empty response)';
    document.getElementById('r-conf').textContent  =
      data.confidence != null ? `${(data.confidence * 100).toFixed(1)}%` : '—';
    document.getElementById('r-orig').textContent  = data.is_original ? 'Yes' : 'No';
    document.getElementById('r-tok').textContent   = data.tokens_used ?? '—';
  } catch (_) {
    document.getElementById('response-text').textContent =
      'Cannot reach NERO API. Is the server running?';
  }
}

/* ─── Themes ─────────────────────────────────────────────────── */
function toggleThemePanel() {
  document.getElementById('theme-panel').classList.toggle('open');
}

function setTheme(name) {
  document.documentElement.setAttribute('data-theme', name);
  localStorage.setItem('nero-theme', name);
  document.querySelectorAll('.swatch').forEach(s => {
    s.classList.toggle('active', s.dataset.t === name);
  });
}

function applyFilter(type, value) {
  const num = parseFloat(value) / 100;
  const varName = type === 'brightness' ? '--f-brightness'
                : type === 'contrast'   ? '--f-contrast'
                :                         '--f-exposure';
  document.documentElement.style.setProperty(varName, num);
  const label = document.getElementById(`lbl-${type}`);
  if (label) label.textContent = `${value}%`;
  localStorage.setItem(`nero-filter-${type}`, value);
}

function resetFilters() {
  ['brightness', 'contrast', 'exposure'].forEach(type => {
    const slider = document.getElementById(`sl-${type}`);
    if (slider) { slider.value = 100; }
    const varName = type === 'brightness' ? '--f-brightness'
                  : type === 'contrast'   ? '--f-contrast'
                  :                         '--f-exposure';
    document.documentElement.style.removeProperty(varName);
    const label = document.getElementById(`lbl-${type}`);
    if (label) label.textContent = '100%';
    localStorage.removeItem(`nero-filter-${type}`);
  });
}

function initTheme() {
  const saved = localStorage.getItem('nero-theme') || 'nordic';
  setTheme(saved);
  // Restore saved form fields
  const fields = ['f-apikey', 'f-baseurl', 'f-model', 'f-kbpath', 'f-threshold', 'f-topics'];
  fields.forEach(id => {
    const val = localStorage.getItem('nero-' + id);
    const el  = document.getElementById(id);
    if (val !== null && el) el.value = val;
  });
  ['brightness', 'contrast', 'exposure'].forEach(type => {
    const stored = localStorage.getItem(`nero-filter-${type}`);
    if (stored !== null) {
      const slider = document.getElementById(`sl-${type}`);
      if (slider) { slider.value = stored; }
      applyFilter(type, stored);
    }
  });
}

/* ─── Boot ───────────────────────────────────────────────────── */
(async function init() {
  // 1. Firebase must come first (PIN fetch depends on it)
  initFirebase();

  // 2. PIN screen — blocks UI via overlay; app init continues behind it
  initPINScreen(); // async, no await needed — PIN check is non-blocking

  // 3. Theme + saved model config + cache config
  initTheme();
  _loadModelConfigFromStorage();
  _loadCacheConfigFromStorage();

  // 4. Derive session key and restore local state
  state.persistenceKey = _sessionKey(localStorage.getItem('nero-f-kbpath') || '');
  _restoreState();

  // 5. Restore richer state from Firestore (overwrites localStorage if more data exists)
  await _restoreFromFirestore();

  // 6. Auto-save training form fields to localStorage
  ['f-kbpath', 'f-threshold', 'f-topics'].forEach(id => {
    const el = document.getElementById(id);
    if (el) el.addEventListener('input', () => localStorage.setItem('nero-' + id, el.value));
  });

  // 7. Auto-save model config fields to localStorage + state as user types
  [['mc-apikey', 'nero-mc-apikey'], ['mc-baseurl', 'nero-mc-baseurl'], ['mc-model', 'nero-mc-model']].forEach(([id, key]) => {
    const el = document.getElementById(id);
    if (el) el.addEventListener('input', () => localStorage.setItem(key, el.value));
  });

  renderPhases();
  await refreshMetrics();
  connectWS();
  setInterval(refreshMetrics, 15_000);
})();
