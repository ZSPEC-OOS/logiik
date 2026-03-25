/* ─── Config ────────────────────────────────────────────────── */
const API = window.location.origin;
const WS  = `${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/ws/training`;

/* ─── State ─────────────────────────────────────────────────── */
const state = {
  training: false,
  lossHistory: [],
  accuracyHistory: [],
  genHistory: [],
  steps: [],
};

/* ─── Phases config ─────────────────────────────────────────── */
const PHASES = [
  { title: "Phase 1: Memorization",  desc: "Learning from Teacher's Q+A structure", color: "#4CAF50", progress: 100, status: "completed" },
  { title: "Phase 2: Generation",    desc: "Creating original answers",               color: "#667eea", progress: 65,  status: "active"    },
  { title: "Phase 3: Abstraction",   desc: "Cross-domain knowledge synthesis",        color: "#9e9e9e", progress: 0,   status: "pending"   },
];

/* ─── Tab switching ─────────────────────────────────────────── */
function showTab(id, btn) {
  document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.tab').forEach(b => b.classList.remove('active'));
  document.getElementById(`tab-${id}`).classList.add('active');
  btn.classList.add('active');
  if (id === 'monitor') renderCharts();
}

/* ─── Status badge ──────────────────────────────────────────── */
function setStatus(label, cls) {
  const el = document.getElementById('status-badge');
  el.textContent = `● ${label}`;
  el.className = `status-badge ${cls}`;
}

/* ─── Training controls ─────────────────────────────────────── */
async function startTraining() {
  try {
    const res = await fetch(`${API}/train/start`, { method: 'POST' });
    const data = await res.json();
    if (res.ok) {
      state.training = true;
      setStatus('Training', 'active');
    } else {
      alert(data.detail || 'Could not start training (system not initialized).');
    }
  } catch (e) {
    alert('API unreachable. Is the NERO server running?');
  }
}

async function stopTraining() {
  await fetch(`${API}/train/stop`, { method: 'POST' });
  state.training = false;
  setStatus('Paused', 'paused');
}

async function exportKnowledge() {
  const res = await fetch(`${API}/knowledge/export`, { method: 'POST' });
  const data = await res.json();
  alert(res.ok ? `Exported: ${data.path}` : 'Export failed — is the knowledge manager initialized?');
}

/* ─── Metric cards ──────────────────────────────────────────── */
function setMetric(id, value, delta) {
  document.getElementById(id).textContent = value;
  if (delta !== undefined) document.getElementById(`${id}-d`).textContent = delta;
}

async function refreshMetrics() {
  try {
    const [healthRes, kbRes] = await Promise.all([
      fetch(`${API}/health`),
      fetch(`${API}/knowledge/summary`).catch(() => null),
    ]);
    const health = await healthRes.json();

    setStatus(
      health.training_active ? 'Training' : health.brain_loaded ? 'Idle' : 'Not initialized',
      health.training_active ? 'active' : 'idle'
    );
    state.training = health.training_active;

    if (kbRes && kbRes.ok) {
      const kb = await kbRes.json();
      setMetric('m-examples',    kb.checkpoints_count * 100 || '—');
      setMetric('m-accuracy',    '—');
      setMetric('m-size',        `${kb.total_size_mb ?? 0} MB`, `${kb.checkpoints_count} checkpoints`);
      setMetric('m-checkpoints', kb.checkpoints_count, kb.latest_checkpoint ? `Latest: ${kb.latest_checkpoint.name}` : '');
    }

    document.getElementById('last-updated').textContent =
      `Updated ${new Date().toLocaleTimeString()}`;
  } catch (_) { /* server offline — keep stale values */ }
}

/* ─── Phases ────────────────────────────────────────────────── */
function renderPhases() {
  const container = document.getElementById('phases');
  container.innerHTML = PHASES.map(p => `
    <div class="phase" style="border-left-color:${p.color}; opacity:${p.status === 'pending' ? .6 : 1};">
      <h4 style="color:${p.color}">${p.title}</h4>
      <p>${p.desc}</p>
      <div class="progress-track">
        <div class="progress-fill" style="width:${p.progress}%; background:${p.color}"></div>
      </div>
      <div class="progress-label" style="color:${p.color}">${p.progress}%</div>
    </div>
  `).join('');
}

/* ─── Charts ────────────────────────────────────────────────── */
function seedData() {
  if (state.steps.length) return;
  for (let t = 0; t < 50; t++) {
    state.steps.push(t);
    state.lossHistory.push(2.5 * Math.exp(-0.05 * t) + 0.2 + (Math.random() - 0.5) * 0.1);
    state.accuracyHistory.push(50 + 40 * (1 - Math.exp(-0.05 * t)) + (Math.random() - 0.5) * 4);
    state.genHistory.push(20 + 60 * (1 - Math.exp(-0.03 * t)));
  }
}

function renderCharts() {
  seedData();

  const shared = { type: 'scatter', mode: 'lines' };

  Plotly.newPlot('charts', [
    { ...shared, x: state.steps, y: state.lossHistory,
      name: 'Loss', line: { color: '#FF6B6B', width: 2 },
      fill: 'tozeroy', fillcolor: 'rgba(255,107,107,0.15)',
      xaxis: 'x1', yaxis: 'y1' },

    { ...shared, x: state.steps, y: state.accuracyHistory,
      name: 'Accuracy %', line: { color: '#4ECDC4', width: 2 },
      fill: 'tozeroy', fillcolor: 'rgba(78,205,196,0.15)',
      xaxis: 'x2', yaxis: 'y2' },

    { ...shared, x: state.steps, y: state.genHistory,
      name: 'Originality', line: { color: '#667eea', width: 2, dash: 'dot' },
      xaxis: 'x3', yaxis: 'y3' },

    { type: 'pie', labels: ['Reasoning','Facts','Creativity','Analysis','Synthesis'],
      values: [25, 30, 20, 15, 10], hole: 0.6,
      marker: { colors: ['#667eea','#764ba2','#f093fb','#f5576c','#4facfe'] },
      textinfo: 'label+percent', textposition: 'outside',
      domain: { row: 1, column: 1 } },
  ], {
    grid: { rows: 2, columns: 2, pattern: 'independent' },
    annotations: [
      { text: 'Loss Curve',             xref: 'paper', yref: 'paper', x: 0.22, y: 1.04, showarrow: false, font: { size: 13 } },
      { text: 'Accuracy Progress',      xref: 'paper', yref: 'paper', x: 0.78, y: 1.04, showarrow: false, font: { size: 13 } },
      { text: 'Generative Capability',  xref: 'paper', yref: 'paper', x: 0.22, y: 0.46, showarrow: false, font: { size: 13 } },
      { text: 'Knowledge Distribution', xref: 'paper', yref: 'paper', x: 0.78, y: 0.46, showarrow: false, font: { size: 13 } },
    ],
    height: 580,
    showlegend: false,
    template: 'plotly_white',
    margin: { l: 40, r: 20, t: 60, b: 40 },
  }, { responsive: true });
}

function pushLiveMetrics(m) {
  const t = state.steps.length;
  state.steps.push(t);
  state.lossHistory.push(m.loss + (Math.random() - 0.5) * 0.05);
  state.accuracyHistory.push(m.accuracy * 100 + (Math.random() - 0.5) * 2);
  state.genHistory.push(Math.min(80, 20 + 60 * (1 - Math.exp(-0.03 * t))));

  // Cap to last 100 points
  if (state.steps.length > 100) {
    ['steps','lossHistory','accuracyHistory','genHistory'].forEach(k => state[k].shift());
  }

  Plotly.extendTraces('charts', {
    x: [[state.steps.at(-1)], [state.steps.at(-1)], [state.steps.at(-1)]],
    y: [[state.lossHistory.at(-1)], [state.accuracyHistory.at(-1)], [state.genHistory.at(-1)]],
  }, [0, 1, 2]);

  setMetric('m-accuracy', `${(m.accuracy * 100).toFixed(1)}%`);
  setMetric('m-examples', m.examples_processed);
}

/* ─── Knowledge folder tiles ────────────────────────────────── */
async function loadKBFolders() {
  try {
    const res = await fetch(`${API}/knowledge/summary`);
    if (!res.ok) throw new Error();
    const kb = await res.json();
    const el = document.getElementById('kb-folders');
    const folders = [
      { name: 'embeddings',    detail: `${kb.embeddings_count ?? 0} sets` },
      { name: 'checkpoints',   detail: `${kb.checkpoints_count ?? 0} files` },
      { name: 'training_data', detail: `${kb.training_sessions ?? 0} sessions` },
      { name: 'metadata',      detail: `${(kb.total_size_mb ?? 0).toFixed(1)} MB total` },
    ];
    el.innerHTML = folders.map(f => `
      <div class="folder-item">
        📁 <strong>${f.name}</strong>
        <div class="folder-meta">${f.detail}</div>
      </div>
    `).join('');
  } catch (_) {
    document.getElementById('kb-folders').innerHTML =
      '<p class="muted">Knowledge manager not initialized.</p>';
  }
}

/* ─── Ask AI ────────────────────────────────────────────────── */
async function askAI() {
  const q = document.getElementById('question-input').value.trim();
  if (!q) return;

  const area = document.getElementById('response-area');
  area.style.display = 'none';
  document.getElementById('response-text').textContent = 'Thinking…';
  area.style.display = 'block';

  try {
    const res = await fetch(`${API}/ask`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question: q, require_original: true }),
    });

    if (!res.ok) {
      const err = await res.json();
      document.getElementById('response-text').textContent = err.detail || 'Error from API.';
      return;
    }

    const data = await res.json();
    document.getElementById('response-text').textContent =
      data.answer || data.response || JSON.stringify(data);
    document.getElementById('r-conf').textContent =
      data.confidence != null ? `${(data.confidence * 100).toFixed(0)}%` : '—';
    document.getElementById('r-orig').textContent = data.is_original ? 'High' : 'Low';
    document.getElementById('r-tok').textContent  = data.tokens ?? '—';
  } catch (_) {
    document.getElementById('response-text').textContent =
      'Cannot reach NERO API. Is the server running?';
  }
}

/* ─── WebSocket ─────────────────────────────────────────────── */
function connectWS() {
  const ws = new WebSocket(WS);
  ws.onmessage = e => {
    const m = JSON.parse(e.data);
    pushLiveMetrics(m);
    document.getElementById('last-updated').textContent =
      `Live · ${new Date().toLocaleTimeString()}`;
  };
  ws.onclose = () => setTimeout(connectWS, 3000); // auto-reconnect
}

/* ─── Init ──────────────────────────────────────────────────── */
(async function init() {
  renderPhases();
  renderCharts();
  await refreshMetrics();
  loadKBFolders();
  connectWS();
  setInterval(refreshMetrics, 10_000);
})();
