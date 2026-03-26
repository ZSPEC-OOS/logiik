/* ─── Config ─────────────────────────────────────────────────── */
const API = window.location.origin;
const WS  = `${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/ws/training`;

/* ─── State ──────────────────────────────────────────────────── */
const state = {
  initialized:     false,
  training:        false,
  trainingComplete: false,
  phase:           null,
  steps:           [],
  trainLoss:       [],
  valLoss:         [],
  valPpl:          [],
  valSteps:        [],
  logEntries:      [],
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

/* ─── Initialize system ──────────────────────────────────────── */
async function browseFolder() {
  try {
    const res  = await fetch(`${API}/browse-folder`);
    const data = await res.json();
    if (data.path) {
      document.getElementById('f-kbpath').value = data.path;
      localStorage.setItem('nero-kbpath', data.path);
    }
  } catch {
    alert('Could not open folder picker. Is the NERO server running?');
  }
}

async function initializeSystem() {
  const apiKey    = document.getElementById('f-apikey').value.trim();
  const baseUrl   = document.getElementById('f-baseurl').value.trim();
  const model     = document.getElementById('f-model').value.trim();
  const topicsRaw = document.getElementById('f-topics').value.trim();
  const kbPath    = document.getElementById('f-kbpath').value.trim() || './knowledge_base';
  const threshold = parseInt(document.getElementById('f-threshold').value, 10);

  const msg = document.getElementById('init-msg');

  if (!apiKey || !baseUrl || !model || !topicsRaw.trim()) {
    msg.textContent = 'API Key, Base URL, Model ID, and Topics are required.';
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
    refreshMetrics();
  } catch (e) {
    msg.textContent = 'Cannot reach NERO API. Is the server running?';
    msg.className   = 'init-msg error';
  }
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

async function stopTraining() {
  await fetch(`${API}/train/stop`, { method: 'POST' });
  state.training = false;
  setStatus('Paused', 'paused');
  addLog('system', 'Training stopped');
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
    }

    // Phase tracking
    if (m.phase && m.phase !== state.phase) {
      state.phase = m.phase;
      renderPhases();
      addLog('phase', `Phase changed → ${m.phase}`);
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

    // Training complete
    if (m.training_complete && !state.trainingComplete) {
      state.trainingComplete = true;
      state.training = false;
      setStatus('Complete', 'complete');
      document.getElementById('complete-banner').style.display = 'flex';
      addLog('phase', 'Training complete — repeat threshold reached');
      loadReport();
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
  // Restore saved KB path
  const savedPath = localStorage.getItem('nero-kbpath');
  if (savedPath) {
    const el = document.getElementById('f-kbpath');
    if (el) el.value = savedPath;
  }
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
  initTheme();
  renderPhases();
  await refreshMetrics();
  connectWS();
  setInterval(refreshMetrics, 15_000);
})();
