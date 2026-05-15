'use strict';

// ── Backend configuration ─────────────────────────────────────────────────────
//
// Three backends:
//   local  → FastAPI running on this machine (development)
//   live   → Cloudflare Tunnel from the demo workplace (set URL after tunnel setup)
//   demo   → Render.com cloud deployment (set URL after Render deploy)
//
// Selection order: ?backend=<key> param → hostname detection → 'demo' fallback
//
const BACKENDS = {
  local: `${window.location.protocol}//${window.location.host}`,
  live:  'https://FILL_IN_CLOUDFLARE_TUNNEL_URL',
  demo:  'https://FILL_IN_RENDER_APP_URL',
};

const _backendKey = (() => {
  const param = new URLSearchParams(window.location.search).get('backend');
  if (param && BACKENDS[param]) return param;
  return ['localhost', '127.0.0.1'].includes(window.location.hostname) ? 'local' : 'demo';
})();

const BACKEND_URL = BACKENDS[_backendKey];
const WS_URL = BACKEND_URL.replace(/^http/, 'ws') + '/ws/live';

// ── Constants ────────────────────────────────────────────────────────────────

const COLORS = {
  signalX:    '#0F8B8D',
  signalY:    '#F28F3B',
  rms:        '#2D7DD2',
  movingAvg:  '#35A675',
  jhi:        '#D8434B',
  warning:    '#E8A317',
  critical:   '#D8434B',
};

const STATE_COLORS = {
  NORMAL:   '#23B26D',
  WARNING:  '#E8A317',
  CRITICAL: '#D8434B',
};

const PLOT_DEFAULTS = {
  paper_bgcolor: 'rgba(0,0,0,0)',
  plot_bgcolor:  '#F8FAFB',
  font:          { family: 'Segoe UI, system-ui, sans-serif', size: 12, color: '#18332B' },
  margin:        { t: 16, r: 16, b: 44, l: 56 },
  showlegend:    true,
  legend:        { bgcolor: 'rgba(255,255,255,0.7)', bordercolor: 'rgba(0,0,0,0)', x: 0, y: 1 },
};

const PLOTLY_CONFIG = { responsive: true, displayModeBar: false };

const LIVE_MAX_POINTS = 500;

// ── State ────────────────────────────────────────────────────────────────────

let ws = null;
let wsReconnectTimer = null;

// ── Bootstrap ────────────────────────────────────────────────────────────────

document.addEventListener('DOMContentLoaded', async () => {
  initEmptyCharts();
  await loadMeasurements();
  await loadJoints();
  connectWebSocket();

  document.getElementById('analyze-btn').addEventListener('click', onAnalyzeClick);
  document.getElementById('joint-select').addEventListener('change', onJointChange);
});

// ── Measurements ─────────────────────────────────────────────────────────────

async function loadMeasurements() {
  const res = await fetch(`${BACKEND_URL}/api/measurements`);
  const measurements = await res.json();
  const sel = document.getElementById('measurement-select');
  sel.innerHTML = '';
  measurements.forEach(m => {
    const opt = document.createElement('option');
    opt.value = m.id;
    opt.textContent = m.label;
    sel.appendChild(opt);
  });
}

async function loadJoints() {
  const res = await fetch(`${BACKEND_URL}/api/joints`);
  const joints = await res.json();
  const sel = document.getElementById('joint-select');
  sel.innerHTML = '';
  joints.forEach(j => {
    const opt = document.createElement('option');
    opt.value = j;
    opt.textContent = j;
    sel.appendChild(opt);
  });
}

// ── Analysis ─────────────────────────────────────────────────────────────────

async function onAnalyzeClick() {
  const id = document.getElementById('measurement-select').value;
  if (id === '') return;

  const btn = document.getElementById('analyze-btn');
  btn.textContent = 'Analyzing…';
  btn.disabled = true;

  try {
    const res = await fetch(`${BACKEND_URL}/api/analysis/${id}`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    renderSignalChart(data.signal);
    renderTrendChart(data.trend);
    updateStatusCards(data.status, data.meta);
  } catch (err) {
    console.error('Analysis error:', err);
    alert('Analysis failed. Check the server log.');
  } finally {
    btn.textContent = 'Analyze';
    btn.disabled = false;
  }
}

// ── Signal chart ─────────────────────────────────────────────────────────────

function renderSignalChart(signal) {
  const n = signal.x.length;
  const idx = Array.from({ length: n }, (_, i) => i);

  const traces = [
    {
      x: idx, y: signal.x,
      name: 'Raw X',
      type: 'scatter', mode: 'lines',
      line: { color: COLORS.signalX, width: 1.3 },
    },
    {
      x: idx, y: signal.y,
      name: 'Raw Y',
      type: 'scatter', mode: 'lines',
      line: { color: COLORS.signalY, width: 1.0 },
      opacity: 0.9,
    },
  ];

  const layout = {
    ...PLOT_DEFAULTS,
    xaxis: { title: 'Sample index (downsampled view)', gridcolor: '#e4ece8', zeroline: false },
    yaxis: { title: 'Acceleration [mg]',               gridcolor: '#e4ece8', zeroline: false },
    legend: { ...PLOT_DEFAULTS.legend, x: 0.72, y: 1 },
  };

  Plotly.react('signal-chart', traces, layout, PLOTLY_CONFIG);
}

// ── Trend chart ──────────────────────────────────────────────────────────────

function renderTrendChart(trend) {
  const traces = [
    {
      x: trend.windows, y: trend.rms,
      name: 'RMS vector [mg]',
      type: 'scatter', mode: 'lines',
      line: { color: COLORS.rms, width: 1.8 },
      yaxis: 'y',
    },
    {
      x: trend.windows, y: trend.moving_avg_rms,
      name: 'Moving avg RMS',
      type: 'scatter', mode: 'lines',
      line: { color: COLORS.movingAvg, width: 1.4, dash: 'dash' },
      yaxis: 'y',
    },
    {
      x: trend.windows, y: trend.jhi,
      name: 'Joint Health Index',
      type: 'scatter', mode: 'lines',
      line: { color: COLORS.jhi, width: 2.1 },
      yaxis: 'y2',
    },
  ];

  const layout = {
    ...PLOT_DEFAULTS,
    margin: { ...PLOT_DEFAULTS.margin, r: 64 },
    xaxis:  { title: 'Sliding window', gridcolor: '#e4ece8', zeroline: false },
    yaxis:  { title: 'RMS vector [mg]', gridcolor: '#e4ece8', zeroline: false },
    yaxis2: {
      title: 'Joint Health Index',
      overlaying: 'y', side: 'right',
      range: [0, 100],
      gridcolor: 'rgba(0,0,0,0)',
      zeroline: false,
    },
    shapes: [
      {
        type: 'line', xref: 'paper', x0: 0, x1: 1,
        yref: 'y2', y0: trend.warning_score, y1: trend.warning_score,
        line: { color: COLORS.warning, dash: 'dot', width: 1.4 },
      },
      {
        type: 'line', xref: 'paper', x0: 0, x1: 1,
        yref: 'y2', y0: trend.critical_score, y1: trend.critical_score,
        line: { color: COLORS.critical, dash: 'dot', width: 1.4 },
      },
    ],
  };

  Plotly.react('trend-chart', traces, layout, PLOTLY_CONFIG);
}

// ── Live chart ───────────────────────────────────────────────────────────────

function initEmptyCharts() {
  const emptyLayout = {
    ...PLOT_DEFAULTS,
    xaxis: { title: 'Sample', gridcolor: '#e4ece8', zeroline: false },
    yaxis: { title: 'Acceleration [mg]', gridcolor: '#e4ece8', zeroline: false },
    margin: { ...PLOT_DEFAULTS.margin, t: 10 },
    legend: { ...PLOT_DEFAULTS.legend, x: 0.82, y: 1 },
  };

  // Empty signal / trend placeholders
  Plotly.newPlot('signal-chart', [], { ...emptyLayout, xaxis: { ...emptyLayout.xaxis, title: 'Sample index' } }, PLOTLY_CONFIG);
  Plotly.newPlot('trend-chart',  [], emptyLayout, PLOTLY_CONFIG);

  // Live chart — two pre-allocated traces
  Plotly.newPlot('live-chart', [
    {
      x: [], y: [],
      name: 'X axis',
      type: 'scatter', mode: 'lines',
      line: { color: COLORS.signalX, width: 1.0 },
    },
    {
      x: [], y: [],
      name: 'Y axis',
      type: 'scatter', mode: 'lines',
      line: { color: COLORS.signalY, width: 1.0 },
    },
  ], {
    ...emptyLayout,
    margin: { t: 8, r: 16, b: 40, l: 56 },
  }, PLOTLY_CONFIG);
}

function appendLivePoint(point) {
  Plotly.extendTraces('live-chart', {
    x: [[point.t], [point.t]],
    y: [[point.x], [point.y]],
  }, [0, 1], LIVE_MAX_POINTS);
}

// ── WebSocket ─────────────────────────────────────────────────────────────────

function connectWebSocket() {
  if (ws) {
    ws.onclose = null;
    ws.close();
  }

  const badge = document.getElementById('live-badge');
  ws = new WebSocket(WS_URL);

  ws.onopen = () => {
    badge.textContent = '● LIVE';
    badge.classList.add('connected');
    if (wsReconnectTimer) {
      clearTimeout(wsReconnectTimer);
      wsReconnectTimer = null;
    }
  };

  ws.onmessage = (ev) => {
    try {
      appendLivePoint(JSON.parse(ev.data));
    } catch (_) { /* ignore malformed */ }
  };

  ws.onerror = () => {
    badge.textContent = '○ OFFLINE';
    badge.classList.remove('connected');
  };

  ws.onclose = () => {
    badge.textContent = '○ OFFLINE';
    badge.classList.remove('connected');
    wsReconnectTimer = setTimeout(connectWebSocket, 2500);
  };
}

async function onJointChange() {
  const joint = document.getElementById('joint-select').value;
  await fetch(`${BACKEND_URL}/api/live/joint/${encodeURIComponent(joint)}`, { method: 'POST' });
  // Clear current live trace
  Plotly.react('live-chart', [
    { x: [], y: [], name: 'X axis', type: 'scatter', mode: 'lines', line: { color: COLORS.signalX, width: 1.0 } },
    { x: [], y: [], name: 'Y axis', type: 'scatter', mode: 'lines', line: { color: COLORS.signalY, width: 1.0 } },
  ], {
    ...PLOT_DEFAULTS,
    margin: { t: 8, r: 16, b: 40, l: 56 },
    xaxis: { title: 'Sample', gridcolor: '#e4ece8', zeroline: false },
    yaxis: { title: 'Acceleration [mg]', gridcolor: '#e4ece8', zeroline: false },
    legend: { ...PLOT_DEFAULTS.legend, x: 0.82, y: 1 },
  }, PLOTLY_CONFIG);
}

// ── Status cards ──────────────────────────────────────────────────────────────

function updateStatusCards(status, meta) {
  // Health light + text
  const light = document.getElementById('health-light');
  light.style.background = STATE_COLORS[status.state] ?? '#94a3b8';
  document.getElementById('health-state').textContent = status.state;
  document.getElementById('health-jhi').textContent   = `JHI: ${status.jhi}`;
  document.getElementById('health-rec').textContent   = status.recommendation;

  // Prediction
  document.getElementById('hours-critical').textContent =
    status.hours_to_critical !== null
      ? `${status.hours_to_critical} h`
      : 'Stable / no crossing';

  document.getElementById('op-hours-critical').textContent =
    status.op_hours_to_critical !== null
      ? `${status.op_hours_to_critical} h`
      : 'Historical estimate unavailable';

  document.getElementById('rms-value').textContent  = `${status.current_rms} mg`;
  document.getElementById('freq-value').textContent = `${status.dominant_freq_hz} Hz`;

  // Metadata
  const thresholdLine = (meta.warning_hours !== null && meta.critical_hours !== null)
    ? `<div class="meta-threshold-row">Op. hours to warning / critical: <strong>${meta.warning_hours} h</strong> / <strong>${meta.critical_hours} h</strong></div>`
    : `<div class="meta-threshold-row">Operating-hour thresholds: not enough historical data</div>`;

  document.getElementById('meta-content').innerHTML = [
    row('Joint',         meta.joint),
    row('Date',          meta.date),
    row('Station',       meta.station),
    row('Trajectory',    meta.trajectory),
    row('Sampling rate', `${meta.sampling_rate_hz} Hz`),
    row('Signals',       `${meta.x_column} / ${meta.y_column}`),
    thresholdLine,
  ].join('');
}

function row(key, val) {
  return `<div class="meta-row"><span class="meta-key">${key}</span><span class="meta-val">${val}</span></div>`;
}
