"""
ML Traffic Monitor — live dashboard.

Serves a single-page HTML dashboard at http://0.0.0.0:8080/
Auto-refreshes every 2 s via JS fetch.

Data sources (PostgreSQL):
  traffic_responses  — all traffic (passed + blocked by any layer)
  protection_events  — detailed block events (source: ml_proxy | nemesida_waf)
  blocked_requests   — summary of blocked requests
"""

import json
import logging
import os
from contextlib import asynccontextmanager

import asyncpg
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse

logger = logging.getLogger("monitor")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

_pool = None


async def get_pool():
    global _pool
    if _pool is None:
        dsn = (
            f"postgresql://{os.getenv('DB_USER', 'vtsk')}"
            f":{os.getenv('DB_PASSWORD', '1234')}"
            f"@{os.getenv('DB_HOST', 'postgres')}"
            f":{os.getenv('DB_PORT', '5432')}"
            f"/{os.getenv('DB_NAME', 'vtsk_db')}"
        )
        _pool = await asyncpg.create_pool(dsn, min_size=1, max_size=5)
    return _pool


@asynccontextmanager
async def lifespan(app: FastAPI):
    await get_pool()
    logger.info("Monitor ready")
    yield
    if _pool:
        await _pool.close()


app = FastAPI(title="ML Traffic Monitor", lifespan=lifespan)

# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------

@app.get("/api/stats")
async def api_stats():
    pool = await get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow("""
            SELECT
                COUNT(*)                                                          AS total,
                COUNT(*) FILTER (WHERE NOT was_blocked)                           AS passed,
                COUNT(*) FILTER (WHERE was_blocked AND blocked_by = 'ml_proxy')   AS ml_blocked,
                COUNT(*) FILTER (WHERE was_blocked AND blocked_by != 'ml_proxy')  AS waf_blocked,
                ROUND(AVG(response_time_ms)::numeric, 2)                          AS avg_latency_ms
            FROM traffic_responses
        """)
        total     = row["total"]     or 0
        passed    = row["passed"]    or 0
        ml_blk    = row["ml_blocked"]  or 0
        waf_blk   = row["waf_blocked"] or 0
        avg_lat   = float(row["avg_latency_ms"] or 0)

    return {
        "total":       total,
        "passed":      passed,
        "ml_blocked":  ml_blk,
        "waf_blocked": waf_blk,
        "avg_latency_ms": avg_lat,
        "pass_pct":    round(passed    / total * 100, 1) if total else 0,
        "ml_pct":      round(ml_blk    / total * 100, 1) if total else 0,
        "waf_pct":     round(waf_blk   / total * 100, 1) if total else 0,
    }


@app.get("/api/timeline")
async def api_timeline():
    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch("""
            SELECT
                to_char(date_trunc('minute', received_at), 'HH24:MI') AS minute,
                COUNT(*)                                                AS total,
                COUNT(*) FILTER (WHERE NOT was_blocked)                AS passed,
                COUNT(*) FILTER (WHERE was_blocked AND blocked_by = 'ml_proxy')  AS ml_blocked,
                COUNT(*) FILTER (WHERE was_blocked AND blocked_by != 'ml_proxy') AS waf_blocked
            FROM traffic_responses
            WHERE received_at >= NOW() - INTERVAL '30 minutes'
            GROUP BY date_trunc('minute', received_at)
            ORDER BY date_trunc('minute', received_at)
        """)
    return {
        "labels":      [r["minute"]      for r in rows],
        "passed":      [r["passed"]      for r in rows],
        "ml_blocked":  [r["ml_blocked"]  for r in rows],
        "waf_blocked": [r["waf_blocked"] for r in rows],
    }


@app.get("/api/ml-events")
async def api_ml_events():
    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch("""
            SELECT
                to_char(timestamp, 'HH24:MI:SS') AS time,
                details,
                session_id
            FROM protection_events
            WHERE source = 'ml_proxy'
            ORDER BY timestamp DESC
            LIMIT 30
        """)
    events = []
    for r in rows:
        try:
            d = json.loads(r["details"] or "{}")
        except Exception:
            d = {}
        events.append({
            "time":          r["time"],
            "sender":        d.get("sender", "?"),
            "amount":        d.get("amount", 0),
            "anomaly_label": d.get("anomaly_label", "?"),
            "prob":          d.get("prob", 0),
            "session_id":    r["session_id"] or "",
        })
    return events


@app.get("/api/waf-events")
async def api_waf_events():
    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch("""
            SELECT
                to_char(timestamp, 'HH24:MI:SS') AS time,
                source,
                details,
                severity
            FROM protection_events
            WHERE source != 'ml_proxy'
            ORDER BY timestamp DESC
            LIMIT 30
        """)
    return [
        {
            "time":     r["time"],
            "source":   r["source"],
            "details":  r["details"] or "",
            "severity": r["severity"] or "medium",
        }
        for r in rows
    ]


# ---------------------------------------------------------------------------
# Dashboard HTML
# ---------------------------------------------------------------------------

DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="ru">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>ML Traffic Monitor</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

    :root {
      --bg:        #0d1117;
      --surface:   #161b22;
      --border:    #30363d;
      --text:      #c9d1d9;
      --muted:     #8b949e;
      --green:     #3fb950;
      --red:       #f85149;
      --orange:    #d29922;
      --blue:      #58a6ff;
      --purple:    #bc8cff;
    }

    body {
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
      background: var(--bg);
      color: var(--text);
      min-height: 100vh;
    }

    /* ---- header ---- */
    header {
      background: var(--surface);
      border-bottom: 1px solid var(--border);
      padding: 16px 28px;
      display: flex;
      align-items: center;
      justify-content: space-between;
    }
    header h1 {
      font-size: 18px;
      font-weight: 600;
      letter-spacing: .5px;
    }
    header h1 span { color: var(--blue); }
    #last-updated { font-size: 12px; color: var(--muted); }
    .dot {
      display: inline-block; width: 8px; height: 8px;
      border-radius: 50%; background: var(--green);
      margin-right: 6px;
      animation: pulse 2s infinite;
    }
    @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.4} }

    /* ---- stat cards ---- */
    .stats-row {
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 16px;
      padding: 20px 28px 0;
    }
    .stat-card {
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 20px 24px;
      position: relative;
      overflow: hidden;
    }
    .stat-card::before {
      content: '';
      position: absolute; top: 0; left: 0; right: 0; height: 3px;
    }
    .stat-card.total::before  { background: var(--blue); }
    .stat-card.passed::before { background: var(--green); }
    .stat-card.ml::before     { background: var(--purple); }
    .stat-card.waf::before    { background: var(--red); }

    .stat-label { font-size: 12px; color: var(--muted); text-transform: uppercase; letter-spacing: .8px; }
    .stat-value { font-size: 36px; font-weight: 700; margin: 8px 0 4px; line-height: 1; }
    .stat-card.total  .stat-value { color: var(--blue); }
    .stat-card.passed .stat-value { color: var(--green); }
    .stat-card.ml     .stat-value { color: var(--purple); }
    .stat-card.waf    .stat-value { color: var(--red); }
    .stat-pct { font-size: 13px; color: var(--muted); }
    .stat-sub { font-size: 12px; color: var(--muted); margin-top: 4px; }

    /* ---- chart ---- */
    .chart-section {
      margin: 20px 28px 0;
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 20px 24px;
    }
    .section-title {
      font-size: 13px;
      font-weight: 600;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: .8px;
      margin-bottom: 16px;
    }
    .chart-wrap { height: 200px; }

    /* ---- event feeds ---- */
    .feeds {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 16px;
      margin: 20px 28px 28px;
    }
    .feed {
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 10px;
      overflow: hidden;
    }
    .feed-header {
      padding: 14px 18px;
      border-bottom: 1px solid var(--border);
      display: flex;
      align-items: center;
      gap: 10px;
    }
    .feed-header h3 { font-size: 13px; font-weight: 600; }
    .badge {
      font-size: 11px; font-weight: 600; padding: 2px 8px;
      border-radius: 12px; letter-spacing: .4px;
    }
    .badge-ml  { background: rgba(188,140,255,.15); color: var(--purple); border: 1px solid rgba(188,140,255,.3); }
    .badge-waf { background: rgba(248,81,73,.15);   color: var(--red);    border: 1px solid rgba(248,81,73,.3); }

    .feed table { width: 100%; border-collapse: collapse; font-size: 12px; }
    .feed th {
      padding: 8px 12px;
      text-align: left;
      font-size: 11px;
      font-weight: 600;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: .6px;
      background: rgba(255,255,255,.02);
      border-bottom: 1px solid var(--border);
    }
    .feed td {
      padding: 8px 12px;
      border-bottom: 1px solid rgba(48,54,61,.6);
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
      max-width: 180px;
    }
    .feed tr:last-child td { border-bottom: none; }
    .feed tr:hover td { background: rgba(255,255,255,.025); }

    .tag {
      display: inline-block; padding: 2px 7px; border-radius: 4px;
      font-size: 11px; font-weight: 600;
    }
    .tag-vs  { background: rgba(248,81,73,.15);   color: var(--red);    }
    .tag-as  { background: rgba(210,153,34,.15);  color: var(--orange); }
    .tag-ap  { background: rgba(210,153,34,.15);  color: var(--orange); }
    .tag-aa  { background: rgba(210,153,34,.15);  color: var(--orange); }
    .tag-sql { background: rgba(248,81,73,.15);   color: var(--red);    }
    .tag-xss { background: rgba(210,153,34,.15);  color: var(--orange); }
    .tag-pt  { background: rgba(88,166,255,.15);  color: var(--blue);   }
    .tag-cmd { background: rgba(248,81,73,.15);   color: var(--red);    }
    .tag-ok  { background: rgba(63,185,80,.15);   color: var(--green);  }

    .prob-bar {
      display: inline-block; height: 6px; border-radius: 3px;
      vertical-align: middle; margin-left: 6px;
    }

    .no-data { padding: 32px; text-align: center; color: var(--muted); font-size: 13px; }

    @media (max-width: 900px) {
      .stats-row { grid-template-columns: 1fr 1fr; }
      .feeds { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>

<header>
  <h1><span class="dot"></span>ML Traffic <span>Monitor</span></h1>
  <span id="last-updated">—</span>
</header>

<!-- Stats -->
<div class="stats-row">
  <div class="stat-card total">
    <div class="stat-label">Всего запросов</div>
    <div class="stat-value" id="s-total">—</div>
    <div class="stat-sub" id="s-latency">avg latency —</div>
  </div>
  <div class="stat-card passed">
    <div class="stat-label">Пропущено ✓</div>
    <div class="stat-value" id="s-passed">—</div>
    <div class="stat-pct" id="s-pass-pct">— %</div>
  </div>
  <div class="stat-card ml">
    <div class="stat-label">Заблокировано ML 🤖</div>
    <div class="stat-value" id="s-ml">—</div>
    <div class="stat-pct" id="s-ml-pct">— %</div>
  </div>
  <div class="stat-card waf">
    <div class="stat-label">Заблокировано WAF 🛡️</div>
    <div class="stat-value" id="s-waf">—</div>
    <div class="stat-pct" id="s-waf-pct">— %</div>
  </div>
</div>

<!-- Timeline -->
<div class="chart-section">
  <div class="section-title">Трафик за последние 30 минут</div>
  <div class="chart-wrap">
    <canvas id="timeline-chart"></canvas>
  </div>
</div>

<!-- Event feeds -->
<div class="feeds">

  <!-- ML blocks -->
  <div class="feed">
    <div class="feed-header">
      <h3>ML блокировки (CatBoost)</h3>
      <span class="badge badge-ml">BEHAVIORAL</span>
    </div>
    <table>
      <thead>
        <tr>
          <th>Время</th>
          <th>Отправитель</th>
          <th>Сумма ₽</th>
          <th>Тип аномалии</th>
          <th>P(anomaly)</th>
        </tr>
      </thead>
      <tbody id="ml-tbody">
        <tr><td colspan="5" class="no-data">Ожидание данных…</td></tr>
      </tbody>
    </table>
  </div>

  <!-- WAF blocks -->
  <div class="feed">
    <div class="feed-header">
      <h3>WAF блокировки (сигнатуры)</h3>
      <span class="badge badge-waf">SIGNATURES</span>
    </div>
    <table>
      <thead>
        <tr>
          <th>Время</th>
          <th>Источник</th>
          <th>Тип атаки</th>
          <th>Детали</th>
        </tr>
      </thead>
      <tbody id="waf-tbody">
        <tr><td colspan="4" class="no-data">Ожидание данных…</td></tr>
      </tbody>
    </table>
  </div>

</div>

<script>
// ---- Chart setup ----
const ctx = document.getElementById('timeline-chart').getContext('2d');
const chart = new Chart(ctx, {
  type: 'bar',
  data: {
    labels: [],
    datasets: [
      { label: 'Пропущено',       data: [], backgroundColor: 'rgba(63,185,80,.7)',   stack: 'a' },
      { label: 'ML заблок.',      data: [], backgroundColor: 'rgba(188,140,255,.7)', stack: 'a' },
      { label: 'WAF заблок.',     data: [], backgroundColor: 'rgba(248,81,73,.7)',   stack: 'a' },
    ]
  },
  options: {
    responsive: true,
    maintainAspectRatio: false,
    animation: false,
    plugins: {
      legend: { labels: { color: '#8b949e', boxWidth: 12, font: { size: 12 } } }
    },
    scales: {
      x: { stacked: true, ticks: { color: '#8b949e', maxTicksLimit: 15 }, grid: { color: 'rgba(48,54,61,.5)' } },
      y: { stacked: true, ticks: { color: '#8b949e' },                    grid: { color: 'rgba(48,54,61,.5)' } }
    }
  }
});

// ---- Helpers ----
function fmt(n) {
  if (n >= 1e6) return (n/1e6).toFixed(1) + 'M';
  if (n >= 1e3) return (n/1e3).toFixed(1) + 'K';
  return n;
}

const ANOMALY_TAG = {
  velocity_spike:            ['tag-vs',  'velocity spike'],
  amount_spike_exponential:  ['tag-as',  'amount spike (exp)'],
  amount_spike_poisson:      ['tag-ap',  'amount spike (poi)'],
  amount_spike_pareto:       ['tag-aa',  'amount spike (par)'],
};

function anomalyTag(label) {
  const [cls, text] = ANOMALY_TAG[label] || ['tag-ok', label || '?'];
  return `<span class="tag ${cls}">${text}</span>`;
}

function probBar(p) {
  const w = Math.round(p * 60);
  const c = p >= .8 ? '#f85149' : p >= .5 ? '#d29922' : '#3fb950';
  return `<span style="color:${c};font-weight:600;">${p.toFixed(3)}</span>`
       + `<span class="prob-bar" style="width:${w}px;background:${c};opacity:.6;"></span>`;
}

function attackTag(details) {
  const d = details.toLowerCase();
  if (d.includes('sql'))    return '<span class="tag tag-sql">SQL Inj</span>';
  if (d.includes('xss'))    return '<span class="tag tag-xss">XSS</span>';
  if (d.includes('path'))   return '<span class="tag tag-pt">Path Trav</span>';
  if (d.includes('cmd'))    return '<span class="tag tag-cmd">Cmd Inj</span>';
  if (d.includes('ua:'))    return '<span class="tag tag-vs">Bad UA</span>';
  if (d.includes('ip'))     return '<span class="tag tag-ap">Spoofed IP</span>';
  return '<span class="tag tag-ok">' + details.split(':')[0] + '</span>';
}

// ---- Data fetch ----
async function updateStats() {
  try {
    const d = await fetch('/api/stats').then(r => r.json());
    document.getElementById('s-total').textContent   = fmt(d.total);
    document.getElementById('s-passed').textContent  = fmt(d.passed);
    document.getElementById('s-ml').textContent      = fmt(d.ml_blocked);
    document.getElementById('s-waf').textContent     = fmt(d.waf_blocked);
    document.getElementById('s-pass-pct').textContent = d.pass_pct + ' %';
    document.getElementById('s-ml-pct').textContent   = d.ml_pct  + ' %';
    document.getElementById('s-waf-pct').textContent  = d.waf_pct + ' %';
    document.getElementById('s-latency').textContent  = 'avg ' + d.avg_latency_ms + ' ms';
  } catch(e) { console.error(e); }
}

async function updateTimeline() {
  try {
    const d = await fetch('/api/timeline').then(r => r.json());
    chart.data.labels              = d.labels;
    chart.data.datasets[0].data   = d.passed;
    chart.data.datasets[1].data   = d.ml_blocked;
    chart.data.datasets[2].data   = d.waf_blocked;
    chart.update('none');
  } catch(e) { console.error(e); }
}

async function updateMLEvents() {
  try {
    const rows = await fetch('/api/ml-events').then(r => r.json());
    const tb = document.getElementById('ml-tbody');
    if (!rows.length) { tb.innerHTML = '<tr><td colspan="5" class="no-data">Нет блокировок</td></tr>'; return; }
    tb.innerHTML = rows.map(r =>
      `<tr>
        <td style="color:var(--muted)">${r.time}</td>
        <td style="color:var(--blue)">${r.sender}</td>
        <td>${r.amount.toLocaleString('ru')}</td>
        <td>${anomalyTag(r.anomaly_label)}</td>
        <td>${probBar(r.prob)}</td>
      </tr>`
    ).join('');
  } catch(e) { console.error(e); }
}

async function updateWAFEvents() {
  try {
    const rows = await fetch('/api/waf-events').then(r => r.json());
    const tb = document.getElementById('waf-tbody');
    if (!rows.length) { tb.innerHTML = '<tr><td colspan="4" class="no-data">Нет блокировок</td></tr>'; return; }
    tb.innerHTML = rows.map(r =>
      `<tr>
        <td style="color:var(--muted)">${r.time}</td>
        <td style="color:var(--red)">${r.source}</td>
        <td>${attackTag(r.details)}</td>
        <td style="color:var(--muted);font-size:11px;max-width:200px;overflow:hidden;text-overflow:ellipsis">${r.details.substring(0,50)}</td>
      </tr>`
    ).join('');
  } catch(e) { console.error(e); }
}

function tick() {
  const now = new Date();
  document.getElementById('last-updated').textContent =
    'обновлено ' + now.toLocaleTimeString('ru');
  updateStats();
  updateMLEvents();
  updateWAFEvents();
}

// timeline refreshes a bit less often
setInterval(updateTimeline, 5000);
setInterval(tick, 2000);

// initial load
tick();
updateTimeline();
</script>

</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
async def dashboard():
    return DASHBOARD_HTML
