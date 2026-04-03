from flask import Flask, render_template_string, jsonify
from datetime import datetime
import threading
import time
import numpy as np

try:
    from app.analysis.stability_monitor import StabilityMonitor, create_monitor
except ImportError:
    from analysis.stability_monitor import StabilityMonitor, create_monitor

app = Flask(__name__)

monitor: StabilityMonitor = None
simulation_thread = None
is_simulating = False


DASHBOARD_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>ВРПС Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: 'Segoe UI', sans-serif; background: #1a1a2e; color: #eee; }
        .header { background: #16213e; padding: 20px; text-align: center; }
        .header h1 { color: #0f3460; background: linear-gradient(90deg, #e94560, #0f3460); 
                     -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
        .container { display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px; padding: 20px; }
        .card { background: #16213e; border-radius: 10px; padding: 20px; }
        .card h3 { color: #e94560; margin-bottom: 15px; }
        .metric { display: flex; justify-content: space-between; padding: 8px 0; border-bottom: 1px solid #0f3460; }
        .metric-value { font-weight: bold; }
        .status-healthy { color: #4ade80; }
        .status-degraded { color: #fbbf24; }
        .status-critical { color: #ef4444; }
        .mode-badge { display: inline-block; padding: 5px 15px; border-radius: 20px; font-weight: bold; margin: 5px; }
        .mode-1, .mode-2, .mode-3 { background: #4ade80; color: #000; }
        .mode-4, .mode-5, .mode-6 { background: #fbbf24; color: #000; }
        .mode-7, .mode-8, .mode-9 { background: #ef4444; color: #fff; }
        .full-width { grid-column: span 3; }
        .half-width { grid-column: span 2; }
        #vrps-chart, #sust-chart { width: 100%; height: 250px; }
        .controls { text-align: center; padding: 10px; }
        .btn { padding: 10px 20px; margin: 5px; border: none; border-radius: 5px; cursor: pointer; }
        .btn-start { background: #4ade80; }
        .btn-stop { background: #ef4444; color: #fff; }
        .btn-attack { background: #fbbf24; }
    </style>
</head>
<body>
    <div class="header">
        <h1>ВРПС Dashboard - Мониторинг устойчивости платежной системы</h1>
    </div>
    <div class="controls">
        <button class="btn btn-start" onclick="startSimulation()">Старт симуляции</button>
        <button class="btn btn-stop" onclick="stopSimulation()">Стоп</button>
        <button class="btn btn-attack" onclick="simulateAttack()">Симулировать атаку</button>
    </div>
    <div class="container">
        <div class="card">
            <h3>Текущий статус</h3>
            <div id="current-status">
                <div class="metric"><span>Статус:</span><span id="status-value" class="metric-value status-healthy">HEALTHY</span></div>
                <div class="metric"><span>Индекс устойчивости:</span><span id="sust-value" class="metric-value">0.85</span></div>
                <div class="metric"><span>Косинусное сходство:</span><span id="sim-value" class="metric-value">0.95</span></div>
                <div class="metric"><span>В ОСР:</span><span id="osr-value" class="metric-value">✓</span></div>
            </div>
        </div>
        <div class="card">
            <h3>Режим реагирования</h3>
            <div style="text-align: center; padding: 20px;">
                <span id="mode-badge" class="mode-badge mode-1">MODE 1: MONITORING</span>
                <p id="mode-action" style="margin-top: 15px; color: #888;">Действие: pass</p>
                <p id="mode-reason" style="margin-top: 10px; font-size: 12px; color: #666;"></p>
            </div>
        </div>
        <div class="card">
            <h3>Компоненты ВРПС</h3>
            <div class="metric"><span>C (Capacity):</span><span id="c-value" class="metric-value">0.90</span></div>
            <div class="metric"><span>L (Load):</span><span id="l-value" class="metric-value">0.85</span></div>
            <div class="metric"><span>Q (Quality):</span><span id="q-value" class="metric-value">0.92</span></div>
            <div class="metric"><span>R (Resources):</span><span id="r-value" class="metric-value">0.88</span></div>
            <div class="metric"><span>A (Anomaly):</span><span id="a-value" class="metric-value">0.95</span></div>
        </div>
        <div class="card half-width">
            <h3>Траектория ВРПС</h3>
            <canvas id="vrps-chart"></canvas>
        </div>
        <div class="card">
            <h3>Индекс устойчивости</h3>
            <canvas id="sust-chart"></canvas>
        </div>
        <div class="card full-width">
            <h3>Статистика режимов</h3>
            <div id="mode-stats" style="display: flex; justify-content: space-around; flex-wrap: wrap;"></div>
        </div>
    </div>
    <script>
        let vrpsChart, sustChart;
        let updateInterval;
        function initCharts() {
            const vrpsCtx = document.getElementById('vrps-chart').getContext('2d');
            vrpsChart = new Chart(vrpsCtx, {
                type: 'line',
                data: {
                    labels: [],
                    datasets: [
                        { label: 'C', data: [], borderColor: '#ef4444', fill: false },
                        { label: 'L', data: [], borderColor: '#fbbf24', fill: false },
                        { label: 'Q', data: [], borderColor: '#4ade80', fill: false },
                        { label: 'R', data: [], borderColor: '#3b82f6', fill: false },
                        { label: 'A', data: [], borderColor: '#a855f7', fill: false }
                    ]
                },
                options: { responsive: true, scales: { y: { min: 0, max: 1 } }, animation: false }
            });
            const sustCtx = document.getElementById('sust-chart').getContext('2d');
            sustChart = new Chart(sustCtx, {
                type: 'line',
                data: {
                    labels: [],
                    datasets: [{ label: 'Sust', data: [], borderColor: '#e94560', fill: true, backgroundColor: 'rgba(233, 69, 96, 0.2)' }]
                },
                options: { responsive: true, scales: { y: { min: 0, max: 1 } }, animation: false }
            });
        }
        async function updateData() {
            try {
                const response = await fetch('/api/status');
                const data = await response.json();
                if (data.status === 'no_data') return;
                document.getElementById('status-value').textContent = data.sustainability.status.toUpperCase();
                document.getElementById('status-value').className = 'metric-value status-' + data.sustainability.status;
                document.getElementById('sust-value').textContent = data.sustainability.index.toFixed(3);
                document.getElementById('sim-value').textContent = data.similarity.toFixed(3);
                document.getElementById('osr-value').textContent = data.osr.in_osr ? '✓' : '✗ ' + data.osr.violations.join(', ');
                document.getElementById('c-value').textContent = data.vrps.C.toFixed(3);
                document.getElementById('l-value').textContent = data.vrps.L.toFixed(3);
                document.getElementById('q-value').textContent = data.vrps.Q.toFixed(3);
                document.getElementById('r-value').textContent = data.vrps.R.toFixed(3);
                document.getElementById('a-value').textContent = data.vrps.A.toFixed(3);
                const mode = data.decision.mode;
                const modeBadge = document.getElementById('mode-badge');
                modeBadge.textContent = 'MODE ' + mode + ': ' + data.decision.action.toUpperCase();
                modeBadge.className = 'mode-badge mode-' + mode;
                document.getElementById('mode-action').textContent = 'Действие: ' + data.decision.action;
                document.getElementById('mode-reason').textContent = data.decision.reason;
            } catch (e) { console.error(e); }
        }
        async function updateCharts() {
            try {
                const response = await fetch('/api/dashboard');
                const data = await response.json();
                if (!data.timestamps) return;
                vrpsChart.data.labels = data.timestamps.slice(-50);
                vrpsChart.data.datasets[0].data = data.vrps_trajectory.C.slice(-50);
                vrpsChart.data.datasets[1].data = data.vrps_trajectory.L.slice(-50);
                vrpsChart.data.datasets[2].data = data.vrps_trajectory.Q.slice(-50);
                vrpsChart.data.datasets[3].data = data.vrps_trajectory.R.slice(-50);
                vrpsChart.data.datasets[4].data = data.vrps_trajectory.A.slice(-50);
                vrpsChart.update();
                sustChart.data.labels = data.timestamps.slice(-50);
                sustChart.data.datasets[0].data = data.sustainability_index.slice(-50);
                sustChart.update();
                if (data.statistics && data.statistics.mode_distribution) {
                    const statsDiv = document.getElementById('mode-stats');
                    statsDiv.innerHTML = Object.entries(data.statistics.mode_distribution)
                        .map(([mode, count]) => `<span class="mode-badge">${mode}: ${count}</span>`).join('');
                }
            } catch (e) { console.error(e); }
        }
        function startSimulation() {
            fetch('/api/start').then(() => {
                updateInterval = setInterval(() => { updateData(); updateCharts(); }, 1000);
            });
        }
        function stopSimulation() {
            fetch('/api/stop');
            if (updateInterval) clearInterval(updateInterval);
        }
        function simulateAttack() { fetch('/api/attack'); }
        initCharts();
        setInterval(updateData, 2000);
        setInterval(updateCharts, 3000);
    </script>
</body>
</html>
"""


@app.route('/')
def index():
    return render_template_string(DASHBOARD_HTML)


@app.route('/api/status')
def api_status():
    global monitor
    if monitor is None:
        return jsonify({'status': 'no_data'})
    return jsonify(monitor.get_current_status())


@app.route('/api/dashboard')
def api_dashboard():
    global monitor
    if monitor is None:
        return jsonify({})
    return jsonify(monitor.export_for_dashboard())


@app.route('/api/history')
def api_history():
    global monitor
    if monitor is None:
        return jsonify([])
    return jsonify(monitor.get_history(last_n=100))


@app.route('/api/start')
def api_start():
    global monitor, simulation_thread, is_simulating
    if monitor is None:
        monitor = create_monitor(enable_lstm=False, enable_kalman=True)
    is_simulating = True
    def simulate_normal():
        global is_simulating
        while is_simulating:
            monitor.process_metrics(
                T=15 + np.random.normal(0, 3),
                rho=0.3 + np.random.normal(0, 0.05),
                P_block=0.002 + np.random.normal(0, 0.001),
                U_cpu=0.4 + np.random.normal(0, 0.05),
                U_ram=0.5 + np.random.normal(0, 0.05),
                N_anom=int(max(0, np.random.poisson(3))),
                N_bg=100
            )
            time.sleep(1)
    simulation_thread = threading.Thread(target=simulate_normal, daemon=True)
    simulation_thread.start()
    return jsonify({'status': 'started'})


@app.route('/api/stop')
def api_stop():
    global is_simulating
    is_simulating = False
    return jsonify({'status': 'stopped'})


@app.route('/api/attack')
def api_attack():
    global monitor, is_simulating
    if monitor is None:
        return jsonify({'status': 'no_monitor'})
    def simulate_attack():
        for i in range(15):
            if not is_simulating:
                break
            monitor.process_metrics(
                T=100 + i * 30,
                rho=0.6 + i * 0.025,
                P_block=0.02 + i * 0.008,
                U_cpu=0.7 + i * 0.015,
                U_ram=0.65 + i * 0.02,
                N_anom=30 + i * 8,
                N_bg=100
            )
            time.sleep(0.5)
    threading.Thread(target=simulate_attack, daemon=True).start()
    return jsonify({'status': 'attack_started'})


@app.route('/api/reset')
def api_reset():
    global monitor
    if monitor:
        monitor.reset()
    return jsonify({'status': 'reset'})


def run_dashboard(host='0.0.0.0', port=5050, debug=False):
    global monitor
    monitor = create_monitor(enable_lstm=False, enable_kalman=True)
    print(f"\nDashboard running at http://{host}:{port}")
    print("Press Ctrl+C to stop\n")
    app.run(host=host, port=port, debug=debug, threaded=True)


if __name__ == '__main__':
    run_dashboard(debug=True)
