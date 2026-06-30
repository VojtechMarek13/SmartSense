<div align="center">
  <img src="App/web/static/smartsense_logo.svg" alt="SmartSense" height="80" />
</div>

<p align="center">
  <b>Vibrodiagnostics & predictive maintenance for collaborative robot harmonic drives.</b><br/>
  <a href="https://vojtechmarek13.github.io/SmartSense/?backend=https://YOUR_TUNNEL.trycloudflare.com">Live Workplace</a>
</p>

---

SmartSense monitors the health of harmonic-drive joints in collaborative robots by analysing paired X/Y vibration signals. It computes a **Joint Health Index (JHI 0–100)**, estimates time to critical threshold, and streams live sensor data over WebSocket — accessible from any browser.

## Features

- **Joint Health Index** — weighted scoring from RMS, crest factor, spectral analysis, trend and operating age
- **Predictive maintenance with Bayesian uncertainty** — linear regression with delta-method propagation estimates hours to critical threshold + 90 % confidence interval
- **Live OPC UA streaming** — real-time vibration waveform over WebSocket; connects to B&R APC PLC via `asyncua`, falls back to CSV simulator
- **Historical calibration** — dynamic warning/critical thresholds adapted from measurement history across dates
- **CSV upload & ad-hoc analysis** — upload any CSV, pick X/Y columns and sampling rate; system re-calibrates baseline and shows full health analysis in-session without saving to disk
- **Dual interface** — PyQt6 desktop dashboard + FastAPI web dashboard from the same pipeline

## Architecture

```
[B&R APC PLC] ── OPC UA ──► [FastAPI backend] ── REST + WebSocket (wss://) ──► [GitHub Pages frontend]
  192.168.209.35:4840           Python 3.11                                        vojtechmarek13.github.io
  3× gCMCtrl (6 joints)         Cloudflare Tunnel
```

**Signal pipeline:**
```
CSV (AxisX / AxisY) → VibrationDataLoader → VibrationFeatureExtractor (FFT, Hanning 2048)
  → JointHealthAnalyzer (JHI) → HistoricalThresholdCalibrator → API / GUI
```

**OPC UA sensor mapping:**
```
gCMCtrl_1:  [0]=Joint 1 X  [1]=Joint 1 Y  [2]=Joint 2 X  [3]=Joint 2 Y
gCMCtrl_2:  [0]=Joint 3 X  [1]=Joint 3 Y  [2]=Joint 4 X  [3]=Joint 4 Y
gCMCtrl_3:  [0]=Joint 5 X  [1]=Joint 5 Y  [2]=Cobot X    [3]=Cobot Y
```

## Tech stack

| Layer | Technology |
|-------|-----------|
| Backend | Python 3.11, FastAPI, Uvicorn, asyncua |
| Signal processing | NumPy, FFT |
| Desktop GUI | PyQt6, Matplotlib |
| Uncertainty | NumPy delta method — 90 % CI on time-to-critical prediction |
| Frontend | Vanilla JS, Plotly.js |
| Deployment | GitHub Pages · Cloudflare Tunnel |

## Local setup

**Web server** (Python 3.11):
```powershell
git clone https://github.com/VojtechMarek13/SmartSense.git
cd SmartSense
py -3.11 -m venv .venv311
.venv311\Scripts\Activate.ps1
pip install -e .
pip install -r requirements-web.txt

python -m App.web.server          # simulator mode → http://localhost:8000
```

**Desktop GUI** (any Python ≥ 3.11 with PyQt6):
```powershell
python -m App.main
```

**With real PLC** (must be on network 192.168.209.x):
```powershell
$env:OPCUA_MODE = "opcua"
python -m App.web.server
```

Place measurement data in `data/` following:
```
data/Joint {1-5}/{DD.MM.YYYY}/Cobot Stand/Trajectory {1-5}/
  CobotStandAxisX_*.csv
  CobotStandAxisY_*.csv
```

> `data_demo/` (committed) contains truncated 5 000-row samples for the static demo.
> Regenerate after adding new measurements: `python scripts/make_demo_data.py`

## Deployment

### GitHub Pages — frontend
Deployed automatically on every push to `main` via `.github/workflows/deploy-pages.yml`.
Enable in **Settings → Pages → Source: GitHub Actions**.

### Cloudflare Tunnel — live workplace backend
Run in two terminals simultaneously:
```powershell
# Terminal 1 — backend
$env:OPCUA_MODE = "opcua"
python -m App.web.server

# Terminal 2 — tunnel
cloudflared tunnel --url http://127.0.0.1:8000
```
Share the dashboard with the generated tunnel URL:
```
https://vojtechmarek13.github.io/SmartSense/?backend=https://YOUR_TUNNEL.trycloudflare.com
```
> The tunnel URL changes on every restart — update the `?backend=` parameter accordingly.

## Data & privacy

Raw measurement CSV files (~1.7 GB) are excluded from the repository via `.gitignore`.
Only `data_demo/` (13 MB, truncated samples) is committed for the static demo.

## Project status

- Functional heuristic JHI scoring across 5 joints, 3 measurement dates (Feb–Mar 2026)
- GitHub Pages frontend (static sample data), Cloudflare Tunnel for live workplace access
- Real OPC UA client implemented and tested — live data from B&R APC PLC via `asyncua` subscriptions at 20 Hz
- Bayesian uncertainty (delta method) on JHI time-to-critical prediction — 90 % CI displayed in both web and desktop UI
- CSV upload for ad-hoc analysis in both web and desktop dashboard (session-only, no disk storage)
- PyQt6 desktop dashboard visually aligned with the web app (navy/sky-blue design system)

---

*Internal research project — JIC (Jihomoravské inovační centrum)*
