<div align="center">
  <img src="App/web/static/smartsense_logo.svg" alt="SmartSense" height="80" />
</div>

<p align="center">
  <b>Vibrodiagnostics & predictive maintenance for collaborative robot harmonic drives.</b><br/>
  <a href="https://vojtechmarek13.github.io/SmartSense/">Live Demo</a> &nbsp;·&nbsp;
  <a href="https://vojtechmarek13.github.io/SmartSense/?backend=https://YOUR_TUNNEL.trycloudflare.com">Live Workplace</a>
</p>

---

SmartSense monitors the health of harmonic-drive joints in collaborative robots by analysing paired X/Y vibration signals. It computes a **Joint Health Index (JHI 0–100)**, estimates time to critical threshold, and streams live sensor data over WebSocket — accessible from any browser.

## Features

- **Joint Health Index** — weighted scoring from RMS, crest factor, spectral analysis, trend and operating age
- **Predictive maintenance** — linear regression extrapolation estimates hours remaining to critical threshold
- **Live OPC UA streaming** — real-time vibration waveform over WebSocket (simulator included, real `asyncua` client ready)
- **Historical calibration** — dynamic warning/critical thresholds adapted from measurement history across dates
- **Dual interface** — PyQt6 desktop dashboard + FastAPI web dashboard from the same backend

## Architecture

```
[Cobot PLC] ── OPC UA ──► [FastAPI backend] ── REST + WebSocket (wss://) ──► [GitHub Pages frontend]
                                │
                    ┌───────────┴───────────┐
              Cloudflare Tunnel         Render.com
            (live workplace demo)    (cloud demo 24/7)
```

**Signal pipeline:**
```
CSV (AxisX / AxisY) → VibrationDataLoader → VibrationFeatureExtractor (FFT, Hanning 2048)
  → JointHealthAnalyzer (JHI) → HistoricalThresholdCalibrator → API / GUI
```

## Tech stack

| Layer | Technology |
|-------|-----------|
| Backend | Python 3.14, FastAPI, Uvicorn, asyncua |
| Signal processing | NumPy, FFT |
| Desktop GUI | PyQt6, Matplotlib |
| ML bridge | PyTorch (placeholder — ready for model integration) |
| Frontend | Vanilla JS, Plotly.js |
| Deployment | GitHub Pages · Cloudflare Tunnel · Render.com |

## Local setup

```powershell
git clone https://github.com/VojtechMarek13/SmartSense.git
cd SmartSense
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e .

python -m App.web.server     # web dashboard → http://localhost:8000
python -m App.main           # desktop dashboard
```

Place measurement data in `data/` following:
```
data/Joint {1-5}/{DD.MM.YYYY}/Cobot Stand/Trajectory {1-5}/
  CobotStandAxisX_*.csv
  CobotStandAxisY_*.csv
```

> `data_demo/` (committed) contains truncated 5 000-row samples for cloud deployment.
> Regenerate after adding new measurements: `python scripts/make_demo_data.py`

## Deployment

### GitHub Pages — frontend
Deployed automatically on every push to `main` via `.github/workflows/deploy-pages.yml`.
Enable in **Settings → Pages → Source: GitHub Actions**.

### Cloudflare Tunnel — live workplace backend
```powershell
cloudflared tunnel --url http://127.0.0.1:8000
```
Share the dashboard with the tunnel URL as a parameter:
```
https://vojtechmarek13.github.io/SmartSense/?backend=https://YOUR_TUNNEL.trycloudflare.com
```

### Render.com — cloud demo backend
Connect this repository on [render.com](https://render.com) — `render.yaml` is detected automatically.
Set `BACKENDS.demo` in `App/web/static/app.js` to the Render service URL.

## Data & privacy

Raw measurement CSV files (~1.7 GB) are excluded from the repository via `.gitignore`.
Only `data_demo/` (13 MB, truncated samples) is committed for cloud deployment.

## Project status

- Functional heuristic JHI scoring across 5 joints, 3 measurement dates (Feb–Mar 2026)
- GitHub Pages frontend live, Cloudflare Tunnel tested and working
- OPC UA simulator active; real PLC connection ready to configure via `asyncua`
- PyTorch bridge prepared for future ML model integration

---

*Internal research project — JIC (Jihomoravské inovační centrum)*
