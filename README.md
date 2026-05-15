# SmartSense

**Vibrodiagnostics & predictive maintenance for collaborative robot harmonic drives.**

SmartSense monitors the health of harmonic-drive joints in collaborative robots (cobots) by analysing vibration signals from paired X/Y sensors. It computes a Joint Health Index (JHI), estimates time to critical threshold, and streams live sensor data over WebSocket — all accessible from a web browser.

---

## Live demo

| Mode | URL |
|------|-----|
| Cloud demo (Render) | _coming soon_ |
| Live workplace (Cloudflare Tunnel) | _coming soon_ |

> **GitHub Pages frontend:** https://vojtechmarek13.github.io/SmartSense/
>
> Append `?backend=live` to the URL to connect to the live workplace backend instead of the cloud demo.

---

## Key features

- **Joint Health Index (JHI 0–100)** — weighted scoring from RMS, crest factor, spectral analysis, trend and joint age
- **Predictive maintenance** — linear regression extrapolation estimates hours to critical threshold
- **Live OPC UA streaming** — real-time vibration waveform over WebSocket (simulator included, real `asyncua` client ready to connect)
- **Historical trend calibration** — dynamic warning/critical thresholds adapted from measurement history
- **Dual interface** — PyQt6 desktop dashboard + FastAPI web dashboard served from the same backend

---

## Architecture

```
[Cobot PLC]
    │  OPC UA (asyncua)
    ▼
[FastAPI backend]  ──── REST + WebSocket (wss://) ────►  [GitHub Pages — JS frontend]
    │                                                         │
    ├── /api/measurements      list available CSV datasets    │
    ├── /api/analysis/{id}     full JHI analysis + signal     │
    ├── /api/joints            list joints                    │
    └── /ws/live               live vibration stream          │
                                                              │
                                             ?backend=live  ──┤── Cloudflare Tunnel → workplace PC
                                             (default)      ──┘── Render.com cloud
```

### Signal processing pipeline

```
CSV (AxisX / AxisY)
    └─► VibrationDataLoader
            └─► VibrationFeatureExtractor  (FFT, Hanning window, 2048 samples)
                    └─► JointHealthAnalyzer  (JHI scoring)
                            └─► HistoricalThresholdCalibrator
                                    └─► SmartSenseAnalysisPipeline → API / GUI
```

---

## Tech stack

| Layer | Technology |
|-------|-----------|
| Backend | Python 3.14, FastAPI, Uvicorn, asyncua |
| Signal processing | NumPy, FFT |
| Desktop GUI | PyQt6, Matplotlib |
| ML bridge | PyTorch (placeholder — ready for model integration) |
| Frontend | Vanilla JS, Plotly.js |
| Deployment | GitHub Pages · Cloudflare Tunnel · Render.com |

---

## Local setup

```powershell
# Clone and install
git clone https://github.com/VojtechMarek13/SmartSense.git
cd SmartSense
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e .

# Run web server (http://localhost:8000)
python -m App.web.server

# Run desktop dashboard
python -m App.main
```

Place your measurement data in `data/` following the structure:
```
data/
  Joint {1-5}/
    {DD.MM.YYYY}/
      Cobot Stand/
        Trajectory {1-5}/
          CobotStandAxisX_*.csv
          CobotStandAxisY_*.csv
```

> The `data_demo/` folder (committed) contains truncated sample data (5 000 rows per file) for cloud deployment. Regenerate it after adding new measurements:
> ```powershell
> python scripts/make_demo_data.py
> ```

---

## Deployment

### GitHub Pages (frontend)

Automatically deployed on every push to `main` via `.github/workflows/deploy-pages.yml`.  
Enable in repository **Settings → Pages → Source: GitHub Actions**.

### Cloudflare Tunnel (live workplace backend)

```powershell
# Install cloudflared, then:
cloudflared tunnel login
cloudflared tunnel create smartsense
cloudflared tunnel run --url http://localhost:8000 smartsense
```

Copy the tunnel URL into `BACKENDS.live` in [App/web/static/app.js](App/web/static/app.js).

### Render.com (cloud demo backend)

Connect this repository on [render.com](https://render.com) — `render.yaml` is detected automatically.  
Copy the service URL into `BACKENDS.demo` in [App/web/static/app.js](App/web/static/app.js).

---

## Project status

- Functional heuristic JHI scoring across 5 joints
- 3 measurement dates (Feb–Mar 2026), historical trend calibration active
- OPC UA simulator included; real PLC connection ready to configure
- PyTorch bridge placeholder prepared for future ML model integration

---

## License

Internal research project — JIC (Jihomoravské inovační centrum).
