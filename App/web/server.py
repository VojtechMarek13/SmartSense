from __future__ import annotations

import asyncio
import csv
import io
import os
import sys
import uuid
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.health.index import HealthAssessment
from src.opcua.client import JOINT_MAP, OpcUaClient, OpcUaSimulator
from src.pipeline.analysis import DashboardAnalysis, SmartSenseAnalysisPipeline

app = FastAPI(title="SmartSense")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

_static_dir = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=_static_dir), name="static")

_pipeline = SmartSenseAnalysisPipeline()
_opcua_mode = os.environ.get("OPCUA_MODE", "simulator").lower() == "opcua"

_uploads: dict[str, dict] = {}
_MAX_UPLOADS = 10


class _UploadAnalyzeRequest(BaseModel):
    upload_id: str
    x_col: str
    y_col: str
    label: str = "Uploaded"
    sampling_rate_hz: float = 1000.0
_live: OpcUaSimulator | OpcUaClient = (
    OpcUaClient() if _opcua_mode else OpcUaSimulator(_pipeline.loader)
)
_live_task: asyncio.Task | None = None


@app.on_event("startup")
async def _startup() -> None:
    global _live_task
    if isinstance(_live, OpcUaSimulator):
        measurements = _pipeline.list_measurements()
        _live_task = asyncio.create_task(_live.run(measurements))
    else:
        _live_task = asyncio.create_task(_live.run())


@app.on_event("shutdown")
async def _shutdown() -> None:
    if _live_task:
        _live_task.cancel()


@app.get("/", include_in_schema=False)
async def index() -> FileResponse:
    return FileResponse(_static_dir / "index.html")


@app.get("/api/measurements")
async def list_measurements() -> list[dict]:
    return [
        {
            "id": i,
            "label": f"{m.joint} | {m.measurement_date} | {m.station} | {m.trajectory}",
            "joint": m.joint,
            "date": m.measurement_date,
            "station": m.station,
            "trajectory": m.trajectory,
        }
        for i, m in enumerate(_pipeline.list_measurements())
    ]


@app.get("/api/analysis/{measurement_id}")
async def analyze(measurement_id: int) -> dict:
    measurements = _pipeline.list_measurements()
    if measurement_id >= len(measurements):
        raise HTTPException(status_code=404, detail="Measurement not found")
    measurement = measurements[measurement_id]
    analysis = await asyncio.to_thread(_pipeline.analyze_measurement, measurement)
    return _serialize(analysis)


@app.get("/api/joints")
async def list_joints() -> list[str]:
    if isinstance(_live, OpcUaClient):
        return list(JOINT_MAP.keys())
    measurements = _pipeline.list_measurements()
    return sorted({m.joint for m in measurements})


@app.post("/api/live/joint/{joint_name}")
async def set_live_joint(joint_name: str) -> dict:
    _live.set_joint(joint_name)
    return {"joint": joint_name}


@app.websocket("/ws/live")
async def live_stream(websocket: WebSocket) -> None:
    await websocket.accept()
    queue = _live.subscribe()
    try:
        while True:
            point = await queue.get()
            await websocket.send_json(point)
    except (WebSocketDisconnect, Exception):
        pass
    finally:
        _live.unsubscribe(queue)


# ---------------------------------------------------------------------------
# CSV Upload
# ---------------------------------------------------------------------------

@app.post("/api/upload/preview")
async def upload_preview(file: UploadFile = File(...)) -> dict:
    content = await file.read()
    try:
        text = content.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = content.decode("latin-1")

    try:
        dialect = csv.Sniffer().sniff(text[:4096], delimiters=",;\t")
    except csv.Error:
        dialect = csv.excel

    reader = csv.DictReader(io.StringIO(text), dialect=dialect)
    columns = list(reader.fieldnames or [])
    if not columns:
        raise HTTPException(status_code=400, detail="Could not detect CSV columns")

    rows = list(reader)
    if not rows:
        raise HTTPException(status_code=400, detail="CSV file is empty")

    if len(_uploads) >= _MAX_UPLOADS:
        oldest = next(iter(_uploads))
        del _uploads[oldest]

    upload_id = str(uuid.uuid4())[:8]
    _uploads[upload_id] = {"columns": columns, "rows": rows}
    return {"upload_id": upload_id, "columns": columns, "rows": len(rows)}


@app.post("/api/upload/analyze")
async def upload_analyze(body: _UploadAnalyzeRequest) -> dict:
    if body.upload_id not in _uploads:
        raise HTTPException(status_code=404, detail="Upload session not found — please re-upload the file")

    rows = _uploads[body.upload_id]["rows"]
    x_values, y_values = [], []
    for row in rows:
        try:
            x_values.append(float(str(row[body.x_col]).replace(",", ".")))
            y_values.append(float(str(row[body.y_col]).replace(",", ".")))
        except (KeyError, ValueError):
            continue

    if not x_values:
        raise HTTPException(status_code=400, detail=f"No numeric data found in columns '{body.x_col}' / '{body.y_col}'")

    x_arr = np.asarray(x_values, dtype=np.float64)
    y_arr = np.asarray(y_values, dtype=np.float64)

    try:
        analysis = await asyncio.to_thread(
            _pipeline.analyze_uploaded_signal,
            x_arr, y_arr, body.x_col, body.y_col, body.label, body.sampling_rate_hz,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    return _serialize(analysis)


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------

def _serialize(analysis: DashboardAnalysis) -> dict:
    assessment: HealthAssessment = analysis.health_assessment
    timeline = assessment.timeline
    last_feature = analysis.feature_sets[-1]

    n = min(3000, len(analysis.raw_signal_x))
    indices = np.linspace(0, len(analysis.raw_signal_x) - 1, n, dtype=int)

    profile = assessment.historical_threshold_profile
    warning_score = (
        profile.warning_score if profile else assessment.thresholds.normal_max_score
    )
    critical_score = (
        profile.critical_score if profile else assessment.thresholds.warning_max_score
    )

    dominant_freq = (
        last_feature.dominant_frequency_x_hz + last_feature.dominant_frequency_y_hz
    ) / 2.0

    moving_avg_rms = (assessment.moving_average * assessment.baseline_rms).tolist()

    return {
        "signal": {
            "x": analysis.raw_signal_x[indices].tolist(),
            "y": analysis.raw_signal_y[indices].tolist(),
        },
        "trend": {
            "windows": list(range(len(timeline))),
            "rms": [float(p.rms_vector) for p in timeline],
            "moving_avg_rms": moving_avg_rms,
            "jhi": [float(p.health_index) for p in timeline],
            "warning_score": float(warning_score),
            "critical_score": float(critical_score),
        },
        "status": {
            "state": assessment.current_state,
            "jhi": round(assessment.current_health_index, 1),
            "recommendation": assessment.recommendation,
            "hours_to_critical": (
                round(assessment.estimated_hours_to_critical, 1)
                if assessment.estimated_hours_to_critical is not None
                else None
            ),
            "hours_to_critical_lower_90": (
                round(assessment.hours_to_critical_lower_90, 1)
                if assessment.hours_to_critical_lower_90 is not None
                else None
            ),
            "hours_to_critical_upper_90": (
                round(assessment.hours_to_critical_upper_90, 1)
                if assessment.hours_to_critical_upper_90 is not None
                else None
            ),
            "op_hours_to_critical": (
                round(assessment.estimated_operating_hours_to_critical, 1)
                if assessment.estimated_operating_hours_to_critical is not None
                else None
            ),
            "current_rms": round(float(timeline[-1].rms_vector), 3) if timeline else 0.0,
            "dominant_freq_hz": round(dominant_freq, 1),
        },
        "meta": {
            "joint": analysis.measurement.joint,
            "date": analysis.measurement.measurement_date,
            "station": analysis.measurement.station,
            "trajectory": analysis.measurement.trajectory,
            "sampling_rate_hz": round(analysis.sampling_rate_hz, 1),
            "x_column": analysis.selected_x_column,
            "y_column": analysis.selected_y_column,
            "warning_hours": (
                round(profile.warning_hours, 1)
                if profile and profile.warning_hours is not None
                else None
            ),
            "critical_hours": (
                round(profile.critical_hours, 1)
                if profile and profile.critical_hours is not None
                else None
            ),
        },
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    dev = os.environ.get("ENV", "development") == "development"
    uvicorn.run("App.web.server:app", host="0.0.0.0", port=port, reload=dev)
