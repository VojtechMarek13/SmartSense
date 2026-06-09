from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.data_loading.loader import VibrationDataLoader
    from src.data_loading.models import TrajectoryMeasurement

try:
    from asyncua import Client as _AsyncuaClient
    _HAS_ASYNCUA = True
except ImportError:
    _AsyncuaClient = None  # type: ignore[assignment,misc]
    _HAS_ASYNCUA = False

_log = logging.getLogger(__name__)

# ── OPC UA topology ──────────────────────────────────────────────────────────

ENDPOINT = "opc.tcp://192.168.209.35:4840"
USERNAME = "Easycon"
PASSWORD = "Easycon!2026"

SENSOR_NODES: dict[str, dict[str, str]] = {
    "gCMCtrl_1": {
        "analog": "ns=6;s=::AsGlobalPV:gCMCtrl_1.Param.AnalogInput",
        "rms":    "ns=6;s=::AsGlobalPV:gCMCtrl_1.Diagnostic.RMSVelEnvelope",
    },
    "gCMCtrl_2": {
        "analog": "ns=6;s=::AsGlobalPV:gCMCtrl_2.Param.AnalogInput",
        "rms":    "ns=6;s=::AsGlobalPV:gCMCtrl_2.Diagnostic.RMSVelEnvelope",
    },
    "gCMCtrl_3": {
        "analog": "ns=6;s=::AsGlobalPV:gCMCtrl_3.Param.AnalogInput",
        "rms":    "ns=6;s=::AsGlobalPV:gCMCtrl_3.Diagnostic.RMSVelEnvelope",
    },
}

# (sensor, x_index, y_index) into REAL[0..3]
# gCMCtrl_N carries two joints: indices 0/1 = first joint X/Y, 2/3 = second joint X/Y
JOINT_MAP: dict[str, tuple[str, int, int]] = {
    "Joint 1": ("gCMCtrl_1", 0, 1),
    "Joint 2": ("gCMCtrl_1", 2, 3),
    "Joint 3": ("gCMCtrl_2", 0, 1),
    "Joint 4": ("gCMCtrl_2", 2, 3),
    "Joint 5": ("gCMCtrl_3", 0, 1),
    "Cobot":   ("gCMCtrl_3", 2, 3),
}


@dataclass(frozen=True)
class LiveDataPoint:
    t: int
    x: float
    y: float


# ── Simulator ────────────────────────────────────────────────────────────────

class OpcUaSimulator:
    """
    Replays CSV vibration data as a live OPC-UA stream at a fixed rate.
    Used in demo/cloud deployments where the PLC is not reachable.
    """

    EMIT_HZ: int = 20

    def __init__(self, loader: VibrationDataLoader) -> None:
        self._loader = loader
        self._queues: set[asyncio.Queue[dict]] = set()
        self._joint: str = "Joint 1"

    def set_joint(self, joint: str) -> None:
        self._joint = joint

    def subscribe(self) -> asyncio.Queue[dict]:
        q: asyncio.Queue[dict] = asyncio.Queue(maxsize=300)
        self._queues.add(q)
        return q

    def unsubscribe(self, q: asyncio.Queue[dict]) -> None:
        self._queues.discard(q)

    async def run(self, measurements: list[TrajectoryMeasurement]) -> None:
        """Load CSV data once and stream samples indefinitely at EMIT_HZ."""
        interval = 1.0 / self.EMIT_HZ

        while True:
            joint_measurements = [m for m in measurements if m.joint == self._joint]
            if not joint_measurements:
                await asyncio.sleep(1.0)
                continue

            measurement = joint_measurements[-1]
            rows, _, _ = await asyncio.to_thread(
                self._loader.load_selected_signals, measurement
            )

            x_values, y_values = self._parse_rows(rows)
            if not x_values:
                await asyncio.sleep(1.0)
                continue

            sample_index = 0
            for idx in range(len(x_values)):
                point = {"t": sample_index, "x": x_values[idx], "y": y_values[idx]}
                self._broadcast(point)
                sample_index += 1
                await asyncio.sleep(interval)

    def _parse_rows(self, rows: list[dict]) -> tuple[list[float], list[float]]:
        x_values: list[float] = []
        y_values: list[float] = []
        for row in rows:
            try:
                x = float(str(row["analog_raw_input_x"]).replace(",", "."))
                y = float(str(row["analog_raw_input_y"]).replace(",", "."))
                x_values.append(x)
                y_values.append(y)
            except (KeyError, ValueError):
                continue
        return x_values, y_values

    def _broadcast(self, point: dict) -> None:
        for q in list(self._queues):
            try:
                q.put_nowait(point)
            except asyncio.QueueFull:
                try:
                    q.get_nowait()
                except asyncio.QueueEmpty:
                    pass
                try:
                    q.put_nowait(point)
                except asyncio.QueueFull:
                    pass


# ── Real OPC UA client ───────────────────────────────────────────────────────

class _SubscriptionHandler:
    """asyncua DataChange handler — updates sensor cache in place."""

    def __init__(
        self,
        node_to_key: dict[str, tuple[str, str]],
        data: dict[str, dict[str, list[float]]],
    ) -> None:
        self._node_to_key = node_to_key
        self._data = data

    def datachange_notification(self, node, val, _data) -> None:  # noqa: ANN001
        key = self._node_to_key.get(node.nodeid.to_string())
        if key is None:
            return
        sensor, kind = key
        try:
            self._data[sensor][kind] = [float(v) for v in val]
        except (TypeError, ValueError):
            pass


class OpcUaClient:
    """
    Live OPC UA client for B&R APC PLC (opc.tcp://192.168.209.35:4840).

    Subscribes to AnalogInput and RMSVelEnvelope on all 3 sensor modules,
    then emits the selected joint's X/Y channels at EMIT_HZ over the shared
    queue interface (same API as OpcUaSimulator).

    Set OPCUA_MODE=opcua env var on the backend to activate.
    Automatically retries on connection loss.
    """

    EMIT_HZ: int = 20

    def __init__(self) -> None:
        self._data: dict[str, dict[str, list[float]]] = {
            s: {"analog": [0.0] * 4, "rms": [0.0] * 4}
            for s in SENSOR_NODES
        }
        self._node_to_key: dict[str, tuple[str, str]] = {}
        self._queues: set[asyncio.Queue[dict]] = set()
        self._joint: str = "Joint 1"
        self._t: int = 0

    def set_joint(self, joint: str) -> None:
        if joint in JOINT_MAP:
            self._joint = joint

    def subscribe(self) -> asyncio.Queue[dict]:
        q: asyncio.Queue[dict] = asyncio.Queue(maxsize=300)
        self._queues.add(q)
        return q

    def unsubscribe(self, q: asyncio.Queue[dict]) -> None:
        self._queues.discard(q)

    def _broadcast(self, point: dict) -> None:
        for q in list(self._queues):
            try:
                q.put_nowait(point)
            except asyncio.QueueFull:
                try:
                    q.get_nowait()
                except asyncio.QueueEmpty:
                    pass
                try:
                    q.put_nowait(point)
                except asyncio.QueueFull:
                    pass

    async def run(self) -> None:
        if not _HAS_ASYNCUA:
            raise RuntimeError(
                "asyncua is not installed — run: pip install asyncua>=1.0.0"
            )
        while True:
            try:
                await self._connect_and_stream()
            except Exception as exc:
                _log.warning("OPC UA error: %s — retrying in 5 s", exc)
                await asyncio.sleep(5)

    async def _connect_and_stream(self) -> None:
        client = _AsyncuaClient(url=ENDPOINT)
        client.set_user(USERNAME)
        client.set_password(PASSWORD)

        async with client:
            self._node_to_key.clear()
            nodes = []
            for sensor, node_ids in SENSOR_NODES.items():
                for kind, node_id_str in node_ids.items():
                    node = client.get_node(node_id_str)
                    nodes.append(node)
                    self._node_to_key[node.nodeid.to_string()] = (sensor, kind)

            handler = _SubscriptionHandler(self._node_to_key, self._data)
            sub = await client.create_subscription(1000 // self.EMIT_HZ, handler)
            await sub.subscribe_data_change(nodes)
            _log.info(
                "OPC UA connected — %d nodes subscribed @ %s", len(nodes), ENDPOINT
            )

            interval = 1.0 / self.EMIT_HZ
            while True:
                sensor, xi, yi = JOINT_MAP.get(self._joint, JOINT_MAP["Joint 1"])
                point = {
                    "t": self._t,
                    "x": self._data[sensor]["analog"][xi],
                    "y": self._data[sensor]["analog"][yi],
                }
                self._t += 1
                self._broadcast(point)
                await asyncio.sleep(interval)
