from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.data_loading.loader import VibrationDataLoader
    from src.data_loading.models import TrajectoryMeasurement


@dataclass(frozen=True)
class LiveDataPoint:
    t: int
    x: float
    y: float


class OpcUaSimulator:
    """
    Replays CSV vibration data as a live OPC-UA stream at a fixed rate.

    Swap `run()` for a real asyncua.Client subscription when connecting to PLC.
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

            # Use the latest (highest operating-hour) measurement for simulation
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

            # Loop back to start of file continuously

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
