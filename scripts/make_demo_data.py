"""
Generate data_demo/ — truncated CSV files for cloud deployment (Render.com).

Copies only Trajectory 1 for each joint and measurement date, keeping
the first ROWS_TO_KEEP rows. The result is ~43 MB vs 1.7 GB for full data.

Usage (from project root):
    python scripts/make_demo_data.py
"""

from __future__ import annotations

import shutil
from pathlib import Path

ROWS_TO_KEEP = 5_000
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_SRC = PROJECT_ROOT / "data"
DATA_DST = PROJECT_ROOT / "data_demo"


def truncate_csv(src: Path, dst: Path, rows: int) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with src.open("r", encoding="utf-8", errors="replace") as fin, \
         dst.open("w", encoding="utf-8", newline="") as fout:
        for i, line in enumerate(fin):
            if i >= rows + 1:  # +1 for header row
                break
            fout.write(line)


def main() -> None:
    if DATA_DST.exists():
        shutil.rmtree(DATA_DST)
    DATA_DST.mkdir()

    copied = 0
    for joint_dir in sorted(DATA_SRC.iterdir()):
        if not joint_dir.is_dir() or not joint_dir.name.lower().startswith("joint"):
            continue
        for date_dir in sorted(joint_dir.iterdir()):
            if not date_dir.is_dir():
                continue
            for station_dir in sorted(date_dir.iterdir()):
                if not station_dir.is_dir():
                    continue
                # Only Trajectory 1
                traj_dir = station_dir / "Trajectory 1"
                if not traj_dir.exists():
                    continue
                for csv_file in sorted(traj_dir.glob("*.csv")):
                    rel = csv_file.relative_to(DATA_SRC)
                    dst_file = DATA_DST / rel
                    truncate_csv(csv_file, dst_file, ROWS_TO_KEEP)
                    copied += 1
                    print(f"  {rel}")

    print(f"\nDone — {copied} files written to data_demo/")


if __name__ == "__main__":
    main()
