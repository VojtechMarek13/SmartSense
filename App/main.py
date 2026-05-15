from __future__ import annotations

import sys
from pathlib import Path

if __package__ in {None, ""}:
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from App.dashboard import run
else:
    from App.dashboard import run


if __name__ == "__main__":
    raise SystemExit(run())
