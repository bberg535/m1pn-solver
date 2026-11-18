from os import sched_getaffinity
from pathlib import Path

BASE = Path(__file__).resolve().parents[3] / "tmp" / "adaptive_pn_high_order"
CPUS = sorted(sched_getaffinity(0))
