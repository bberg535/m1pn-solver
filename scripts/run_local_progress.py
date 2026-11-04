#!/usr/bin/env python3
"""Run a local standard-B Crossing test with live terminal progress."""
from __future__ import annotations

import argparse
import datetime
import json
import math
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import time
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]


def duration(seconds: float) -> str:
    minutes, seconds = divmod(max(0, round(seconds)), 60)
    return f"{minutes:d}:{seconds:02d}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--elements", type=int, default=16)
    parser.add_argument("--pn-order", type=int, default=3)
    parser.add_argument("--time", type=float, default=.32)
    parser.add_argument("--dt", type=float, default=.0001)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--interval", type=float, default=1, help="Progress interval in seconds.")
    parser.add_argument("--exe", type=Path, default=ROOT / "nx/tmp/build_local/m1pn_mc_study")
    parser.add_argument("--output", type=Path, help="New result directory; default: tmp/local_B_<timestamp>.")
    args = parser.parse_args()
    if min(args.elements, args.pn_order, args.threads) < 1:
        parser.error("elements, pn-order and threads must be positive")
    if not all(math.isfinite(v) and v > 0 for v in (args.time, args.dt, args.interval)):
        parser.error("time, dt and interval must be finite and positive")
    if args.interval > 30:
        parser.error("interval must not exceed 30 seconds")
    exe = args.exe.resolve()
    if not exe.is_file() or not os.access(exe, os.X_OK):
        parser.error(f"Study executable missing: {exe}. Build the study first or select --exe.")
    stamp = datetime.datetime.now(ZoneInfo("Europe/Berlin")).strftime("%Y%m%d_%H%M%S_%f")
    output = (args.output or ROOT / "tmp" / ("local_B_" + stamp)).resolve()
    log_path = output.with_name(output.name + ".log")
    if output.exists() or log_path.exists():
        parser.error(f"Choose a new output directory: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [str(exe), "--case", "crossing-study", "--inflow", "crossing-study-transport",
               "--dimension", "2", "--mesh-type", "hex", "--elements", str(args.elements),
               "--pn-order", str(args.pn_order), "--time", str(args.time), "--dt", str(args.dt),
               "--cfl", "1", "--model", "r2", "--mass-mode", "lumped",
               "--stabilization", "m1pn_mass_correction", "--pn-viscosity-scale", "1",
               "--biharmonic-gamma", "0", "--kv-gamma", "0", "--time-integrator", "ssprk3",
               "--pulse-width", ".07", "--background", ".001", "--threads", str(args.threads),
               "--max-steps", "1000000", "--memory-mib", "4096", "--output", str(output)]
    env = dict(os.environ, OMP_NUM_THREADS=str(args.threads), OPENBLAS_NUM_THREADS="1",
               MKL_NUM_THREADS="1", OMP_DYNAMIC="FALSE")
    print(f"Standard B · Crossing {args.elements}×{args.elements} · P{args.pn_order} · {args.threads} Thread(s)", flush=True)
    print(f"Ergebnisse: {output}\nSolver-Log: {log_path}", flush=True)
    start = time.monotonic()
    terminal = sys.stdout.isatty()

    def show() -> None:
        elapsed = time.monotonic() - start
        try:
            progress = json.loads((output / "progress.json").read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            line = f"Initialisierung … Laufzeit {duration(elapsed)}"
        else:
            t = float(progress["time"])
            fraction = min(1., max(0., t / args.time))
            remaining = elapsed * (1. - fraction) / fraction if fraction > 0 else None
            eta = duration(remaining) if remaining is not None else "--:--"
            line = (f"{fraction:6.1%} | Schritt {progress['steps']:5d} | t={t:.5f}/{args.time:g}"
                    f" | Laufzeit {duration(elapsed)} | Rest ≈ {eta}")
        print(("\r" + line.ljust(105)) if terminal else line, end="" if terminal else "\n", flush=True)

    with log_path.open("x") as log:
        log.write(shlex.join(command) + "\n")
        log.flush()
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                   env=env, start_new_session=True)
        try:
            show()
            while process.poll() is None:
                try:
                    process.wait(timeout=args.interval)
                except subprocess.TimeoutExpired:
                    pass
                show()
        except KeyboardInterrupt:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGINT)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
            print("\nAbgebrochen. Bisherige Daten und Log bleiben erhalten.", flush=True)
            return 130
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
    if terminal:
        print()
    if process.returncode:
        print(f"Solver beendet mit Code {process.returncode}. Letzte Logzeilen:", file=sys.stderr)
        print("\n".join(log_path.read_text().splitlines()[-15:]), file=sys.stderr)
        return process.returncode if process.returncode > 0 else 1
    print("Prüfe gespeicherte Felder, M1-Kegel und globale Bilanz …", flush=True)
    verified = subprocess.run([sys.executable, str(ROOT / "scripts/verify_run.py"), str(output)],
                              capture_output=True, text=True, env=env)
    if verified.returncode:
        print(verified.stdout + verified.stderr, file=sys.stderr)
        return 1
    result = json.loads(verified.stdout)
    print(f"Bestanden · {result['nodes']} Knoten · globale Bilanz {result['maximum_global_Q_balance']:.3e}")
    print(f"Endfeld: {output / 'final.csv'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
