#!/usr/bin/env python3
"""NX campaign: boundary-path optimization and moment-preserving positivity projection (docs/POSITIVITY_PROJECTION.md).

  prepare --campaign DIR --reference-bin REF --bin NEW --bench BENCH   freeze binaries, write manifest and pack files
  submit  --campaign DIR [--execute] [--max-concurrent 2]             sbatch the packs (dry run without --execute)
  collect --campaign DIR                                              results.jsonl (+ bench.jsonl), missing runs listed

Packs (nx/batches/element_weno_pack.sbatch + scripts/element_weno/pack_node.py, copied into the campaign):
  runtime_p1, runtime_p2  SLOT_CORES=32 (one slot per socket: independent memory bandwidth for timings)
      crossing-study 2D, T=0.8 (both pulses leave through the outflow boundary), E in 32..256, N in 3,7,15,
      variants ref (frozen bdd5d3f), c6 (optimized, same method), proj (rule 2N+1), projN1 (rule N+1)
  scaling                 SLOT_CORES=16
      N scaling: E=64, p=1, N in 3,5,7,11,15,21,31 (c6, proj); grid scaling: N=7, p=1, E in 512, 1024 (c6, proj);
      kernel benchmark (single thread, N up to 31)
Nothing is submitted without --execute; an existing submission record blocks resubmission.
"""
import argparse
import hashlib
import json
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
COMMON = ["--problem", "crossing-study", "--dimension", "2", "--time", ".8", "--cfl", ".9", "--sample-grid", "256"]
VARIANTS = {
    "ref": ("ref", []),
    "c6": ("new", ["--c6-ghost"]),  # the projection is the default since 2026-11-14
    "proj": ("new", ["--positivity-projection"]),
    "projN1": ("new", ["--positivity-projection", "--projection-rule", "N+1"]),
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


OUTFLOW = ["--problem", "smooth-outflow", "--dimension", "2", "--time", ".6", "--cfl", ".9", "--sample-grid", "256"]


def run_task(campaign, name, binary, N, p, E, extra, common=COMMON):
    extra = [str(N + 1) if x == "N+1" else x for x in extra]
    args = [str(campaign / "bin" / binary), *common, "--pn-order", str(N), "--degree", str(p), "--elements", str(E),
            *extra, "--output", str(campaign / "runs" / name)]
    return name, shlex.join(args)


def tasks(campaign, suite="crossing"):
    if suite == "outflow":
        # smooth-outflow: smooth angular data (PN error small), exact kinetic reference; accuracy campaign.
        items = []
        for p in (2, 1):
            for E in (256, 128, 64, 32):
                for N in (15, 7, 3):
                    for v in ("c6", "proj", "projN1"):
                        binary, extra = VARIANTS[v]
                        items.append(run_task(campaign, f"of_{v}_p{p}_E{E}_N{N}", binary, N, p, E, extra, OUTFLOW))
        return {"outflow": items}
    packs = {"runtime_p1": [], "runtime_p2": [], "scaling": []}
    for p in (1, 2):
        # Largest first: the long runs start immediately (better packing).
        for E in (256, 128, 64, 32):
            for N in (15, 7, 3):
                for v, (binary, extra) in VARIANTS.items():
                    packs[f"runtime_p{p}"].append(
                        run_task(campaign, f"rt_{v}_p{p}_E{E}_N{N}", binary, N, p, E, extra))
    for E in (1024, 512):
        for v in ("c6", "proj"):
            binary, extra = VARIANTS[v]
            packs["scaling"].append(run_task(campaign, f"grid_{v}_p1_E{E}_N7", binary, 7, 1, E, extra))
    for N in (31, 21, 15, 11, 7, 5, 3):
        for v in ("c6", "proj"):
            binary, extra = VARIANTS[v]
            packs["scaling"].append(run_task(campaign, f"order_{v}_p1_E64_N{N}", binary, N, 1, 64, extra))
    bench = (f"OMP_NUM_THREADS=1 {shlex.quote(str(campaign / 'bin' / 'bench'))} --orders 3,5,7,11,15,21,31 "
             f"--states 2000 --repeat 3 > {shlex.quote(str(campaign / 'runs' / 'bench.jsonl'))}")
    packs["scaling"].insert(2, ("bench", bench))
    if suite == "runtime":
        return {k: v for k, v in packs.items() if k.startswith("runtime")}
    return packs


def prepare(ns):
    campaign = ns.campaign.resolve()
    if campaign.exists():
        sys.exit(f"{campaign} exists; choose a new campaign folder")
    (campaign / "bin").mkdir(parents=True)
    (campaign / "runs").mkdir()
    (campaign / "packs").mkdir()
    for src, dst in ((ns.reference_bin, "ref"), (ns.bin, "new"), (ns.bench, "bench")):
        shutil.copy2(src, campaign / "bin" / dst)
    for rel in ("nx/batches/element_weno_pack.sbatch", "scripts/element_weno/pack_node.py",
                "scripts/positivity/campaign.py", "scripts/positivity/compare_runs.py"):
        (campaign / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO / rel, campaign / rel)
    commit = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", str(REPO), "status", "--porcelain", "--untracked-files=no"],
                           capture_output=True, text=True).stdout.strip()
    manifest = {"created": time.strftime("%Y-%m-%dT%H:%M:%S"), "commit": commit, "dirty": bool(dirty),
                "reference_commit": "bdd5d3f", "binaries": {}, "packs": {}}
    for b in ("ref", "new", "bench"):
        manifest["binaries"][b] = {"sha256": sha256(campaign / "bin" / b)}
    slot = {"runtime_p1": 32, "runtime_p2": 32, "scaling": 16, "outflow": 16}
    manifest["suite"] = ns.suite
    for pack, items in tasks(campaign, ns.suite).items():
        path = campaign / "packs" / f"{pack}.tsv"
        path.write_text("".join(f"{name}\t{command}\n" for name, command in items))
        manifest["packs"][pack] = {"taskfile": str(path), "slot_cores": slot[pack], "tasks": len(items),
                                   "commands": {name: command for name, command in items}}
    (campaign / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(json.dumps({k: (v["tasks"], v["slot_cores"]) for k, v in manifest["packs"].items()}))


def submit(ns):
    campaign = ns.campaign.resolve()
    manifest = json.loads((campaign / "manifest.json").read_text())
    record = campaign / "submission.json"
    if record.exists():
        sys.exit(f"{record} exists: already submitted")
    for b, info in manifest["binaries"].items():
        if sha256(campaign / "bin" / b) != info["sha256"]:
            sys.exit(f"binary {b} changed since prepare")
    sbatch = shutil.which("sbatch") or "sbatch"
    jobs, submitted = [], {}
    for k, (pack, info) in enumerate(manifest["packs"].items()):
        deps = [f"--dependency=afterany:{jobs[k - ns.max_concurrent]}"] if k >= ns.max_concurrent else []
        cmd = [sbatch, "--parsable", *deps, f"--job-name=pp-{pack}", "--chdir=" + str(campaign / "packs"),
               f"--export=ALL,TASKFILE={info['taskfile']},REPO={campaign},SLOT_CORES={info['slot_cores']}",
               str(campaign / "nx/batches/element_weno_pack.sbatch")]
        print(" ".join(cmd))
        if ns.execute:
            out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.strip()
            jobs.append(out.split(";")[0])
            submitted[pack] = {"job": jobs[-1], "command": cmd}
        else:
            jobs.append(f"<{pack}>")
    if ns.execute:
        record.write_text(json.dumps({"submitted": time.strftime("%Y-%m-%dT%H:%M:%S"), "jobs": submitted}, indent=1))
        print(json.dumps({k: v["job"] for k, v in submitted.items()}))
    else:
        print("dry run: nothing submitted (use --execute)")


def collect(ns):
    campaign = ns.campaign.resolve()
    manifest = json.loads((campaign / "manifest.json").read_text())
    rows, missing = [], []
    for pack, info in manifest["packs"].items():
        for name in info["commands"]:
            if name == "bench":
                continue
            summary = campaign / "runs" / name / "summary.json"
            if not summary.exists():
                missing.append(name)
                continue
            d = json.loads(summary.read_text())
            d["name"], d["pack"] = name, pack
            d["variant"] = name.split("_")[1]
            rows.append(d)
    (campaign / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(f"{len(rows)} runs collected, {len(missing)} missing" + (": " + ", ".join(missing) if missing else ""))
    return 0 if not missing else 1


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("prepare")
    a.add_argument("--campaign", type=Path, required=True)
    a.add_argument("--reference-bin", type=Path, required=True)
    a.add_argument("--bin", type=Path, required=True)
    a.add_argument("--bench", type=Path, required=True)
    a.add_argument("--suite", choices=("crossing", "outflow", "runtime"), default="crossing")
    a = sub.add_parser("submit")
    a.add_argument("--campaign", type=Path, required=True)
    a.add_argument("--execute", action="store_true")
    a.add_argument("--max-concurrent", type=int, default=2)
    a = sub.add_parser("collect")
    a.add_argument("--campaign", type=Path, required=True)
    ns = p.parse_args()
    return {"prepare": prepare, "submit": submit, "collect": collect}[ns.cmd](ns) or 0


if __name__ == "__main__":
    sys.exit(main())
