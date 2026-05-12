"""
run_doe.py — Batch DoE runner with 2-GPU parallel dispatch.

Reads a DoE JSON config, builds a run list, and dispatches each run
to the waLBerla app on one of the available GPUs.  Supports --dry-run
for validation without launching any processes.

Usage:
    uv run python scripts/run_doe.py --config config/doe_tier1.json
    uv run python scripts/run_doe.py --config config/doe_tier1.json --dry-run
    uv run python scripts/run_doe.py --config config/doe_tier1.json --gpus 0,1
    uv run python scripts/run_doe.py --config config/doe_tier1.json --run-ids T1_01,T1_07
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# ── Defaults ──────────────────────────────────────────────────────────────────

BUILD_DIR    = Path("build/apps")
RESULTS_DIR  = Path("results")
DEFAULT_APP  = "droplet_generator"


# ── Config handling ───────────────────────────────────────────────────────────

def load_config(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def expand_runs(config: dict, run_ids: list[str] | None = None) -> list[dict]:
    """
    Merge fixed_params into each run dict.
    Optionally filter by run_ids.
    """
    fixed = config.get("fixed_params", {})
    runs = []
    for run in config.get("runs", []):
        merged = {**fixed, **run}
        merged.setdefault("run_id", run.get("id", "unknown"))
        merged.setdefault("hdf5_out_dir", str(RESULTS_DIR / merged["run_id"]))
        runs.append(merged)

    if run_ids:
        runs = [r for r in runs if r.get("id") in run_ids]
        if not runs:
            print(f"ERROR: No runs matched filter {run_ids}")
            sys.exit(1)

    return runs


# ── Run dispatch ──────────────────────────────────────────────────────────────

def run_single(
    run_params: dict,
    app: Path,
    gpu_id: int,
    dry_run: bool,
) -> tuple[str, bool, float]:
    """
    Launch one simulation run.
    Returns (run_id, success, elapsed_seconds).
    """
    run_id = run_params.get("id", "unknown")
    out_dir = Path(run_params.get("hdf5_out_dir", str(RESULTS_DIR / run_id)))

    if not dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)

    # Write per-run JSON config to temp file
    run_config_path = out_dir / "run_config.json" if not dry_run else Path(
        tempfile.mktemp(suffix=".json")
    )

    if not dry_run:
        with open(run_config_path, "w") as f:
            json.dump(run_params, f, indent=2)

    cmd = [str(app), str(run_config_path)]
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu_id)}

    print(f"  [GPU {gpu_id}] {run_id}: {' '.join(cmd)}")
    if dry_run:
        print(f"    DRY-RUN — params: Q_oil={run_params.get('Q_oil_uL_min','?')} "
              f"Q_aq={run_params.get('Q_aq_uL_min','?')} "
              f"sigma={run_params.get('sigma_mN_m','?')} mN/m "
              f"θ={run_params.get('contact_angle_deg','?')}°")
        return (run_id, True, 0.0)

    t0 = time.perf_counter()
    log_path = out_dir / "sim.log"
    try:
        with open(log_path, "w") as log:
            proc = subprocess.run(
                cmd, env=env, stdout=log, stderr=subprocess.STDOUT,
                timeout=7200,   # 2 hr hard limit per run
            )
        elapsed = time.perf_counter() - t0
        success = proc.returncode == 0
        status = "OK" if success else f"FAILED (rc={proc.returncode})"
        print(f"  [GPU {gpu_id}] {run_id}: {status}  ({elapsed/60:.1f} min)")
        return (run_id, success, elapsed)
    except subprocess.TimeoutExpired:
        elapsed = time.perf_counter() - t0
        print(f"  [GPU {gpu_id}] {run_id}: TIMEOUT after {elapsed/60:.0f} min")
        return (run_id, False, elapsed)
    except FileNotFoundError:
        print(f"  ERROR: app not found: {app}")
        print("  Did you run `make build`?")
        return (run_id, False, 0.0)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Batch DoE runner")
    parser.add_argument("--config", required=True, help="Path to DoE JSON config")
    parser.add_argument("--dry-run", action="store_true", help="Print run plan, no execution")
    parser.add_argument("--gpus", default="0,1", help="Comma-separated GPU IDs (default: 0,1)")
    parser.add_argument("--app", default=DEFAULT_APP, help="App binary name")
    parser.add_argument("--run-ids", default="", help="Comma-separated run IDs to execute (all if empty)")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        sys.exit(f"ERROR: config not found: {config_path}")

    config = load_config(config_path)
    run_ids_filter = [r.strip() for r in args.run_ids.split(",") if r.strip()] or None
    runs = expand_runs(config, run_ids_filter)
    gpu_ids = [int(g.strip()) for g in args.gpus.split(",")]

    app_binary = BUILD_DIR / args.app
    app_name = config.get("app", args.app)
    if app_name != DEFAULT_APP:
        app_binary = BUILD_DIR / app_name

    print("=" * 60)
    print(f"DoE: {config.get('name', config_path.stem)}")
    print(f"  {config.get('description', '')}")
    print(f"  Runs: {len(runs)}   GPUs: {gpu_ids}   Dry-run: {args.dry_run}")
    print("=" * 60)

    # Assign GPUs round-robin
    gpu_assignments = [(run, gpu_ids[i % len(gpu_ids)]) for i, run in enumerate(runs)]

    results: list[tuple[str, bool, float]] = []
    with ThreadPoolExecutor(max_workers=len(gpu_ids)) as pool:
        futures = {
            pool.submit(run_single, run, app_binary, gpu, args.dry_run): run["id"]
            for run, gpu in gpu_assignments
        }
        for future in as_completed(futures):
            results.append(future.result())

    # Summary
    print("\n" + "=" * 60)
    print("Summary:")
    n_ok  = sum(1 for _, ok, _ in results if ok)
    n_fail = len(results) - n_ok
    total_h = sum(t for _, _, t in results) / 3600
    for run_id, ok, elapsed in sorted(results, key=lambda x: x[0]):
        status = "OK " if ok else "FAIL"
        print(f"  {status}  {run_id}  ({elapsed/60:.1f} min)")
    print(f"\nTotal: {n_ok}/{len(results)} succeeded, "
          f"{total_h:.1f} GPU-hours")

    if n_fail > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
