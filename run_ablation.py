"""Watchdog: wait for the main 5-pair run to finish, then launch the
HRRL_NO_Q ablation on seed 7, and afterwards the LANGGRAPH_INFORMED fair
baseline on seeds 7 and 17. Each phase runs in its own output directory so
checkpoints cannot collide.

Why isolated
------------
- Main watchdog (final_runner.py) writes to ``benchmark_results_v7/``.
- Ablation writes to ``benchmark_results_v7_ablation/``.
- Fair baseline writes to ``benchmark_results_v7_fairbaseline/``.

Phase chain
-----------
1. Wait for both ``logs_hrrl_seed256.json`` and
   ``logs_langgraph_seed256.json`` to be present (main 5-pair run done).
2. Run HRRL_NO_Q ablation (seed 7).
3. Run LANGGRAPH_INFORMED fair baseline (seeds 7, 17). Two seeds chosen to
   provide a contrast preliminar without exhausting the daily API quota.

Usage
-----
  python run_ablation.py --launch    # detached background process
  python run_ablation.py --run       # foreground (debug)
  python run_ablation.py --status    # print state
"""
from __future__ import annotations

import ctypes
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MAIN_OUTPUT_DIR = ROOT / "benchmark_results_v7"
OUTPUT_DIR = ROOT / "benchmark_results_v7_ablation"
FAIRBASE_OUTPUT_DIR = ROOT / "benchmark_results_v7_fairbaseline"
LOG_PATH = ROOT / "ablation_runner.log"
STATE_PATH = ROOT / "ablation_runner_state.json"
PID_FILE = ROOT / "ablation_runner_pid.txt"

ABLATION_SEED = 7
FAIRBASE_SEEDS = [7, 17]
ITERATIONS = 80

WAIT_FILES = [
    MAIN_OUTPUT_DIR / "logs_hrrl_seed256.json",
    MAIN_OUTPUT_DIR / "logs_langgraph_seed256.json",
]

POLL_INTERVAL_S = 60
MAX_WAIT_HOURS = 12


def _bench_cmd(modes: list[str], seeds: list[int], output_dir: Path) -> list[str]:
    return [
        sys.executable,
        str(ROOT / "benchmark.py"),
        "--iterations", str(ITERATIONS),
        "--seeds", *[str(s) for s in seeds],
        "--modes", *modes,
        "--config", str(ROOT / "calibration_results.json"),
        "--output-dir", str(output_dir),
        "--api-hard-limit", "500",
        "--log-level", "WARNING",
        "--no-halt-on-red-flags",
    ]


ABLATION_CMD = _bench_cmd(["hrrl_no_q"], [ABLATION_SEED], OUTPUT_DIR)
FAIRBASE_CMD = _bench_cmd(["langgraph_informed"], FAIRBASE_SEEDS, FAIRBASE_OUTPUT_DIR)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _log(msg: str) -> None:
    line = f"[{_now_iso()}] {msg}\n"
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line)
    try:
        sys.stdout.write(line)
        sys.stdout.flush()
    except Exception:
        pass


def _save_state(state: dict) -> None:
    state["updated_at"] = _now_iso()
    with open(STATE_PATH, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)


def _wait_for_main_run() -> bool:
    """Block until both seed-256 log files exist or hard timeout fires."""
    deadline = time.time() + MAX_WAIT_HOURS * 3600
    while time.time() < deadline:
        present = [p.exists() for p in WAIT_FILES]
        if all(present):
            _log(f"Wait sentinels satisfied: {[p.name for p in WAIT_FILES]}")
            return True
        missing = [p.name for p, ok in zip(WAIT_FILES, present) if not ok]
        _log(f"Waiting for seed-256 pair to finish; still missing: {missing}")
        time.sleep(POLL_INTERVAL_S)
    _log("Timed out waiting for main run to finish; aborting ablation launch.")
    return False


def _run_phase(label: str, cmd: list[str], output_dir: Path, log_name: str) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    _log(f"[{label}] Launching: {' '.join(cmd)}")
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    bench_log = ROOT / log_name
    with open(bench_log, "a", encoding="utf-8") as lf:
        lf.write(f"\n\n===== [{label}] {_now_iso()} =====\n")
        lf.flush()
        proc = subprocess.run(
            cmd, cwd=ROOT, env=env, stdout=lf, stderr=lf, text=True, check=False,
        )
    _log(f"[{label}] benchmark.py exited with code {proc.returncode}")
    return proc.returncode


def _run_benchmark() -> int:
    return _run_phase("ablation", ABLATION_CMD, OUTPUT_DIR, "ablation_runner_bench.log")


def _run_fair_baseline() -> int:
    return _run_phase(
        "fair-baseline", FAIRBASE_CMD, FAIRBASE_OUTPUT_DIR, "fairbaseline_runner_bench.log"
    )


def _run_loop() -> int:
    PID_FILE.write_text(str(os.getpid()), encoding="utf-8")
    state = {
        "phase": "waiting_for_main",
        "pid": os.getpid(),
        "started_at": _now_iso(),
        "wait_files": [p.name for p in WAIT_FILES],
    }
    _save_state(state)
    _log(f"ablation_runner started, PID={os.getpid()}")

    if not _wait_for_main_run():
        state["phase"] = "aborted_timeout"
        _save_state(state)
        PID_FILE.unlink(missing_ok=True)
        return 2

    state["phase"] = "running_ablation"
    _save_state(state)

    rc_ablation = _run_benchmark()
    state["ablation_return_code"] = rc_ablation
    _save_state(state)

    state["phase"] = "running_fair_baseline"
    _save_state(state)
    rc_fair = _run_fair_baseline()
    state["fair_baseline_return_code"] = rc_fair

    overall_rc = rc_ablation or rc_fair
    state["phase"] = "done" if overall_rc == 0 else "failed"
    state["finished_at"] = _now_iso()
    state["return_code"] = overall_rc
    _save_state(state)
    PID_FILE.unlink(missing_ok=True)
    return overall_rc


def _print_status() -> int:
    if STATE_PATH.exists():
        print(STATE_PATH.read_text(encoding="utf-8"))
    else:
        print("No ablation runner state.")
    if PID_FILE.exists():
        pid_text = PID_FILE.read_text(encoding="utf-8").strip()
        print(f"PID file: {pid_text}")
    return 0


def _launch_detached() -> int:
    """Spawn ourselves with --run as a fully detached process."""
    if PID_FILE.exists():
        pid_text = PID_FILE.read_text(encoding="utf-8").strip()
        _log(f"PID file already present ({pid_text}); not launching another instance.")
        return 1
    cmd = [sys.executable, str(Path(__file__).resolve()), "--run"]
    DETACHED_PROCESS = 0x00000008
    CREATE_NEW_PROCESS_GROUP = 0x00000200
    creation = DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP
    proc = subprocess.Popen(
        cmd,
        cwd=ROOT,
        creationflags=creation,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        stdin=subprocess.DEVNULL,
        close_fds=True,
    )
    _log(f"launched detached ablation_runner, PID={proc.pid}")
    return 0


def main() -> int:
    if "--launch" in sys.argv:
        return _launch_detached()
    if "--status" in sys.argv:
        return _print_status()
    if "--run" in sys.argv:
        return _run_loop()
    print(__doc__)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
