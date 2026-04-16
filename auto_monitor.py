"""Autonomous monitor: watchdog + auto-fix of contaminated hrrl__seed7.

Designed to run as a fully detached process on Windows.
Survives terminal closure, IDE restart, everything.

Phases:
  1. WATCH  — poll benchmark_pid; if dead, check if worker is also dead and
              restart via run_full_experiment.py --detach
  2. CLEAN  — after main benchmark finishes, remove contaminated hrrl__seed7
              from checkpoint, backup the log, re-run only that combo
  3. DONE   — everything complete, exit

Usage:
  python auto_monitor.py --launch          # spawn detached instance
  python auto_monitor.py --run             # run in foreground (for debugging)
  python auto_monitor.py --status          # show current state
"""
from __future__ import annotations

import ctypes
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "benchmark_results_v7"
CHECKPOINT_PATH = OUTPUT_DIR / "benchmark_checkpoint.json"
STATUS_PATH = ROOT / "experiment_status.json"
EVENTS_PATH = ROOT / "experiment_events.jsonl"
LOG_PATH = ROOT / "auto_monitor.log"
STATE_PATH = ROOT / "auto_monitor_state.json"
PID_FILE = ROOT / "auto_monitor_pid.txt"

POLL_INTERVAL = 120  # seconds between checks
CONTAMINATED_KEY = "hrrl__seed7"

RERUN_CMD = [
    sys.executable,
    str(ROOT / "benchmark.py"),
    "--iterations", "80",
    "--seeds", "7",
    "--modes", "hrrl",
    "--config", str(ROOT / "calibration_results.json"),
    "--output-dir", str(OUTPUT_DIR),
    "--api-hard-limit", "500",
    "--log-level", "WARNING",
]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _log(msg: str) -> None:
    line = f"[{_now_iso()}] {msg}\n"
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line)


def _append_event(event: dict) -> None:
    event["timestamp"] = _now_iso()
    event["source"] = "auto_monitor"
    with open(EVENTS_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")


def _save_state(state: dict) -> None:
    with open(STATE_PATH, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)


def _read_json(path: Path) -> dict | list:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _pid_is_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
    PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
    STILL_ACTIVE = 259
    handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
    if not handle:
        return False
    try:
        exit_code = ctypes.c_ulong()
        if kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
            return exit_code.value == STILL_ACTIVE
        return False
    finally:
        kernel32.CloseHandle(handle)


def _find_benchmark_pid() -> int | None:
    """Find the benchmark.py subprocess via the worker's child processes."""
    status = _read_json(STATUS_PATH)
    worker_pid = status.get("worker_pid")
    if worker_pid and _pid_is_alive(worker_pid):
        return worker_pid
    return None


def _is_main_benchmark_done() -> bool:
    """Check if the main benchmark (all seeds except contaminated) is done.

    We detect this by checking experiment_status.json:
    - phase == 'completed' and success == True -> fully done
    - OR worker is dead and benchmark checkpoint has >= 9 entries
      (all combos except the contaminated hrrl__seed7 that's already there)
    """
    status = _read_json(STATUS_PATH)

    if status.get("phase") == "completed" and status.get("success"):
        return True

    worker_pid = status.get("worker_pid")
    if worker_pid and _pid_is_alive(worker_pid):
        return False

    checkpoint = _read_json(CHECKPOINT_PATH)
    if isinstance(checkpoint, list):
        keys = {e.get("_ck") for e in checkpoint if "_ck" in e}
    elif isinstance(checkpoint, dict):
        keys = set(checkpoint.keys())
    else:
        return False

    expected = set()
    for seed in [7, 17, 42, 123, 256]:
        for mode in ["hrrl", "langgraph"]:
            expected.add(f"{mode}__seed{seed}")

    missing = expected - keys
    remaining_non_contaminated = missing - {CONTAMINATED_KEY}
    return len(remaining_non_contaminated) == 0


def _is_seed7_contaminated() -> bool:
    """Check if hrrl__seed7 has exploded Q-weights (contamination signal)."""
    checkpoint = _read_json(CHECKPOINT_PATH)
    if not isinstance(checkpoint, list):
        return False
    for entry in checkpoint:
        if entry.get("_ck") != CONTAMINATED_KEY:
            continue
        weights = entry.get("final_q_weights", {})
        for agent_weights in weights.values():
            if isinstance(agent_weights, dict):
                for val in agent_weights.values():
                    if abs(val) > 100:
                        return True
        avg_reward = entry.get("avg_reward", 0)
        if avg_reward < -1.0:
            return True
        return False
    return False


def _clean_seed7() -> bool:
    """Remove contaminated hrrl__seed7 from checkpoint and backup log."""
    if not CHECKPOINT_PATH.exists():
        _log("ERROR: checkpoint not found, cannot clean seed7")
        return False

    with open(CHECKPOINT_PATH, "r", encoding="utf-8") as f:
        entries = json.load(f)

    before = len(entries)
    entries = [e for e in entries if e.get("_ck") != CONTAMINATED_KEY]
    after = len(entries)

    if before == after:
        _log("seed7 entry not found in checkpoint, nothing to clean")
        return True

    with open(CHECKPOINT_PATH, "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2, ensure_ascii=False)

    contaminated_log = OUTPUT_DIR / "logs_hrrl_seed7.json"
    backup_log = OUTPUT_DIR / "logs_hrrl_seed7_CONTAMINATED.json"
    if contaminated_log.exists():
        shutil.move(str(contaminated_log), str(backup_log))
        _log(f"Backed up contaminated log -> {backup_log.name}")

    run_ck = OUTPUT_DIR / "run_checkpoints" / "hrrl__seed7.json"
    if run_ck.exists():
        run_ck.unlink()
        _log("Deleted stale run checkpoint for hrrl__seed7")

    _log(f"Cleaned seed7 from checkpoint ({before} -> {after} entries)")
    _append_event({
        "event": "seed7_cleaned",
        "entries_before": before,
        "entries_after": after,
    })
    return True


def _rerun_seed7() -> int:
    """Run benchmark for hrrl seed=7 only. Returns exit code."""
    _log(f"Starting re-run of hrrl seed=7: {' '.join(RERUN_CMD)}")
    _append_event({"event": "seed7_rerun_started"})

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"

    rerun_log = ROOT / "rerun_seed7.log"
    with open(rerun_log, "w", encoding="utf-8") as lf:
        proc = subprocess.run(
            RERUN_CMD,
            cwd=ROOT,
            env=env,
            stdout=lf,
            stderr=lf,
            text=True,
            check=False,
        )

    _log(f"Re-run finished with code {proc.returncode}")
    _append_event({
        "event": "seed7_rerun_finished",
        "returncode": proc.returncode,
    })
    return proc.returncode


def _restart_experiment() -> None:
    """Restart the main experiment via run_full_experiment.py --detach."""
    _log("Worker dead, restarting experiment...")
    _append_event({"event": "monitor_restart_experiment"})

    cmd = [
        sys.executable,
        str(ROOT / "run_full_experiment.py"),
        "--detach",
        "--benchmark-output-dir", "benchmark_results_v7",
    ]
    subprocess.run(cmd, cwd=ROOT, check=False)
    _log("Experiment restarted via --detach")


def _run_monitor() -> int:
    """Main monitor loop."""
    PID_FILE.write_text(str(os.getpid()), encoding="utf-8")
    _log(f"Monitor started, PID={os.getpid()}")
    _append_event({"event": "monitor_started", "pid": os.getpid()})

    state = {"phase": "watching", "started_at": _now_iso()}
    _save_state(state)

    consecutive_dead = 0

    while True:
        try:
            if _is_main_benchmark_done():
                _log("Main benchmark finished. Checking seed7 contamination...")
                state["phase"] = "cleaning"
                _save_state(state)

                if _is_seed7_contaminated():
                    _log("seed7 IS contaminated. Cleaning and re-running...")
                    if not _clean_seed7():
                        _log("ERROR: cleanup failed, retrying in next cycle")
                        time.sleep(POLL_INTERVAL)
                        continue

                    rc = _rerun_seed7()
                    if rc == 0:
                        _log("seed7 re-run completed successfully!")
                        _append_event({"event": "monitor_completed"})
                        state["phase"] = "done"
                        state["finished_at"] = _now_iso()
                        _save_state(state)
                        PID_FILE.unlink(missing_ok=True)
                        return 0
                    elif rc == 75:
                        _log("seed7 re-run hit rate limit, will retry after cooldown")
                        time.sleep(300)
                        continue
                    else:
                        _log(f"seed7 re-run failed with code {rc}, retrying...")
                        time.sleep(POLL_INTERVAL)
                        continue
                else:
                    _log("seed7 is NOT contaminated (or already cleaned). Done.")
                    _append_event({"event": "monitor_completed_no_contamination"})
                    state["phase"] = "done"
                    state["finished_at"] = _now_iso()
                    _save_state(state)
                    PID_FILE.unlink(missing_ok=True)
                    return 0

            worker_pid = (_read_json(STATUS_PATH) or {}).get("worker_pid")
            if worker_pid and not _pid_is_alive(worker_pid):
                consecutive_dead += 1
                _log(f"Worker PID {worker_pid} dead (count={consecutive_dead})")
                if consecutive_dead >= 2:
                    status = _read_json(STATUS_PATH)
                    if not (status.get("finished") and status.get("success")):
                        _restart_experiment()
                        consecutive_dead = 0
                        time.sleep(60)
                        continue
            else:
                consecutive_dead = 0

        except Exception as exc:
            _log(f"ERROR in monitor loop: {exc}")

        time.sleep(POLL_INTERVAL)


def _launch_detached() -> int:
    """Spawn the monitor as a fully detached process."""
    child_args = [sys.executable, str(Path(__file__).resolve()), "--run"]

    creationflags = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP

    with open(LOG_PATH, "a", encoding="utf-8") as lf:
        proc = subprocess.Popen(
            child_args,
            cwd=ROOT,
            creationflags=creationflags,
            close_fds=True,
            stdout=lf,
            stderr=lf,
        )

    print(f"Auto-monitor launched as detached PID {proc.pid}")
    print(f"  Log: {LOG_PATH}")
    print(f"  State: {STATE_PATH}")
    print(f"  PID file: {PID_FILE}")
    return 0


def _show_status() -> int:
    """Print current monitor state."""
    if STATE_PATH.exists():
        print(json.dumps(_read_json(STATE_PATH), indent=2, ensure_ascii=False))
    else:
        print("No monitor state found.")

    if PID_FILE.exists():
        pid = int(PID_FILE.read_text().strip())
        alive = _pid_is_alive(pid)
        print(f"Monitor PID: {pid} ({'alive' if alive else 'DEAD'})")
    else:
        print("No PID file found.")
    return 0


if __name__ == "__main__":
    if "--launch" in sys.argv:
        raise SystemExit(_launch_detached())
    elif "--run" in sys.argv:
        raise SystemExit(_run_monitor())
    elif "--status" in sys.argv:
        raise SystemExit(_show_status())
    else:
        print("Usage: python auto_monitor.py [--launch | --run | --status]")
        raise SystemExit(1)
