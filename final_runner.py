"""Final watchdog: launch benchmark.py for all remaining seeds with auto-retry.

Replaces auto_monitor.py (whose seed7-only logic no longer applies after the
broadcast fix in _hrrl_single_tick).

Strategy
--------
- Single benchmark.py invocation for all 5 seeds x 2 modes (10 combos total).
- benchmark.py is idempotent w.r.t. its checkpoint: skips combos already in
  benchmark_checkpoint.json, resumes mid-run combos via run_checkpoints/.
- If benchmark.py exits non-zero we retry with backoff:
    * rc == 75  (rate-limit): sleep 600s
    * rc != 0:               sleep 120s, max 5 consecutive failures
- Exit cleanly when checkpoint contains all 10 combos.

Usage
-----
  python final_runner.py --launch    # spawn detached
  python final_runner.py --run       # foreground (debug)
  python final_runner.py --status    # print state
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
OUTPUT_DIR = ROOT / "benchmark_results_v7"
CHECKPOINT_PATH = OUTPUT_DIR / "benchmark_checkpoint.json"
LOG_PATH = ROOT / "final_runner.log"
STATE_PATH = ROOT / "final_runner_state.json"
PID_FILE = ROOT / "final_runner_pid.txt"

SEEDS = [7, 17, 42, 123, 256]
MODES = ["hrrl", "langgraph"]
ITERATIONS = 80

POLL_RETRY = 120
RATE_LIMIT_BACKOFF = 600
MAX_CONSECUTIVE_FAILURES = 5

BENCH_CMD = [
    sys.executable,
    str(ROOT / "benchmark.py"),
    "--iterations", str(ITERATIONS),
    "--seeds", *(str(s) for s in SEEDS),
    "--modes", *MODES,
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
    try:
        sys.stdout.write(line)
        sys.stdout.flush()
    except Exception:
        pass


def _save_state(state: dict) -> None:
    state["updated_at"] = _now_iso()
    with open(STATE_PATH, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)


def _read_checkpoint_keys() -> set[str]:
    if not CHECKPOINT_PATH.exists():
        return set()
    try:
        with open(CHECKPOINT_PATH, "r", encoding="utf-8") as f:
            entries = json.load(f)
    except Exception:
        return set()
    keys = set()
    for e in entries if isinstance(entries, list) else []:
        if "_ck" in e:
            keys.add(e["_ck"])
    return keys


def _expected_keys() -> set[str]:
    return {f"{m}__seed{s}" for s in SEEDS for m in MODES}


def _missing_keys() -> set[str]:
    return _expected_keys() - _read_checkpoint_keys()


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


def _run_benchmark_once() -> int:
    _log(f"Launching benchmark: {' '.join(BENCH_CMD)}")
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"

    bench_log = ROOT / "final_runner_bench.log"
    with open(bench_log, "a", encoding="utf-8") as lf:
        lf.write(f"\n\n===== {_now_iso()} =====\n")
        lf.flush()
        proc = subprocess.run(
            BENCH_CMD,
            cwd=ROOT,
            env=env,
            stdout=lf,
            stderr=lf,
            text=True,
            check=False,
        )
    _log(f"benchmark.py exited with code {proc.returncode}")
    return proc.returncode


def _run_loop() -> int:
    PID_FILE.write_text(str(os.getpid()), encoding="utf-8")
    _log(f"final_runner started, PID={os.getpid()}")

    state = {
        "phase": "running",
        "pid": os.getpid(),
        "started_at": _now_iso(),
        "attempts": 0,
        "consecutive_failures": 0,
    }
    _save_state(state)

    consecutive_failures = 0
    attempt = 0

    while True:
        missing = _missing_keys()
        if not missing:
            _log("All 10 combos present in checkpoint. Done.")
            state["phase"] = "done"
            state["finished_at"] = _now_iso()
            _save_state(state)
            PID_FILE.unlink(missing_ok=True)
            return 0

        attempt += 1
        state["attempts"] = attempt
        state["consecutive_failures"] = consecutive_failures
        state["missing"] = sorted(missing)
        _save_state(state)
        _log(f"Attempt {attempt}: {len(missing)} combos missing -> {sorted(missing)}")

        rc = _run_benchmark_once()

        missing_after = _missing_keys()
        progress = len(missing) - len(missing_after)
        _log(f"Progress this attempt: {progress} combos completed (now missing {len(missing_after)})")

        if not missing_after:
            _log("Checkpoint complete. Exiting cleanly.")
            state["phase"] = "done"
            state["finished_at"] = _now_iso()
            _save_state(state)
            PID_FILE.unlink(missing_ok=True)
            return 0

        if rc == 0:
            consecutive_failures = 0
            _log("benchmark exit=0 but checkpoint incomplete. Retrying immediately.")
            time.sleep(5)
            continue

        if rc == 75:
            _log(f"Rate limit hit. Sleeping {RATE_LIMIT_BACKOFF}s.")
            consecutive_failures = 0
            time.sleep(RATE_LIMIT_BACKOFF)
            continue

        consecutive_failures += 1
        _log(f"benchmark crashed (rc={rc}). Consecutive failures: {consecutive_failures}/{MAX_CONSECUTIVE_FAILURES}")

        if progress > 0:
            consecutive_failures = 0
            _log("But there was progress; resetting failure counter.")

        if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
            _log("Max consecutive failures reached. Aborting.")
            state["phase"] = "aborted_max_failures"
            state["finished_at"] = _now_iso()
            _save_state(state)
            PID_FILE.unlink(missing_ok=True)
            return 1

        time.sleep(POLL_RETRY)


def _launch_detached() -> int:
    child = [sys.executable, str(Path(__file__).resolve()), "--run"]
    flags = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP  # type: ignore[attr-defined]
    with open(LOG_PATH, "a", encoding="utf-8") as lf:
        lf.write(f"\n[{_now_iso()}] launching detached child...\n")
        lf.flush()
        proc = subprocess.Popen(
            child,
            cwd=ROOT,
            creationflags=flags,
            close_fds=True,
            stdout=lf,
            stderr=lf,
        )
    print(f"final_runner detached PID = {proc.pid}")
    print(f"  log:   {LOG_PATH}")
    print(f"  state: {STATE_PATH}")
    print(f"  pid:   {PID_FILE}")
    return 0


def _show_status() -> int:
    if STATE_PATH.exists():
        with open(STATE_PATH, "r", encoding="utf-8") as f:
            print(f.read())
    else:
        print("No state file yet.")
    if PID_FILE.exists():
        pid = int(PID_FILE.read_text().strip())
        print(f"PID {pid}: {'alive' if _pid_is_alive(pid) else 'DEAD'}")
    print(f"Missing combos: {sorted(_missing_keys())}")
    return 0


if __name__ == "__main__":
    if "--launch" in sys.argv:
        raise SystemExit(_launch_detached())
    if "--run" in sys.argv:
        raise SystemExit(_run_loop())
    if "--status" in sys.argv:
        raise SystemExit(_show_status())
    print("Usage: python final_runner.py [--launch | --run | --status]")
    raise SystemExit(1)
