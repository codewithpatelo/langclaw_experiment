"""Detached supervisor for full LangClaw experiment.

Runs:
  1) calibrate_hyperparams.py
  2) benchmark.py

Designed for long runs that must survive IDE crashes/disconnects.
Use --detach to spawn a background worker process on Windows.

Pause semantics:
  - If calibration or benchmark exits with code 75 (rate/quota limit),
    worker status is marked as "paused_rate_limit" and can be resumed by
    launching the same command again after restoring quota.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _pid_is_alive(pid: int | None) -> bool:
    """Best-effort process liveness check."""
    if pid is None or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _append_event(path: Path, event: dict[str, Any]) -> None:
    """Append one JSON event line to the experiment journal."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")


def _write_state_banner(path: Path, *, state: str, phase: str, message: str) -> None:
    """Write a human-readable single-file current state."""
    ts = _now_iso()
    lines = [
        f"STATE: {state}",
        f"PHASE: {phase}",
        f"UPDATED_AT: {ts}",
        f"MESSAGE: {message}",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def _append_event(path: Path, event: dict[str, Any]) -> None:
    """Append one JSON event line to the experiment journal."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")


def _write_state_banner(path: Path, *, state: str, phase: str, message: str) -> None:
    """Write a human-readable single-file current state."""
    ts = _now_iso()
    lines = [
        f"STATE: {state}",
        f"PHASE: {phase}",
        f"UPDATED_AT: {ts}",
        f"MESSAGE: {message}",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def _build_calibration_cmd(args: argparse.Namespace) -> list[str]:
    return [
        sys.executable,
        "calibrate_hyperparams.py",
        "--ticks",
        str(args.calibration_ticks),
        "--seed",
        str(args.calibration_seed),
        "--api-hard-limit",
        str(args.calibration_api_hard_limit),
        "--log-level",
        args.log_level,
        "--output",
        args.calibration_output,
    ]


def _build_benchmark_cmd(args: argparse.Namespace) -> list[str]:
    return [
        sys.executable,
        "benchmark.py",
        "--iterations",
        str(args.iterations),
        "--seeds",
        *[str(s) for s in args.seeds],
        "--modes",
        "hrrl",
        "langgraph",
        "--config",
        args.calibration_output,
        "--output-dir",
        args.benchmark_output_dir,
        "--api-hard-limit",
        str(args.benchmark_api_hard_limit),
        "--log-level",
        args.log_level,
    ]


def _build_preflight_cmd(args: argparse.Namespace) -> list[str]:
    return [
        sys.executable,
        "benchmark.py",
        "--preflight",
        "--preflight-ticks",
        str(args.preflight_ticks),
        "--seeds",
        str(args.seeds[0]),
        "--modes",
        "hrrl",
        "langgraph",
        "--config",
        args.calibration_output,
        "--output-dir",
        args.benchmark_output_dir,
        "--api-hard-limit",
        str(args.benchmark_api_hard_limit),
        "--log-level",
        args.log_level,
    ]


def _run_worker(args: argparse.Namespace) -> int:
    root = Path(args.project_root).resolve()
    status_path = (root / args.status_file).resolve()
    log_path = (root / args.log_file).resolve()
    events_path = (root / args.events_file).resolve()
    state_path = (root / args.state_file).resolve()

    calibration_cmd = _build_calibration_cmd(args)
    preflight_cmd = _build_preflight_cmd(args)
    benchmark_cmd = _build_benchmark_cmd(args)
    child_env = os.environ.copy()
    child_env["PYTHONIOENCODING"] = "utf-8"
    child_env["PYTHONUTF8"] = "1"

    prev_status = _read_json(status_path)
    status = {
        "phase": "starting",
        "started_at": prev_status.get("started_at", _now_iso()),
        "updated_at": _now_iso(),
        "worker_pid": os.getpid(),
        "project_root": str(root),
        "commands": {
            "calibration": calibration_cmd,
            "preflight": preflight_cmd,
            "benchmark": benchmark_cmd,
        },
        "calibration": prev_status.get("calibration", {"done": False, "returncode": None}),
        "preflight": prev_status.get("preflight", {"done": False, "returncode": None}),
        "benchmark": prev_status.get("benchmark", {"done": False, "returncode": None}),
        "finished": False,
        "success": False,
        "error": None,
    }
    _write_json(status_path, status)
    _append_event(
        events_path,
        {
            "timestamp": _now_iso(),
            "event": "worker_started",
            "phase": status["phase"],
            "worker_pid": os.getpid(),
        },
    )
    _write_state_banner(
        state_path,
        state="RUNNING",
        phase=status["phase"],
        message="Worker started.",
    )

    with open(log_path, "a", encoding="utf-8") as log_file:
        log_file.write(f"\n[{_now_iso()}] Worker PID {os.getpid()} started\n")
        log_file.flush()

        calibration_output_path = (root / args.calibration_output).resolve()
        skip_calibration = (
            status["calibration"].get("done") is True
            and status["calibration"].get("returncode") == 0
            and calibration_output_path.exists()
        )
        if skip_calibration:
            _append_event(
                events_path,
                {
                    "timestamp": _now_iso(),
                    "event": "phase_skipped",
                    "phase": "calibration",
                    "message": "Calibration output already present; reusing previous result.",
                },
            )
        else:
            status["phase"] = "calibration"
            status["updated_at"] = _now_iso()
            _write_json(status_path, status)
            _append_event(
                events_path,
                {
                    "timestamp": _now_iso(),
                    "event": "phase_started",
                    "phase": "calibration",
                    "message": "Calibration phase started.",
                },
            )
            _write_state_banner(
                state_path,
                state="RUNNING",
                phase="calibration",
                message="Calibration in progress.",
            )
            cal_proc = subprocess.run(
                calibration_cmd,
                cwd=root,
                env=child_env,
                stdout=log_file,
                stderr=log_file,
                text=True,
                check=False,
            )
            status["calibration"]["done"] = True
            status["calibration"]["returncode"] = cal_proc.returncode
            status["updated_at"] = _now_iso()
            _write_json(status_path, status)
            _append_event(
                events_path,
                {
                    "timestamp": _now_iso(),
                    "event": "phase_finished",
                    "phase": "calibration",
                    "returncode": cal_proc.returncode,
                },
            )
            if cal_proc.returncode == 75:
                status["phase"] = "paused_rate_limit"
                status["finished"] = False
                status["success"] = False
                status["error"] = "paused during calibration due to API rate/quota limit"
                status["updated_at"] = _now_iso()
                _write_json(status_path, status)
                _append_event(
                    events_path,
                    {
                        "timestamp": _now_iso(),
                        "event": "paused_rate_limit",
                        "phase": "calibration",
                        "message": status["error"],
                    },
                )
                _write_state_banner(
                    state_path,
                    state="PAUSED_RATE_LIMIT",
                    phase="calibration",
                    message=status["error"],
                )
                log_file.write(f"[{_now_iso()}] Worker paused (rate/quota) in calibration\n")
                log_file.flush()
                return 75
            if cal_proc.returncode != 0:
                status["phase"] = "failed"
                status["finished"] = True
                status["success"] = False
                status["error"] = f"calibration failed with code {cal_proc.returncode}"
                status["updated_at"] = _now_iso()
                _write_json(status_path, status)
                _append_event(
                    events_path,
                    {
                        "timestamp": _now_iso(),
                        "event": "failed",
                        "phase": "calibration",
                        "message": status["error"],
                        "returncode": cal_proc.returncode,
                    },
                )
                _write_state_banner(
                    state_path,
                    state="FAILED",
                    phase="calibration",
                    message=status["error"],
                )
                return cal_proc.returncode

        # Preflight
        preflight_report_path = (
            root / args.benchmark_output_dir / "preflight" / f"preflight_seed{args.seeds[0]}.json"
        ).resolve()
        skip_preflight = (
            status["preflight"].get("done") is True
            and status["preflight"].get("returncode") == 0
            and preflight_report_path.exists()
        )
        if skip_preflight:
            _append_event(
                events_path,
                {
                    "timestamp": _now_iso(),
                    "event": "phase_skipped",
                    "phase": "preflight",
                    "message": "Preflight report already present; reusing previous result.",
                },
            )
        else:
            status["phase"] = "preflight"
            status["updated_at"] = _now_iso()
            _write_json(status_path, status)
            _append_event(
                events_path,
                {
                    "timestamp": _now_iso(),
                    "event": "phase_started",
                    "phase": "preflight",
                    "message": "Preflight phase started.",
                },
            )
            _write_state_banner(
                state_path,
                state="RUNNING",
                phase="preflight",
                message="Preflight in progress.",
            )
            pf_proc = subprocess.run(
                preflight_cmd,
                cwd=root,
                env=child_env,
                stdout=log_file,
                stderr=log_file,
                text=True,
                check=False,
            )
            status["preflight"]["done"] = True
            status["preflight"]["returncode"] = pf_proc.returncode
            status["updated_at"] = _now_iso()
            _write_json(status_path, status)
            _append_event(
                events_path,
                {
                    "timestamp": _now_iso(),
                    "event": "phase_finished",
                    "phase": "preflight",
                    "returncode": pf_proc.returncode,
                },
            )
            if pf_proc.returncode == 75:
                status["phase"] = "paused_rate_limit"
                status["finished"] = False
                status["success"] = False
                status["error"] = "paused during preflight due to API rate/quota limit"
                status["updated_at"] = _now_iso()
                _write_json(status_path, status)
                _append_event(
                    events_path,
                    {
                        "timestamp": _now_iso(),
                        "event": "paused_rate_limit",
                        "phase": "preflight",
                        "message": status["error"],
                    },
                )
                _write_state_banner(
                    state_path,
                    state="PAUSED_RATE_LIMIT",
                    phase="preflight",
                    message=status["error"],
                )
                log_file.write(f"[{_now_iso()}] Worker paused (rate/quota) in preflight\n")
                log_file.flush()
                return 75
            if pf_proc.returncode == 86:
                status["phase"] = "needs_review"
                status["finished"] = True
                status["success"] = False
                status["error"] = "preflight detected critical red flags; inspect report before benchmarking"
                status["updated_at"] = _now_iso()
                _write_json(status_path, status)
                _append_event(
                    events_path,
                    {
                        "timestamp": _now_iso(),
                        "event": "paused_review",
                        "phase": "preflight",
                        "message": status["error"],
                    },
                )
                _write_state_banner(
                    state_path,
                    state="NEEDS_REVIEW",
                    phase="preflight",
                    message=status["error"],
                )
                log_file.write(f"[{_now_iso()}] Worker halted for review in preflight\n")
                log_file.flush()
                return 86
            if pf_proc.returncode != 0:
                status["phase"] = "failed"
                status["finished"] = True
                status["success"] = False
                status["error"] = f"preflight failed with code {pf_proc.returncode}"
                status["updated_at"] = _now_iso()
                _write_json(status_path, status)
                _append_event(
                    events_path,
                    {
                        "timestamp": _now_iso(),
                        "event": "failed",
                        "phase": "preflight",
                        "message": status["error"],
                        "returncode": pf_proc.returncode,
                    },
                )
                _write_state_banner(
                    state_path,
                    state="FAILED",
                    phase="preflight",
                    message=status["error"],
                )
                return pf_proc.returncode

        # Benchmark
        status["phase"] = "benchmark"
        status["updated_at"] = _now_iso()
        _write_json(status_path, status)
        _append_event(
            events_path,
            {
                "timestamp": _now_iso(),
                "event": "phase_started",
                "phase": "benchmark",
                "message": "Benchmark phase started.",
            },
        )
        _write_state_banner(
            state_path,
            state="RUNNING",
            phase="benchmark",
            message="Benchmark in progress.",
        )
        bm_proc = subprocess.run(
            benchmark_cmd,
            cwd=root,
            env=child_env,
            stdout=log_file,
            stderr=log_file,
            text=True,
            check=False,
        )
        status["benchmark"]["done"] = True
        status["benchmark"]["returncode"] = bm_proc.returncode
        status["updated_at"] = _now_iso()
        _write_json(status_path, status)
        _append_event(
            events_path,
            {
                "timestamp": _now_iso(),
                "event": "phase_finished",
                "phase": "benchmark",
                "returncode": bm_proc.returncode,
            },
        )
        if bm_proc.returncode == 75:
            status["phase"] = "paused_rate_limit"
            status["finished"] = False
            status["success"] = False
            status["error"] = "paused during benchmark due to API rate/quota limit"
            status["updated_at"] = _now_iso()
            _write_json(status_path, status)
            _append_event(
                events_path,
                {
                    "timestamp": _now_iso(),
                    "event": "paused_rate_limit",
                    "phase": "benchmark",
                    "message": status["error"],
                },
            )
            _write_state_banner(
                state_path,
                state="PAUSED_RATE_LIMIT",
                phase="benchmark",
                message=status["error"],
            )
            log_file.write(f"[{_now_iso()}] Worker paused (rate/quota) in benchmark\n")
            log_file.flush()
            return 75
        if bm_proc.returncode == 86:
            status["phase"] = "needs_review"
            status["finished"] = True
            status["success"] = False
            status["error"] = "benchmark halted due to critical red flags; inspect health report"
            status["updated_at"] = _now_iso()
            _write_json(status_path, status)
            _append_event(
                events_path,
                {
                    "timestamp": _now_iso(),
                    "event": "paused_review",
                    "phase": "benchmark",
                    "message": status["error"],
                },
            )
            _write_state_banner(
                state_path,
                state="NEEDS_REVIEW",
                phase="benchmark",
                message=status["error"],
            )
            log_file.write(f"[{_now_iso()}] Worker halted for review in benchmark\n")
            log_file.flush()
            return 86
        if bm_proc.returncode != 0:
            status["phase"] = "failed"
            status["finished"] = True
            status["success"] = False
            status["error"] = f"benchmark failed with code {bm_proc.returncode}"
            status["updated_at"] = _now_iso()
            _write_json(status_path, status)
            _append_event(
                events_path,
                {
                    "timestamp": _now_iso(),
                    "event": "failed",
                    "phase": "benchmark",
                    "message": status["error"],
                    "returncode": bm_proc.returncode,
                },
            )
            _write_state_banner(
                state_path,
                state="FAILED",
                phase="benchmark",
                message=status["error"],
            )
            return bm_proc.returncode

        status["phase"] = "completed"
        status["finished"] = True
        status["success"] = True
        status["updated_at"] = _now_iso()
        _write_json(status_path, status)
        _append_event(
            events_path,
            {
                "timestamp": _now_iso(),
                "event": "completed",
                "phase": "completed",
                "message": "Full experiment completed successfully.",
            },
        )
        _write_state_banner(
            state_path,
            state="COMPLETED",
            phase="completed",
            message="Full experiment completed successfully.",
        )
        log_file.write(f"[{_now_iso()}] Worker completed successfully\n")
        log_file.flush()

    return 0


def _run_detached(args: argparse.Namespace) -> int:
    root = Path(args.project_root).resolve()
    status_path = (root / args.status_file).resolve()
    events_path = (root / args.events_file).resolve()

    child_args = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--project-root",
        str(root),
        "--calibration-ticks",
        str(args.calibration_ticks),
        "--calibration-seed",
        str(args.calibration_seed),
        "--calibration-api-hard-limit",
        str(args.calibration_api_hard_limit),
        "--calibration-output",
        args.calibration_output,
        "--preflight-ticks",
        str(args.preflight_ticks),
        "--iterations",
        str(args.iterations),
        "--seeds",
        *[str(s) for s in args.seeds],
        "--benchmark-api-hard-limit",
        str(args.benchmark_api_hard_limit),
        "--benchmark-output-dir",
        args.benchmark_output_dir,
        "--status-file",
        args.status_file,
        "--log-file",
        args.log_file,
        "--log-level",
        args.log_level,
        "--events-file",
        args.events_file,
        "--state-file",
        args.state_file,
    ]

    creationflags = 0
    if os.name == "nt":
        # Keep running independently from the launching terminal.
        creationflags = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP

    proc = subprocess.Popen(
        child_args,
        cwd=root,
        creationflags=creationflags,
        close_fds=True,
    )

    payload = _read_json(status_path)
    payload["launcher_pid"] = os.getpid()
    payload["detached_worker_pid"] = proc.pid
    payload["updated_at"] = _now_iso()
    _write_json(status_path, payload)
    _append_event(
        events_path,
        {
            "timestamp": _now_iso(),
            "event": "detached_started",
            "phase": payload.get("phase", "starting"),
            "launcher_pid": os.getpid(),
            "detached_worker_pid": proc.pid,
        },
    )

    print(f"Detached worker started with PID {proc.pid}")
    print(f"Status file: {status_path}")
    print(f"Log file: {(root / args.log_file).resolve()}")
    print(f"Events file: {(root / args.events_file).resolve()}")
    print(f"State file: {(root / args.state_file).resolve()}")
    return 0


def _watchdog_check(args: argparse.Namespace) -> int:
    """Hourly supervisor check: restart only when safe and necessary."""
    root = Path(args.project_root).resolve()
    status_path = (root / args.status_file).resolve()
    events_path = (root / args.events_file).resolve()

    status = _read_json(status_path)
    if not status:
        _append_event(
            events_path,
            {
                "timestamp": _now_iso(),
                "event": "watchdog_no_status",
                "message": "No experiment status file found; starting detached worker.",
            },
        )
        return _run_detached(args)

    worker_pid = status.get("worker_pid")
    if _pid_is_alive(worker_pid):
        _append_event(
            events_path,
            {
                "timestamp": _now_iso(),
                "event": "watchdog_skip_running",
                "worker_pid": worker_pid,
                "phase": status.get("phase"),
                "message": "Worker is still running; watchdog took no action.",
            },
        )
        return 0

    if status.get("finished") and status.get("success"):
        _append_event(
            events_path,
            {
                "timestamp": _now_iso(),
                "event": "watchdog_skip_completed",
                "phase": status.get("phase"),
                "message": "Experiment already completed successfully.",
            },
        )
        return 0

    if status.get("finished") and not status.get("success"):
        _append_event(
            events_path,
            {
                "timestamp": _now_iso(),
                "event": "watchdog_skip_failed",
                "phase": status.get("phase"),
                "worker_pid": worker_pid,
                "error": status.get("error"),
                "message": "Experiment is marked failed; watchdog did not auto-restart.",
            },
        )
        return 1

    _append_event(
        events_path,
        {
            "timestamp": _now_iso(),
            "event": "watchdog_restart",
            "phase": status.get("phase"),
            "worker_pid": worker_pid,
            "message": "Worker not running and experiment not finished; restarting detached worker.",
        },
    )
    return _run_detached(args)


def _print_status(args: argparse.Namespace) -> int:
    root = Path(args.project_root).resolve()
    status_path = (root / args.status_file).resolve()
    if not status_path.exists():
        print(f"No status file found at {status_path}")
        return 1
    data = _read_json(status_path)
    print(json.dumps(data, indent=2, ensure_ascii=False))
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detached full experiment runner (calibration + benchmark)."
    )
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--calibration-ticks", type=int, default=10)
    parser.add_argument("--calibration-seed", type=int, default=42)
    parser.add_argument("--calibration-api-hard-limit", type=int, default=200)
    parser.add_argument("--calibration-output", default="calibration_results.json")
    parser.add_argument("--preflight-ticks", type=int, default=12)
    parser.add_argument("--iterations", type=int, default=80)
    parser.add_argument("--seeds", nargs="+", type=int, default=[7, 17, 42, 123, 256])
    parser.add_argument("--benchmark-api-hard-limit", type=int, default=500)
    parser.add_argument("--benchmark-output-dir", default="benchmark_results_v7")
    parser.add_argument("--status-file", default="experiment_status.json")
    parser.add_argument("--log-file", default="experiment_run.log")
    parser.add_argument("--events-file", default="experiment_events.jsonl")
    parser.add_argument("--state-file", default="experiment_state.txt")
    parser.add_argument(
        "--log-level",
        default="WARNING",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )

    parser.add_argument("--detach", action="store_true")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--watchdog-check", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.status:
        return _print_status(args)
    if args.watchdog_check:
        return _watchdog_check(args)
    if args.detach:
        return _run_detached(args)
    if args.worker:
        return _run_worker(args)
    # Foreground execution
    return _run_worker(args)


if __name__ == "__main__":
    raise SystemExit(main())

