"""Starts and reuses persistent per-GPU server processes: each orz_*.py
script is a plain aiohttp server, launched once with CUDA_VISIBLE_DEVICES
pinning it to one GPU, detached so it survives this process exiting. Reuse
is config-hash gated -- different --script-args than last time kills and
restarts instead of silently serving a stale config.
"""

from __future__ import annotations

import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import requests

REPO_ROOT = Path(__file__).resolve().parent.parent
STATE_DIR = REPO_ROOT / ".server_state"


def config_hash(script_args: list[str]) -> str:
  return hashlib.sha256(" ".join(script_args).encode()).hexdigest()[:12]


def process_alive(pid: int) -> bool:
  try:
    os.kill(pid, 0)
    return True
  except ProcessLookupError:
    return False


def _state_path(name: str) -> Path:
  return STATE_DIR / f"{name}.json"


def _read_state(name: str) -> dict | None:
  path = _state_path(name)
  if not path.exists():
    return None
  return json.loads(path.read_text())


def _write_state(name: str, pid: int, config_hash_value: str) -> None:
  STATE_DIR.mkdir(parents=True, exist_ok=True)
  _state_path(name).write_text(json.dumps({"pid": pid, "config_hash": config_hash_value}))


def start_server_process(
    name: str, script: str, script_args: list[str], gpu_id: str, port: int, log_dir: Path
) -> None:
  """Starts opd_rl/{script} as a persistent HTTP server, reused if config matches.

  Args:
    name: Identifies this server's state file (also its log filename).
    script: Filename under opd_rl/ to run.
    script_args: Extra CLI args for that script.
    gpu_id: Physical GPU index, exposed to the process via CUDA_VISIBLE_DEVICES.
    port: Port the script's HTTP server listens on.
    log_dir: Where to write {name}.log (stdout+stderr of the server process).
  """
  desired_hash = config_hash(script_args)
  state = _read_state(name)
  if state and process_alive(state["pid"]):
    if state["config_hash"] == desired_hash:
      return
    os.kill(state["pid"], signal.SIGTERM)
    deadline = time.time() + 30
    while process_alive(state["pid"]) and time.time() < deadline:
      time.sleep(0.5)

  log_dir.mkdir(parents=True, exist_ok=True)
  log_file = open(log_dir / f"{name}.log", "a")
  env = {**os.environ, "CUDA_VISIBLE_DEVICES": gpu_id}
  script_path = REPO_ROOT / "opd_rl" / script
  process = subprocess.Popen(
      [sys.executable, str(script_path), *script_args, "--port", str(port)],
      env=env, stdout=log_file, stderr=subprocess.STDOUT, start_new_session=True,
  )
  time.sleep(2)
  if process.poll() is not None:
    raise RuntimeError(f"{name} exited immediately (code {process.returncode}); see {log_dir / f'{name}.log'}")
  _write_state(name, process.pid, desired_hash)


def wait_until_ready(url: str, timeout_s: int = 300) -> None:
  """Blocks until `url` accepts connections, or raises after `timeout_s`."""
  deadline = time.time() + timeout_s
  while time.time() < deadline:
    try:
      requests.get(url, timeout=10)
      return
    except requests.exceptions.RequestException:
      time.sleep(2)
  raise TimeoutError(f"{url} did not become ready in {timeout_s}s")
