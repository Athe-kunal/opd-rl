"""Generates ORZ student rollouts on OpenR1-Math-220k problems, scores them
with the ORZ critic (per-token value) and ORZ teacher (per-token KL against
the student), and caches everything for opd_rl/orz_app.py to visualize.

Run with: python -m opd_rl.orz_collect
Override any opd_rl/config.yaml value from the CLI, e.g.:
  python -m opd_rl.orz_collect num_prompts=100 student_gpu=0 teacher_gpu=0
"""

from __future__ import annotations

import json
from pathlib import Path

import datasets
import requests
from omegaconf import OmegaConf

from opd_rl.process_utils import start_server_process, wait_until_ready

REPO_ROOT = Path(__file__).resolve().parent.parent
STUDENT_SERVER_NAME = "orz-student"
TEACHER_SERVER_NAME = "orz-teacher"
STUDENT_PORT = 9101
TEACHER_PORT = 9102


def sample_problems(cfg) -> list[dict[str, str]]:
  dataset = datasets.load_dataset(cfg.dataset_name, cfg.dataset_config, split="train")
  dataset = dataset.shuffle(seed=cfg.seed).select(range(cfg.num_prompts))
  return [{"problem": ex["problem"], "answer": ex["answer"]} for ex in dataset]


def load_config():
  base = OmegaConf.load(REPO_ROOT / "opd_rl" / "config.yaml")
  return OmegaConf.merge(base, OmegaConf.from_cli())


def main() -> None:
  cfg = load_config()
  output_dir = Path(cfg.output_dir).resolve()

  print(f"Sampling {cfg.num_prompts} problems (seed={cfg.seed})", flush=True)
  problems = sample_problems(cfg)
  output_dir.mkdir(parents=True, exist_ok=True)
  (output_dir / "prompts.json").write_text(json.dumps(problems))

  print("Ensuring student/teacher servers are up (reused if already running)", flush=True)
  student_url = f"http://localhost:{STUDENT_PORT}"
  teacher_url = f"http://localhost:{TEACHER_PORT}"
  start_server_process(
      STUDENT_SERVER_NAME, "orz_generate_student.py",
      [
          "--model", cfg.student_model,
          "--prompts-file", str(output_dir / "prompts.json"),
          "--output-dir", str(output_dir),
          "--responses-per-prompt", str(cfg.responses_per_prompt),
          "--max-new-tokens", str(cfg.max_new_tokens),
          "--batch-size", str(cfg.batch_size),
          "--gpu-memory-utilization", str(cfg.student_gpu_memory_fraction),
          "--teacher-url", teacher_url,
      ],
      str(cfg.student_gpu), STUDENT_PORT, output_dir,
  )
  start_server_process(
      TEACHER_SERVER_NAME, "orz_teacher_force.py",
      [
          "--model", cfg.teacher_model,
          "--student-model", cfg.student_model,
          "--critic-model", cfg.critic_model,
          "--cache-dir", str(output_dir),
      ],
      str(cfg.teacher_gpu), TEACHER_PORT, output_dir,
  )
  wait_until_ready(f"{student_url}/health")
  wait_until_ready(f"{teacher_url}/health")

  print("Generating (student sends each prompt to teacher as it finishes)", flush=True)
  requests.post(f"{student_url}/run", timeout=None).raise_for_status()

  print("Done. Servers are left running for the next job.", flush=True)


if __name__ == "__main__":
  main()
