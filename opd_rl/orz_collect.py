"""Generates ORZ student rollouts on OpenR1-Math-220k problems, scores them
with the ORZ critic (per-token value) and ORZ teacher (per-token KL against
the student), and caches everything for opd_rl/orz_app.py to visualize.

Run with: python -m opd_rl.orz_collect
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import datasets
import requests

from opd_rl.process_utils import start_server_process, wait_until_ready

STUDENT_MODEL = "Open-Reasoner-Zero/Open-Reasoner-Zero-1.5B"
CRITIC_MODEL = "Open-Reasoner-Zero/Open-Reasoner-Zero-Critic-1.5B"
TEACHER_MODEL = "Open-Reasoner-Zero/Open-Reasoner-Zero-7B"
DATASET_NAME = "open-r1/OpenR1-Math-220k"
DATASET_CONFIG = "default"

STUDENT_GPU_ID = "2"
TEACHER_GPU_ID = "3"
STUDENT_GPU_MEMORY_FRACTION = 0.5  # leaves headroom for the self-score + critic model copies

STUDENT_SERVER_NAME = "orz-student"
TEACHER_SERVER_NAME = "orz-teacher"
STUDENT_PORT = 9101
TEACHER_PORT = 9102


def sample_problems(num_prompts: int, seed: int) -> list[dict[str, str]]:
  """Samples random (problem, answer) pairs from OpenR1-Math-220k."""
  dataset = datasets.load_dataset(DATASET_NAME, DATASET_CONFIG, split="train")
  dataset = dataset.shuffle(seed=seed).select(range(num_prompts))
  return [{"problem": ex["problem"], "answer": ex["answer"]} for ex in dataset]


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--num-prompts", type=int, default=20)
  parser.add_argument("--responses-per-prompt", type=int, default=4)
  parser.add_argument("--max-new-tokens", type=int, default=4096)
  parser.add_argument("--batch-size", type=int, default=20)
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument("--output-dir", type=Path, default=Path("opd_rl/data/orz_value_cache"))
  return parser.parse_args()


def main() -> None:
  args = parse_args()
  output_dir = args.output_dir.resolve()

  print(f"Sampling {args.num_prompts} problems (seed={args.seed})", flush=True)
  problems = sample_problems(args.num_prompts, args.seed)
  output_dir.mkdir(parents=True, exist_ok=True)
  (output_dir / "prompts.json").write_text(json.dumps(problems))

  print("Ensuring student/teacher servers are up (reused if already running)", flush=True)
  student_url = f"http://localhost:{STUDENT_PORT}"
  teacher_url = f"http://localhost:{TEACHER_PORT}"
  start_server_process(
      STUDENT_SERVER_NAME, "orz_generate_student.py",
      [
          "--model", STUDENT_MODEL,
          "--critic-model", CRITIC_MODEL,
          "--prompts-file", str(output_dir / "prompts.json"),
          "--output-dir", str(output_dir),
          "--responses-per-prompt", str(args.responses_per_prompt),
          "--max-new-tokens", str(args.max_new_tokens),
          "--batch-size", str(args.batch_size),
          "--gpu-memory-utilization", str(STUDENT_GPU_MEMORY_FRACTION),
          "--teacher-url", teacher_url,
      ],
      STUDENT_GPU_ID, STUDENT_PORT, output_dir,
  )
  start_server_process(
      TEACHER_SERVER_NAME, "orz_teacher_force.py",
      ["--model", TEACHER_MODEL, "--cache-dir", str(output_dir)],
      TEACHER_GPU_ID, TEACHER_PORT, output_dir,
  )
  wait_until_ready(f"{student_url}/health")
  wait_until_ready(f"{teacher_url}/health")

  print("Generating (student sends each prompt to teacher as it finishes)", flush=True)
  requests.post(f"{student_url}/run", timeout=None).raise_for_status()

  print("Done. Servers are left running for the next job.", flush=True)


if __name__ == "__main__":
  main()
