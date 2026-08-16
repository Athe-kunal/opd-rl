"""Validates whether student/teacher KL divergence could substitute for the
learned critic: per-token Spearman correlation between critic value and KL,
and trajectory-level AUROC of mean KL as a predictor of the math_verify
reward.

Run with: python -m opd_rl.orz_metrics [--cache-dir ...]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def spearman(x: torch.Tensor, y: torch.Tensor) -> float:
  """Spearman rank correlation, as Pearson correlation of ranks."""
  return pearson(x.argsort().argsort().float(), y.argsort().argsort().float())


def pearson(x: torch.Tensor, y: torch.Tensor) -> float:
  x = x - x.mean()
  y = y - y.mean()
  denom = x.norm() * y.norm()
  return float((x @ y) / denom) if denom > 0 else float("nan")


def auroc(scores: torch.Tensor, labels: torch.Tensor) -> float:
  """AUROC via the Mann-Whitney U / rank-sum formula. 0.5 = no better than chance."""
  n_pos = int(labels.sum())
  n_neg = int((1 - labels).sum())
  if n_pos == 0 or n_neg == 0:
    return float("nan")
  ranks = scores.argsort().argsort().float() + 1
  rank_sum_pos = ranks[labels.bool()].sum()
  u = rank_sum_pos - n_pos * (n_pos + 1) / 2
  return float(u / (n_pos * n_neg))


def load_samples(cache_dir: Path) -> list[dict]:
  samples = []
  for sample_dir in sorted(cache_dir.iterdir(), key=lambda p: int(p.name) if p.name.isdigit() else -1):
    meta_path = sample_dir / "meta.json"
    if sample_dir.is_dir() and meta_path.exists():
      meta = json.loads(meta_path.read_text())
      meta["_dir"] = sample_dir
      samples.append(meta)
  return samples


def compute_metrics(cache_dir: Path) -> dict[str, float]:
  all_critic, all_forward_kl, all_reverse_kl = [], [], []
  mean_forward_kl, mean_reverse_kl, rewards = [], [], []

  for sample in load_samples(cache_dir):
    for resp in sample["responses"]:
      sample_dir = sample["_dir"]
      critic_values = torch.load(sample_dir / resp["critic_values_path"]).float()
      forward_kl = torch.load(sample_dir / resp["forward_kl_path"])
      reverse_kl = torch.load(sample_dir / resp["reverse_kl_path"])

      all_critic.append(critic_values)
      all_forward_kl.append(forward_kl)
      all_reverse_kl.append(reverse_kl)
      mean_forward_kl.append(forward_kl.mean())
      mean_reverse_kl.append(reverse_kl.mean())
      rewards.append(resp["reward"])

  all_critic = torch.cat(all_critic)
  all_forward_kl = torch.cat(all_forward_kl)
  all_reverse_kl = torch.cat(all_reverse_kl)
  mean_forward_kl = torch.tensor(mean_forward_kl)
  mean_reverse_kl = torch.tensor(mean_reverse_kl)
  rewards = torch.tensor(rewards)

  return {
      "num_responses": len(mean_forward_kl),
      "num_tokens": len(all_critic),
      "spearman(critic_value, forward_kl)": spearman(all_critic, all_forward_kl),
      "spearman(critic_value, reverse_kl)": spearman(all_critic, all_reverse_kl),
      # Lower KL should mean higher reward, so score by -mean_kl.
      "auroc(-mean_forward_kl -> reward)": auroc(-mean_forward_kl, rewards),
      "auroc(-mean_reverse_kl -> reward)": auroc(-mean_reverse_kl, rewards),
  }


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--cache-dir", type=Path, default=Path("opd_rl/data/orz_value_cache"))
  return parser.parse_args()


def main() -> None:
  args = parse_args()
  metrics = compute_metrics(args.cache_dir.resolve())
  for name, value in metrics.items():
    print(f"{name}: {value}")


if __name__ == "__main__":
  main()
