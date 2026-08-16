"""Shared by orz_app.py and orz_metrics.py: per-token student/teacher KL
divergence over the top-p=0.95 union support.
"""

from __future__ import annotations

import torch

TOP_P = 0.95


def nucleus_indices(probs: torch.Tensor, top_p: float) -> torch.Tensor:
  """Smallest set of token indices (by descending prob) whose cumulative prob reaches top_p."""
  sorted_probs, sorted_idx = torch.sort(probs, descending=True)
  cumulative = torch.cumsum(sorted_probs, dim=0)
  cutoff = min(int(torch.searchsorted(cumulative, top_p).item()) + 1, sorted_idx.numel())
  return sorted_idx[:cutoff]


def per_token_kl(student_probs: torch.Tensor, teacher_probs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
  """Forward KL(teacher||student) and reverse KL(student||teacher), per token position.

  Restricted to the union of each distribution's own top-p=0.95 nucleus,
  renormalized over that union before computing KL, so a handful of
  high-probability tokens (not the full 150k-wide vocab) drive the score.
  """
  num_tokens = student_probs.shape[0]
  forward_kl = torch.zeros(num_tokens)
  reverse_kl = torch.zeros(num_tokens)
  for t in range(num_tokens):
    p = student_probs[t].float()
    q = teacher_probs[t].float()
    union = torch.unique(torch.cat([nucleus_indices(p, TOP_P), nucleus_indices(q, TOP_P)]))

    p_u = p[union]
    p_u = p_u / p_u.sum()
    q_u = q[union]
    q_u = q_u / q_u.sum()

    eps = 1e-12
    forward_kl[t] = torch.sum(q_u * torch.log((q_u + eps) / (p_u + eps)))
    reverse_kl[t] = torch.sum(p_u * torch.log((p_u + eps) / (q_u + eps)))
  return forward_kl, reverse_kl
