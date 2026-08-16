"""Shared by orz_generate_student.py and orz_teacher_force.py.

score_response_text computes a full-vocab softmax via a direct transformers
forward pass, not vLLM's logprobs/prompt_logprobs API, which materializes a
Python dict per position -- slow and memory-hungry at full-vocab scale.
"""

from __future__ import annotations

import datetime
from pathlib import Path

import torch


def log_progress(output_dir: Path, message: str) -> None:
  timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
  with open(output_dir / "progress.log", "a") as f:
    f.write(f"[{timestamp}] {message}\n")


@torch.inference_mode()
def score_response_text(model, tokenizer, prompt_text: str, response_token_ids: list[int]) -> torch.Tensor:
  """Forces `response_token_ids` after `prompt_text` and returns their [num_tokens, vocab_size] float16 softmax."""
  prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
  input_ids = torch.tensor([prompt_ids + response_token_ids], device=model.device)

  logits_to_keep = len(response_token_ids) + 1
  logits = model(input_ids=input_ids, logits_to_keep=logits_to_keep).logits[0]
  response_logits = logits[:len(response_token_ids)].float()
  return torch.softmax(response_logits, dim=-1).to(torch.float16)
