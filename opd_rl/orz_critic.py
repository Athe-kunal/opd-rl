"""Loads and scores with the ORZ critic checkpoint's OpenRLHF value-head
wrapper (a Qwen2.5 backbone plus a linear `value_head`, confirmed via the
checkpoint's safetensors header).

openrlhf.models unconditionally imports flash_attn, which we don't have and
don't need (single-sequence scoring, no ring attention) -- _stub_flash_attn
registers a fake module with just enough surface for that import to succeed.
"""

from __future__ import annotations

import importlib.machinery
import sys
import types

import torch
from transformers import AutoTokenizer


def _stub_flash_attn() -> None:
  if "flash_attn" in sys.modules:
    return

  def _unavailable(*args, **kwargs):
    raise RuntimeError("flash_attn is stubbed out")

  def _make_module(name: str) -> types.ModuleType:
    module = types.ModuleType(name)
    module.__spec__ = importlib.machinery.ModuleSpec(name, loader=None)
    return module

  flash_attn = _make_module("flash_attn")
  flash_attn.__version__ = "0.0.0"
  bert_padding = _make_module("flash_attn.bert_padding")
  for name in ["index_first_axis", "pad_input", "rearrange", "unpad_input"]:
    setattr(bert_padding, name, _unavailable)
  utils = _make_module("flash_attn.utils")
  distributed = _make_module("flash_attn.utils.distributed")
  distributed.all_gather = _unavailable
  flash_attn.bert_padding = bert_padding
  flash_attn.utils = utils
  utils.distributed = distributed

  sys.modules["flash_attn"] = flash_attn
  sys.modules["flash_attn.bert_padding"] = bert_padding
  sys.modules["flash_attn.utils"] = utils
  sys.modules["flash_attn.utils.distributed"] = distributed


_stub_flash_attn()
from openrlhf.models import get_llm_for_sequence_regression  # noqa: E402


def load_critic(model_name: str):
  tokenizer = AutoTokenizer.from_pretrained(model_name)
  model = get_llm_for_sequence_regression(
      model_name,
      "critic",
      param_dtype="bf16",
      normalize_reward=False,
      attn_implementation="sdpa",
      value_head_prefix="value_head",
      init_value_head=False,
      ds_config=None,
      device_map="cuda",
  ).eval()
  return tokenizer, model


@torch.inference_mode()
def score_response(model, tokenizer, prompt_text: str, response_token_ids: list[int]) -> torch.Tensor:
  """Returns the critic's per-token value for `response_token_ids`, shape [num_tokens], float16."""
  prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
  sequence = torch.tensor([prompt_ids + response_token_ids], device=model.device)
  attention_mask = torch.ones_like(sequence)
  action_mask = torch.ones((1, len(response_token_ids)), device=model.device)
  values = model(input_ids=sequence, action_mask=action_mask, attention_mask=attention_mask)
  return values[0].to(torch.float16).cpu()
