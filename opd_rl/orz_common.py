"""Shared by the orz_* scripts: ORZ's training prompt template (Table 1 of
arxiv.org/abs/2503.24290) plus boxed-answer extraction and math_verify
reward scoring.
"""

from __future__ import annotations

import re

from math_verify import parse, verify

_SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and "
    "the Assistant solves it. The assistant first thinks about the reasoning "
    "process in the mind and then provides the user with the answer. The "
    "reasoning process and answer are enclosed within <think> </think> and "
    "<answer> </answer> tags, respectively, i.e., <think> reasoning process "
    "here </think> <answer> answer here </answer>.\n"
)


def render_prompt(problem: str) -> str:
  """Builds the ORZ-style prompt for one math problem, ready to tokenize."""
  return (
      _SYSTEM_PROMPT
      + "User: You must put your answer inside <answer> </answer> tags, i.e., "
      "<answer> answer here </answer>. And your final answer will be extracted "
      f"automatically by the \\boxed{{}} tag. {problem}\n"
      "Assistant: <think>"
  )


def extract_boxed_answer(text: str) -> str | None:
  """Returns the contents of the last \\boxed{...} in `text`, handling nested braces."""
  matches = list(re.finditer(r"\\boxed\{", text))
  if not matches:
    return None
  start = matches[-1].end()
  depth = 1
  for i in range(start, len(text)):
    if text[i] == "{":
      depth += 1
    elif text[i] == "}":
      depth -= 1
      if depth == 0:
        return text[start:i]
  return None


def compute_reward(response_text: str, ground_truth_answer: str) -> float:
  """1.0 if the boxed answer in `response_text` matches `ground_truth_answer`, else 0.0."""
  predicted = extract_boxed_answer(response_text)
  if predicted is None:
    return 0.0
  try:
    # parsing_timeout/timeout_seconds=None: both parse() and verify() default
    # to a signal.alarm()-based timeout, which only works on the main thread.
    # We call this from loop.run_in_executor's worker thread, where it raised
    # every time and got swallowed by this except, silently scoring every
    # reward 0.0.
    gold = parse(ground_truth_answer, parsing_timeout=None)
    pred = parse(predicted, parsing_timeout=None)
    return 1.0 if verify(gold, pred, timeout_seconds=None) else 0.0
  except Exception:
    return 0.0
