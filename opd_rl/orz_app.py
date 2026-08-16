"""Streamlit app for inspecting orz_collect.py's cached rollouts: response
text with tokens highlighted by critic value or student/teacher KL
divergence, plus math_verify reward.

Run with: streamlit run opd_rl/orz_app.py -- --cache-dir opd_rl/data/orz_value_cache
"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import streamlit as st
import torch
from transformers import AutoTokenizer

STUDENT_MODEL = "Open-Reasoner-Zero/Open-Reasoner-Zero-1.5B"

# Sequential white -> dark orange scale, so highlight intensity reads as "how much" regardless of sign.
_COLOR_LOW = (255, 255, 255)
_COLOR_HIGH = (217, 72, 1)


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  parser.add_argument("--cache-dir", type=Path, default=Path("opd_rl/data/orz_value_cache"))
  return parser.parse_args()


@st.cache_resource
def load_tokenizer():
  return AutoTokenizer.from_pretrained(STUDENT_MODEL)


@st.cache_data
def load_samples(cache_dir: str) -> list[dict]:
  cache_dir = Path(cache_dir)
  samples = []
  for sample_dir in sorted(cache_dir.iterdir(), key=lambda p: int(p.name) if p.name.isdigit() else -1):
    meta_path = sample_dir / "meta.json"
    if sample_dir.is_dir() and meta_path.exists():
      meta = json.loads(meta_path.read_text())
      meta["_dir"] = str(sample_dir)
      samples.append(meta)
  return samples


def token_texts(tokenizer, token_ids: list[int]) -> list[str]:
  tokens = tokenizer.convert_ids_to_tokens(token_ids)
  return [tokenizer.convert_tokens_to_string([t]) for t in tokens]


def value_to_color(norm_value: float) -> str:
  r = round(_COLOR_LOW[0] + (_COLOR_HIGH[0] - _COLOR_LOW[0]) * norm_value)
  g = round(_COLOR_LOW[1] + (_COLOR_HIGH[1] - _COLOR_LOW[1]) * norm_value)
  b = round(_COLOR_LOW[2] + (_COLOR_HIGH[2] - _COLOR_LOW[2]) * norm_value)
  return f"#{r:02x}{g:02x}{b:02x}"


def render_highlighted_tokens(tokens: list[str], values: torch.Tensor) -> None:
  vmin, vmax = float(values.min()), float(values.max())
  spread = vmax - vmin or 1.0
  spans = []
  for token, value in zip(tokens, values.tolist()):
    norm_value = (value - vmin) / spread
    color = value_to_color(norm_value)
    text = html.escape(token).replace("\n", "<br>")
    spans.append(
        f'<span style="background-color:{color}; border-radius:2px;" '
        f'title="{value:.4f}">{text}</span>'
    )
  st.markdown(
      f'<div style="white-space:pre-wrap; line-height:1.8; font-family:monospace;">{"".join(spans)}</div>',
      unsafe_allow_html=True,
  )


def render_sample(sample: dict, response_idx: int) -> None:
  resp = sample["responses"][response_idx]
  sample_dir = Path(sample["_dir"])

  critic_values = torch.load(sample_dir / resp["critic_values_path"]).float()
  forward_kl = torch.load(sample_dir / resp["forward_kl_path"])
  reverse_kl = torch.load(sample_dir / resp["reverse_kl_path"])

  st.metric("Reward (math_verify)", resp["reward"])

  metrics = {
      "critic value": critic_values,
      "KL(teacher||student)": forward_kl,
      "KL(student||teacher)": reverse_kl,
  }

  tokenizer = load_tokenizer()
  tokens = token_texts(tokenizer, resp["token_ids"])
  for tab, (name, values) in zip(st.tabs(list(metrics)), metrics.items()):
    with tab:
      render_highlighted_tokens(tokens, values)


def main() -> None:
  args = parse_args()
  st.set_page_config(layout="wide")
  st.title("ORZ on-policy distillation vs. value model")

  samples = load_samples(str(args.cache_dir))
  if not samples:
    st.warning(f"No scored samples found in {args.cache_dir}")
    return

  sample_idx = st.sidebar.selectbox("Prompt", range(len(samples)), format_func=lambda i: f"#{i}")
  sample = samples[sample_idx]
  st.sidebar.text_area("Problem", sample["prompt"], height=150)
  st.sidebar.text(f"Ground truth: {sample['ground_truth_answer']}")

  response_idx = st.sidebar.selectbox("Response", range(len(sample["responses"])))
  render_sample(sample, response_idx)


if __name__ == "__main__":
  main()
