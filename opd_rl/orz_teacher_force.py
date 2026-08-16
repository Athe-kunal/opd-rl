"""Run as a persistent background process, pinned to the teacher GPU via
CUDA_VISIBLE_DEVICES.

Loads the ORZ teacher (Open-Reasoner-Zero-7B) once via plain transformers,
then serves POST /run: each call teacher-forces any sample that has
staging.json but no meta.json yet, scoring against the ORZ plain-text prompt
template (orz_common.render_prompt).
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from aiohttp import web
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from orz_common import render_prompt
from scoring import log_progress, score_response_text


def unscored_sample_dirs(cache_dir: Path) -> list[Path]:
  return sorted(
      (p for p in cache_dir.iterdir()
       if p.is_dir() and (p / "staging.json").exists() and not (p / "meta.json").exists()),
      key=lambda p: int(p.name),
  )


def score_pending(model, tokenizer, cache_dir: Path) -> int:
  pending = unscored_sample_dirs(cache_dir)
  if not pending:
    return 0

  total = len(json.loads((cache_dir / "prompts.json").read_text()))
  scored = 0
  for sample_dir in pending:
    staging = json.loads((sample_dir / "staging.json").read_text())
    prompt_text = render_prompt(staging["prompt"])
    for j, resp in enumerate(staging["responses"]):
      probs = score_response_text(model, tokenizer, prompt_text, resp["token_ids"])
      probs_path = f"response_{j}_teacher_probs.pt"
      torch.save(probs, sample_dir / probs_path)
      resp["teacher_probs_path"] = probs_path
    tmp_path = sample_dir / "meta.json.tmp"
    tmp_path.write_text(json.dumps(staging))
    tmp_path.rename(sample_dir / "meta.json")
    scored += 1
    done = sum(1 for p in cache_dir.iterdir() if p.is_dir() and (p / "meta.json").exists())
    log_progress(cache_dir, f"[teacher] {done}/{total} scored")

  return scored


def make_app(args: argparse.Namespace) -> web.Application:
  state = {}

  async def on_startup(app: web.Application) -> None:
    state["tokenizer"] = AutoTokenizer.from_pretrained(args.model)
    state["model"] = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="cuda"
    ).eval()
    state["lock"] = asyncio.Lock()

  async def handle_run(request: web.Request) -> web.Response:
    async with state["lock"]:
      loop = asyncio.get_running_loop()
      scored = await loop.run_in_executor(
          None, score_pending, state["model"], state["tokenizer"], args.cache_dir
      )
    return web.json_response({"scored": scored})

  async def handle_health(request: web.Request) -> web.Response:
    return web.json_response({"status": "ok"})

  app = web.Application()
  app.router.add_post("/run", handle_run)
  app.router.add_get("/health", handle_health)
  app.on_startup.append(on_startup)
  return app


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--model", required=True)
  parser.add_argument("--cache-dir", type=Path, required=True)
  parser.add_argument("--port", type=int, default=9002)
  args = parser.parse_args()
  web.run_app(make_app(args), host="0.0.0.0", port=args.port, print=None)


if __name__ == "__main__":
  main()
