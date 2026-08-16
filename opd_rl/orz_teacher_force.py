"""Persistent background server on the teacher GPU. Serves POST /score,
called by orz_generate_student.py once per prompt: computes the student's,
teacher's, and critic's distributions over the given token_ids, derives
per-token KL, and writes meta.json.
"""

from __future__ import annotations

import argparse
import asyncio
import io
import json
from pathlib import Path

from aiohttp import web
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from kl_utils import per_token_kl
from orz_common import render_prompt
from orz_critic import load_critic
from orz_critic import score_response as score_critic
from scoring import log_progress, score_response_text


def score_sample(state: dict, output_dir: Path, index: int, payload: dict) -> None:
  sample_dir = output_dir / str(index)
  sample_dir.mkdir(parents=True, exist_ok=True)
  prompt_text = render_prompt(payload["prompt"])

  meta = {"prompt": payload["prompt"], "ground_truth_answer": payload["ground_truth_answer"], "responses": []}
  for j, resp in enumerate(payload["responses"]):
    token_ids = resp["token_ids"]
    student_probs = score_response_text(state["student_model"], state["student_tokenizer"], prompt_text, token_ids)
    teacher_probs = score_response_text(state["teacher_model"], state["teacher_tokenizer"], prompt_text, token_ids)
    critic_values = score_critic(state["critic_model"], state["critic_tokenizer"], prompt_text, token_ids)
    forward_kl, reverse_kl = per_token_kl(student_probs, teacher_probs)

    critic_values_path = f"response_{j}_critic_values.pt"
    forward_kl_path = f"response_{j}_forward_kl.pt"
    reverse_kl_path = f"response_{j}_reverse_kl.pt"
    torch.save(critic_values, sample_dir / critic_values_path)
    torch.save(forward_kl, sample_dir / forward_kl_path)
    torch.save(reverse_kl, sample_dir / reverse_kl_path)

    meta["responses"].append({
        "response": resp["response"],
        "stop_reason": resp["stop_reason"],
        "num_tokens": resp["num_tokens"],
        "token_ids": token_ids,
        "extracted_answer": resp["extracted_answer"],
        "reward": resp["reward"],
        "critic_values_path": critic_values_path,
        "forward_kl_path": forward_kl_path,
        "reverse_kl_path": reverse_kl_path,
    })

  tmp_path = sample_dir / "meta.json.tmp"
  tmp_path.write_text(json.dumps(meta))
  tmp_path.rename(sample_dir / "meta.json")


def make_app(args: argparse.Namespace) -> web.Application:
  state = {}

  async def on_startup(app: web.Application) -> None:
    state["teacher_tokenizer"] = AutoTokenizer.from_pretrained(args.model)
    state["teacher_model"] = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="cuda"
    ).eval()
    state["student_tokenizer"] = AutoTokenizer.from_pretrained(args.student_model)
    state["student_model"] = AutoModelForCausalLM.from_pretrained(
        args.student_model, dtype=torch.bfloat16, device_map="cuda"
    ).eval()
    state["critic_tokenizer"], state["critic_model"] = load_critic(args.critic_model)
    state["lock"] = asyncio.Lock()
    state["scored"] = 0
    state["total"] = len(json.loads((args.cache_dir / "prompts.json").read_text()))

  async def handle_score(request: web.Request) -> web.Response:
    index = int(request.query["index"])
    body = await request.read()
    payload = torch.load(io.BytesIO(body), weights_only=False)

    async with state["lock"]:
      loop = asyncio.get_running_loop()
      await loop.run_in_executor(None, score_sample, state, args.cache_dir, index, payload)
      state["scored"] += 1
      log_progress(args.cache_dir, f"[teacher] {state['scored']}/{state['total']} scored")

    return web.json_response({"status": "ok"})

  async def handle_health(request: web.Request) -> web.Response:
    return web.json_response({"status": "ok"})

  app = web.Application(client_max_size=0)
  app.router.add_post("/score", handle_score)
  app.router.add_get("/health", handle_health)
  app.on_startup.append(on_startup)
  return app


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--model", required=True)
  parser.add_argument("--student-model", required=True)
  parser.add_argument("--critic-model", required=True)
  parser.add_argument("--cache-dir", type=Path, required=True)
  parser.add_argument("--port", type=int, default=9002)
  args = parser.parse_args()
  web.run_app(make_app(args), host="0.0.0.0", port=args.port, print=None)


if __name__ == "__main__":
  main()
