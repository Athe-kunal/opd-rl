"""Persistent background server pinned to the student/critic GPU via
CUDA_VISIBLE_DEVICES. Loads the ORZ student policy via vLLM for generation,
plus transformers copies of the student (self-scoring) and critic
(per-token value), all 1.5B so they share one GPU.

Serves POST /run: generates any pending prompt, scores it, writes
staging.json, and notifies --teacher-url so teacher scoring overlaps with
the next batch. orz_teacher_force.py turns staging.json into meta.json.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

import aiohttp
from aiohttp import web
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import AsyncEngineArgs, SamplingParams
from vllm.sampling_params import RequestOutputKind
from vllm.v1.engine.async_llm import AsyncLLM

from orz_common import compute_reward, extract_boxed_answer, render_prompt
from orz_critic import load_critic
from orz_critic import score_response as score_critic
from scoring import log_progress, score_response_text


async def generate_one(engine: AsyncLLM, request_id: str, prompt: str, sampling_params: SamplingParams):
  """Submits one request and returns its final (fully generated) output."""
  final_output = None
  async for output in engine.generate(prompt, sampling_params, request_id=request_id):
    final_output = output
  return final_output


def save_staging_sample(output_dir: Path, index: int, problem: dict, request_output, state: dict) -> None:
  sample_dir = output_dir / str(index)
  sample_dir.mkdir(parents=True, exist_ok=True)
  prompt_text = render_prompt(problem["problem"])
  staging = {"prompt": problem["problem"], "ground_truth_answer": problem["answer"], "responses": []}
  for j, completion in enumerate(request_output.outputs):
    token_ids = list(completion.token_ids)
    student_probs = score_response_text(state["hf_model"], state["hf_tokenizer"], prompt_text, token_ids)
    critic_values = score_critic(state["critic_model"], state["critic_tokenizer"], prompt_text, token_ids)
    student_probs_path = f"response_{j}_student_probs.pt"
    critic_values_path = f"response_{j}_critic_values.pt"
    torch.save(student_probs, sample_dir / student_probs_path)
    torch.save(critic_values, sample_dir / critic_values_path)
    staging["responses"].append({
        "response": completion.text,
        "stop_reason": str(completion.finish_reason),
        "num_tokens": len(token_ids),
        "token_ids": token_ids,
        "extracted_answer": extract_boxed_answer(completion.text),
        "reward": compute_reward(completion.text, problem["answer"]),
        "student_probs_path": student_probs_path,
        "critic_values_path": critic_values_path,
    })
  (sample_dir / "staging.json").write_text(json.dumps(staging))


def pending_prompt_indices(output_dir: Path, prompts: list[dict]) -> list[int]:
  return [
      i for i in range(len(prompts))
      if not (output_dir / str(i) / "staging.json").exists()
      and not (output_dir / str(i) / "meta.json").exists()
  ]


async def notify_teacher(teacher_url: str) -> None:
  try:
    async with aiohttp.ClientSession() as session:
      await session.post(f"{teacher_url}/run", timeout=aiohttp.ClientTimeout(total=None))
  except Exception:
    pass  # teacher's next explicit /run call will pick up whatever this missed


def make_app(args: argparse.Namespace) -> web.Application:
  state = {}

  async def on_startup(app: web.Application) -> None:
    state["engine"] = AsyncLLM.from_engine_args(AsyncEngineArgs(
        model=args.model,
        gpu_memory_utilization=args.gpu_memory_utilization,
    ))
    state["tokenizer"] = state["engine"].get_tokenizer()
    state["hf_tokenizer"] = AutoTokenizer.from_pretrained(args.model)
    state["hf_model"] = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="cuda"
    ).eval()
    state["critic_tokenizer"], state["critic_model"] = load_critic(args.critic_model)
    state["lock"] = asyncio.Lock()

  async def handle_run(request: web.Request) -> web.Response:
    async with state["lock"]:
      generated = await generate_pending(args, state)
    return web.json_response({"generated": generated})

  async def handle_health(request: web.Request) -> web.Response:
    return web.json_response({"status": "ok"})

  app = web.Application()
  app.router.add_post("/run", handle_run)
  app.router.add_get("/health", handle_health)
  app.on_startup.append(on_startup)
  return app


async def generate_pending(args: argparse.Namespace, state: dict) -> int:
  if not args.prompts_file.exists():
    return 0
  prompts = json.loads(args.prompts_file.read_text())
  todo = pending_prompt_indices(args.output_dir, prompts)
  if not todo:
    return 0

  engine, tokenizer = state["engine"], state["tokenizer"]
  sampling_params = SamplingParams(
      n=args.responses_per_prompt,
      temperature=1.0,
      top_p=1.0,
      max_tokens=args.max_new_tokens,
      # Default output_kind (CUMULATIVE) streams each of the n child
      # completions as separate RequestOutputs; generate_one() only keeps
      # the last one it sees, so without FINAL_ONLY we silently lose all
      # but the last-finishing completion.
      output_kind=RequestOutputKind.FINAL_ONLY,
  )

  loop = asyncio.get_running_loop()
  generated = 0
  for batch_start in range(0, len(todo), args.batch_size):
    batch_indices = todo[batch_start:batch_start + args.batch_size]
    texts = [render_prompt(prompts[i]["problem"]) for i in batch_indices]
    outputs = await asyncio.gather(*(
        generate_one(engine, str(i), text, sampling_params)
        for i, text in zip(batch_indices, texts)
    ))

    def save_batch(idxs=batch_indices, outs=outputs):
      for i, request_output in zip(idxs, outs):
        save_staging_sample(args.output_dir, i, prompts[i], request_output, state)
    await loop.run_in_executor(None, save_batch)

    generated += len(batch_indices)
    done = len(prompts) - len(todo) + generated
    log_progress(args.output_dir, f"[student] {done}/{len(prompts)} generated")
    asyncio.create_task(notify_teacher(args.teacher_url))

  return generated


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--model", required=True)
  parser.add_argument("--critic-model", required=True)
  parser.add_argument("--prompts-file", type=Path, required=True)
  parser.add_argument("--output-dir", type=Path, required=True)
  parser.add_argument("--responses-per-prompt", type=int, default=4)
  parser.add_argument("--max-new-tokens", type=int, default=4096)
  parser.add_argument("--gpu-memory-utilization", type=float, default=0.3)
  parser.add_argument("--batch-size", type=int, default=20)
  parser.add_argument("--port", type=int, default=9001)
  parser.add_argument("--teacher-url", required=True)
  args = parser.parse_args()
  args.output_dir.mkdir(parents=True, exist_ok=True)
  web.run_app(make_app(args), host="0.0.0.0", port=args.port, print=None)


if __name__ == "__main__":
  main()
