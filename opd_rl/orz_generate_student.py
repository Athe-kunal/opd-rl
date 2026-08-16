"""Persistent background server on the student GPU. Serves POST /run,
generating pending prompts and sending each to --teacher-url's POST /score.
"""

from __future__ import annotations

import argparse
import asyncio
import io
import json
from pathlib import Path

import aiohttp
from aiohttp import web
import torch
from vllm import AsyncEngineArgs, SamplingParams
from vllm.sampling_params import RequestOutputKind
from vllm.v1.engine.async_llm import AsyncLLM

from orz_common import compute_reward, extract_boxed_answer, render_prompt


async def generate_one(engine: AsyncLLM, request_id: str, prompt: str, sampling_params: SamplingParams):
  final_output = None
  async for output in engine.generate(prompt, sampling_params, request_id=request_id):
    final_output = output
  return final_output


def build_payload(problem: dict, request_output) -> dict:
  responses = []
  for completion in request_output.outputs:
    token_ids = list(completion.token_ids)
    responses.append({
        "response": completion.text,
        "stop_reason": str(completion.finish_reason),
        "num_tokens": len(token_ids),
        "token_ids": token_ids,
        "extracted_answer": extract_boxed_answer(completion.text),
        "reward": compute_reward(completion.text, problem["answer"]),
    })
  return {"prompt": problem["problem"], "ground_truth_answer": problem["answer"], "responses": responses}


async def send_to_teacher(teacher_url: str, index: int, payload: dict) -> None:
  buf = io.BytesIO()
  torch.save(payload, buf)
  async with aiohttp.ClientSession() as session:
    async with session.post(
        f"{teacher_url}/score", params={"index": index}, data=buf.getvalue(),
        timeout=aiohttp.ClientTimeout(total=None),
    ) as resp:
      resp.raise_for_status()


def pending_prompt_indices(output_dir: Path, prompts: list[dict]) -> list[int]:
  return [i for i in range(len(prompts)) if not (output_dir / str(i) / "meta.json").exists()]


def make_app(args: argparse.Namespace) -> web.Application:
  state = {}

  async def on_startup(app: web.Application) -> None:
    state["engine"] = AsyncLLM.from_engine_args(AsyncEngineArgs(
        model=args.model,
        gpu_memory_utilization=args.gpu_memory_utilization,
    ))
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

  engine = state["engine"]
  sampling_params = SamplingParams(
      n=args.responses_per_prompt,
      temperature=1.0,
      top_p=1.0,
      max_tokens=args.max_new_tokens,
      output_kind=RequestOutputKind.FINAL_ONLY,
  )

  generated = 0
  for batch_start in range(0, len(todo), args.batch_size):
    batch_indices = todo[batch_start:batch_start + args.batch_size]
    texts = [render_prompt(prompts[i]["problem"]) for i in batch_indices]
    outputs = await asyncio.gather(*(
        generate_one(engine, str(i), text, sampling_params)
        for i, text in zip(batch_indices, texts)
    ))

    payloads = [build_payload(prompts[i], out) for i, out in zip(batch_indices, outputs)]
    await asyncio.gather(*(
        send_to_teacher(args.teacher_url, i, payload)
        for i, payload in zip(batch_indices, payloads)
    ))

    generated += len(batch_indices)
    done = len(prompts) - len(todo) + generated
    log_progress(args.output_dir, f"[student] {done}/{len(prompts)} generated and scored")

  return generated


def log_progress(output_dir: Path, message: str) -> None:
  from scoring import log_progress as _log_progress
  _log_progress(output_dir, message)


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--model", required=True)
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
