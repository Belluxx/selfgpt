"""Full evaluation. Every step skips what's already done, so the run can be stopped and resumed, and models added later:
  generate (one process per model) → judge → likelihood (one process per model) → report

  JUDGE_API_KEY=... python -m eval.run models/*.gguf --judge-url https://openrouter.ai/api/v1 \\
      --judge-model deepseek/deepseek-v4.1-flash-1017 --judge-extra '{"provider": {"require_parameters": true, "sort": "price"}}'
  python -m eval.run a.gguf b.gguf --judge-url URL --judge-model M --n 100 --settings roleplay --max-tokens 200 --temperature 1.0

Run from the repo root. The options are saved in OUT/params.json, where the steps read them.
Results in OUT (default eval/out): results.md, samples.md.
Progress is drawn live in the terminal.
Log: OUT/run.log.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import httpx2

from eval.model import SETTINGS
from eval.progress import Phase, read

HERE = Path(__file__).resolve().parent

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("models", nargs="+", help="GGUF files to evaluate (mmproj files are skipped)")
ap.add_argument("--out", type=Path, default=HERE / "out")
ap.add_argument("--n", type=int, default=200, help="test conversations")
ap.add_argument("--settings", nargs="+", default=SETTINGS, choices=SETTINGS)
ap.add_argument("--max-tokens", type=int, default=150)
ap.add_argument("--temperature", type=float, default=0.8)
ap.add_argument("--top-p", type=float, default=1.0, help="nucleus sampling; 1 = off")
ap.add_argument("--min-p", type=float, default=0.0, help="min-p sampling; 0 = off")
ap.add_argument("--n-ctx", type=int, default=10240, help="context of the likelihood step (generation uses the server's)")
ap.add_argument("--server-url", default="http://127.0.0.1:9000", help="llama-server in router mode, for generation (see docs/experiment.md)")
ap.add_argument("--keep-system-prompt", action="store_true", help="keep the system prompt templates insert by default")
ap.add_argument("--validation-n", type=int, default=200, help="control pairs for the judge sanity check")
ap.add_argument("--judge-provider", default="api", choices=["api", "openai-oauth"])
ap.add_argument("--judge-url", help="with --judge-provider api: base URL of an OpenAI-compatible API, e.g. https://openrouter.ai/api/v1")
ap.add_argument("--judge-model", required=True, help="e.g. gpt-5.6-sol for openai-oauth")
ap.add_argument("--judge-reasoning", default="none", help="with openai-oauth: reasoning effort, e.g. none, low, medium, high")
ap.add_argument("--judge-extra", default="{}", help="JSON object added to every judge request")
ap.add_argument("--judge-workers", type=int, default=8, help="judge requests in parallel")
ap.add_argument("--judge-retries", type=int, default=5, help="retries of a failed judge request")
ap.add_argument("--judge-retry-delay", type=float, default=10, help="seconds between retries")
ap.add_argument("--no-logprobs", action="store_true", help="read the judge's answer from its reply, for judges without logprobs")
args = ap.parse_args()
if args.judge_provider == "api" and not args.judge_url:
    ap.error("--judge-url is required with --judge-provider api")
if args.judge_provider == "api" and args.judge_reasoning != "none":
    ap.error("--judge-reasoning is only for --judge-provider openai-oauth; the api judge doesn't reason")

models = [Path(m) for m in args.models if "mmproj" not in Path(m).name]
if pending := [m.stem for m in models if not (args.out / "gen" / f"{m.stem}.json").exists()]:
    try:
        served = {m["id"] for m in httpx2.get(f"{args.server_url}/models").raise_for_status().json()["data"]}
    except httpx2.HTTPError as e:
        sys.exit(f"llama-server not reachable at {args.server_url} ({e}): start it as in docs/experiment.md")
    if missing := [m for m in pending if m not in served]:
        sys.exit(f"llama-server at {args.server_url} doesn't have: {', '.join(missing)}. Start it with --models-dir on their folder")
args.out.mkdir(parents=True, exist_ok=True)
(args.out / "params.json").write_text(json.dumps({**vars(args), "models": [m.stem for m in models]}, indent=1, default=str))
log = (args.out / "run.log").open("a")
ticks = args.out / ".ticks"
live = sys.stdout.isatty()


def show(text, final=False):
    if live:
        print(f"\r\033[K{text}", end="\n" if final else "", flush=True)
    elif final:
        print(text, flush=True)


def step(phase, module, *step_args, weight=1, silent_items=0):
    """Run one step, drawing the phase's bar from the step's ticks. Each of its items counts as `weight` work."""
    print(f"=== {time.strftime('%Y-%m-%d %H:%M:%S')} {module} {' '.join(map(str, step_args))}", file=log, flush=True)
    ticks.unlink(missing_ok=True)
    proc = subprocess.Popen([sys.executable, "-m", f"eval.{module}", *map(str, step_args)], stdout=log, stderr=log,
                            env={**os.environ, "EVAL_TICKS": str(ticks)})
    base, base_work = phase.items_done, phase.work_done
    while True:
        running = proc.poll() is None
        done, total = read(ticks) or (0 if running else silent_items, None)
        if phase.items is None and total:
            phase.items = phase.work = total
        show(phase.text(base + done, base_work + done * weight))
        if not running:
            break
        time.sleep(1)
    phase.items_done, phase.work_done = base + done, base_work + done * weight
    if proc.returncode:
        print(f"!!! {module} failed, see {args.out / 'run.log'}", file=log, flush=True)
        show(f"{phase.text(phase.items_done)}  (!!! {module} failed, see run.log)", final=True)


def per_model(k, name, output_dir, items):
    """A phase that runs step `name` once per model not done yet; a model's turns weigh as much as its file size."""
    todo = [m for m in models if not (args.out / output_dir / f"{m.stem}.json").exists()]
    phase = Phase(k, len(PHASES), name, items * len(todo), items * sum(m.stat().st_size for m in todo))
    for m in todo:
        step(phase, name, m, args.out, weight=m.stat().st_size)
    show(phase.text(phase.items_done), final=True)


def unload_server():
    """The likelihood step loads each model itself, and next to one the server still holds, a big one runs out of
    GPU memory (gemma-4-31B next to UserLM)."""
    try:
        loaded = lambda: [m["id"] for m in httpx2.get(f"{args.server_url}/models").raise_for_status().json()["data"]
                          if m["status"]["value"] != "unloaded"]
        for m in loaded():
            httpx2.post(f"{args.server_url}/models/unload", json={"model": m}).raise_for_status()
        while loaded():
            time.sleep(0.25)
    except httpx2.HTTPError:  # not running
        pass


PHASES = ["generate", "judge", "likelihood", "report"]
per_model(1, "generate", "gen", len(args.settings) * args.n)

judge = Phase(2, len(PHASES), "judge")  # its total comes from the step's first tick
step(judge, "judge", args.out)
if judge.items is None:  # the step never ticked: nothing was pending
    judge.items = 0
show(judge.text(judge.items_done), final=True)

unload_server()
per_model(3, "likelihood", "likelihood", len(SETTINGS) * args.n)

report = Phase(4, len(PHASES), "report", 1)
step(report, "report", args.out, silent_items=1)
show(report.text(1), final=True)
ticks.unlink(missing_ok=True)
