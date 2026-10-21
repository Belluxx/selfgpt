"""python -m eval.likelihood MODEL OUT: bits of each real test turn, and of ending it where the human did,
in every setting -> OUT/likelihood/MODEL.json. Options come from OUT/params.json, written by run.py."""

import argparse
import json
import time
from pathlib import Path

from eval.data import test_turns
from eval.model import SETTINGS, Model
from eval.progress import tick

ap = argparse.ArgumentParser()
ap.add_argument("model")
ap.add_argument("out", type=Path)
args = ap.parse_args()
p = json.loads((args.out / "params.json").read_text())

t0 = time.time()
model = Model(args.model, p["n_ctx"], p["keep_system_prompt"])
turns = test_turns(p["n"])
result = {"model": model.name, "n": p["n"], "settings": {}}
for k, s in enumerate(SETTINGS):
    if s not in model.settings:
        tick((k + 1) * len(turns), len(SETTINGS) * len(turns))
        continue
    rows = result["settings"][s] = []
    for t in turns:
        rows.append(model.likelihood(s, t["context"], t["text"]))
        tick(k * len(turns) + len(rows), len(SETTINGS) * len(turns))
(args.out / "likelihood").mkdir(parents=True, exist_ok=True)
(args.out / "likelihood" / f"{model.name}.json").write_text(json.dumps(result))
print(f"[{model.name}] likelihood: {time.time() - t0:.0f}s", flush=True)
