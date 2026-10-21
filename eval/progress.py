"""Progress between run.py and its steps: a step calls tick(done, total); run.py reads it and draws the bar.
A step run on its own in a terminal draws the bar itself."""

import json
import os
import sys
import time
from collections import deque
from pathlib import Path

TICKS = os.environ.get("EVAL_TICKS")  # set by run.py
BAR = 30
WINDOW = 50  # items the ETA's speed is measured over
_own = None


def tick(done, total):
    global _own
    if TICKS:
        tmp = TICKS + ".tmp"
        Path(tmp).write_text(json.dumps([done, total]))
        os.replace(tmp, TICKS)
    elif sys.stderr.isatty():
        _own = _own or Phase(1, 1, Path(sys.argv[0]).stem, total)
        print(f"\r\033[K{_own.text(done)}", end="\n" if done >= total else "", file=sys.stderr, flush=True)


def read(path):
    try:
        return json.loads(Path(path).read_text())
    except (FileNotFoundError, ValueError):
        return None


def duration(s):
    return f"{s:.0f} s" if s < 60 else f"{s / 60:.0f} min" if s < 3600 else f"{s / 3600:.1f} h"


class Phase:
    """One phase of run.py. The ETA comes from the speed over the last WINDOW items, measured in work: items weighted
    by how costly they are (e.g. a turn of a 31B model vs a 270m one), so it holds when the phase moves to another model."""

    def __init__(self, k, n, name, items=None, work=None):
        self.k, self.n, self.name, self.items, self.work = k, n, name, items, work if work is not None else items
        self.items_done = self.work_done = 0
        self.t0 = time.time()
        self.recent = deque([(self.t0, 0)], maxlen=WINDOW + 1)  # (time, work done) at each change

    def text(self, items_done, work_done=None):
        head = f"Phase {self.k}/{self.n} ({self.name}):"
        if self.items is None:
            return f"{head} starting…"
        if not self.items:
            return f"{head} nothing to do"
        now, work_done = time.time(), items_done if work_done is None else work_done
        if work_done != self.recent[-1][1]:
            self.recent.append((now, work_done))
        filled = round(min(items_done / self.items, 1) * BAR)
        t_old, work_old = self.recent[0]
        if items_done >= self.items:
            tail = f"took {duration(now - self.t0)}"
        elif work_done > work_old:
            tail = f"eta {duration((self.work - work_done) * (now - t_old) / (work_done - work_old))}"
        else:
            tail = "eta –"
        return f"{head} {'█' * filled}{'░' * (BAR - filled)} {items_done}/{self.items} - {tail}"
