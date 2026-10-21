"""python -m eval.report OUT: results.md and samples.md in OUT, from what the other steps wrote."""

import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

from eval.data import test_turns
from eval.judge import control_kind, controls, key, read_scores, visible
from eval.model import SETTINGS, USER_INTENT

out = Path(sys.argv[1])
params = json.loads((out / "params.json").read_text())
gens = [json.loads(f.read_text()) for f in sorted((out / "gen").glob("*.json"))]
likelihood = {d["model"]: d["settings"] for d in (json.loads(f.read_text()) for f in (out / "likelihood").glob("*.json"))}
turns, scores = test_turns(gens[0]["n"]), read_scores(out)
names = [s for s in SETTINGS if any(s in d["settings"] for d in gens)]

ABOUT = {
    "native": "the model writes in its own template's user slot (selfgpt's reverse chat)",
    "roleplay": "the model writes in the assistant slot, asked in the first user turn to play the human",
}

CONTROLS = [
    ("assistant", "a real assistant reply from the conversation", "near 100%"),
    ("copy", "the first paragraph of the assistant's last message", "near 100%"),
    ("repeat", "the person's previous message", "well above 50%"),
    ("other", "the real follow-up of another conversation", "well above 50%"),
    ("null", "the real first message of another conversation", "50%"),
]


def user_model(d):
    """Its template only opens user turns (UserLM's), so it can't role-play: it was made to write them, and native is no reversal."""
    return "roleplay" in params["settings"] and "roleplay" not in d["settings"]


def mean(xs):
    return sum(xs) / len(xs) if xs else None


def bpb(xs):
    """Bits per UTF-8 byte, so models with different tokenizers compare."""
    return sum(x["bits"] for x in xs) / sum(x["bytes"] for x in xs) if xs else None


def interval(xs):
    """(mean, 95% CI half-width) of independent values."""
    if not xs:
        return None, None
    m = mean(xs)
    return m, 1.96 * math.sqrt(sum((v - m) ** 2 for v in xs) / max(len(xs) - 1, 1) / len(xs))


def judged(d, s):
    """{turn index: score} over the turns judged."""
    return {r["idx"]: scores[k] for r in d["settings"].get(s, []) if (k := key(d["model"], s, r["idx"])) in scores}


def rate(d, s, keep=lambda t: True):
    """(mean fooling rate, 95% CI half-width) over the turns kept."""
    return interval([v for i, v in judged(d, s).items() if keep(turns[i])])


def paired(x, y):
    """(mean of x - y, 95% CI half-width) over the turns judged in both. Every model writes the same turns, so two
    rates are compared turn by turn: the turns' own difficulty cancels out."""
    return interval([x[i] - y[i] for i in sorted(x.keys() & y.keys())])


def best(d):
    return max(names, key=lambda s: rate(d, s)[0] or 0)


def label(d):
    name = re.sub(r"-(it|instruct)(?=-|$)", "", re.sub(r"-Q8_0$", "", d["model"]), flags=re.I)
    return name + (" (user model)" if user_model(d) else "")


def pct(m, h=None):
    if m is None:
        return "–"
    return f"{m:.0%}" if h is None else f"{m:.0%} ±{h * 100:.0f}"


def points(m, h):
    return "–" if m is None else f"{round(m * 100):+d} ±{round(h * 100)}".replace("+0 ", "0 ")


def table(header, rows):
    return [f"| {' | '.join(header)} |", "|" + "---|" * len(header), *(f"| {' | '.join(r)} |" for r in rows)]


def control_scores():
    """P(the judge picks the real turn) on each control pair judge.py scored, by kind. "shorter": on null pairs of
    different lengths, P(it picks the shorter turn)."""
    found = defaultdict(list)
    for k, context, fake, real in controls(params["validation_n"]):
        if k not in scores:
            continue
        kind = control_kind(k, context)
        found[kind].append(1 - scores[k])
        if kind == "null" and len(fake.strip()) != len(real.strip()):
            found["shorter"].append(scores[k] if len(fake.strip()) < len(real.strip()) else 1 - scores[k])
    return found


def results_md():
    g, diff, firsts = gens[0], len(names) == 2, sum(t["first"] for t in turns)
    sampled = params["no_logprobs"] or params["judge_provider"] != "api"
    md = ["# Results", "",
          f"{len(gens)} models write the next user turn of {len(turns)} real ShareChat conversations ({firsts} first "
          f"messages, {len(turns) - firsts} follow-ups). The judge, `{params['judge_model']}`, sees each next to the real "
          f"turn, in both orders, and picks the human ({'one sampled answer per order' if sampled else 'P from logprobs'}). "
          f"Sampling: T {g['temperature']}, top-p {g['top_p']}, min-p {g['min_p']}, at most {g['max_tokens']} tokens.", "",
          *(f"- **{s}**: {ABOUT[s]}" for s in names)]

    md += ["", "## Fooling rate", "",
           "How often the judge takes the model's turn for the human's: 50% = indistinguishable, 0% = always caught. "
           "± is the 95% CI, bold the model's better setting. Empty turns and turns cut at the token limit count as caught."
           + (f" **{names[0]} − {names[1]}** is in points, paired turn by turn: a real difference when its CI excludes 0."
              if diff else ""), ""]
    rows = []
    for d in gens:
        cells = [pct(*rate(d, s)) for s in names]
        if len(names) > 1:
            cells = [f"**{c}**" if s == best(d) and c != "–" else c for s, c in zip(names, cells)]
        rows.append([label(d), *cells, *([points(*paired(judged(d, names[0]), judged(d, names[1])))] if diff else [])])
    md += table(["Model", *names, *([f"{names[0]} − {names[1]}"] if diff else [])], rows)
    if any(map(user_model, gens)):
        md += ["", "The user model (UserLM) is trained to write user turns, so native is no reversal for it. It gets the "
                   f"part all its training system prompts share: \"{USER_INTENT}\""]

    md += ["", "## First messages and follow-ups", ""]
    md += table(["Model", *(f"{s}, {k}" for s in names for k in ("first", "follow-up"))],
                [[label(d), *(pct(*rate(d, s, lambda t, f=f: t["first"] == f)) for s in names for f in (True, False))]
                 for d in gens])

    md += ["", "## How turns end", "",
           "A turn ends at the model's end-of-turn marker (**clean**), at the first blank line, or at the token limit "
           "(**cut**). **Empty**: no visible text.", ""]
    ends = [("clean", lambda r: r["finish"] == "turn end" and visible(r["text"])),
            ("cut", lambda r: r["finish"] == "length"), ("empty", lambda r: not visible(r["text"]))]
    md += table(["Model", *(f"{e}, {s}" for e, _ in ends for s in names)],
                [[label(d), *(pct(mean([f(r) for r in d["settings"].get(s, [])])) for _, f in ends for s in names)]
                 for d in gens])

    if likelihood:
        md += ["", "## Likelihood of the real turns", "",
               "**bits/byte**: how surprised the model is by the real human turns, per UTF-8 byte so tokenizers compare "
               "(lower = closer to real people). **end bits**: −log2 P(ending the turn where the human did).", ""]
        rows = []
        for d in gens:
            lk = likelihood.get(d["model"], {})
            rows.append([label(d), *("–" if not lk.get(s) else f"{bpb(lk[s]):.2f}" for s in names),
                         *("–" if not lk.get(s) else f"{mean([x['end_bits'] for x in lk[s]]):.1f}" for s in names)])
        md += table(["Model", *(f"bits/byte, {s}" for s in names), *(f"end bits, {s}" for s in names)], rows)

    c = control_scores()
    md += ["", "## Judge controls", "",
           "Held-out real turns against what a failing model could write in their place, judged the same way. "
           "P = how often the judge picks the real turn.", ""]
    md += table(["Control", "The other turn is", "Pairs", "P", "Should be"],
                [[k, what, str(len(c[k])), pct(*interval(c[k])), should] for k, what, should in CONTROLS if c[k]])
    if c["shorter"]:
        md += ["", f"On null pairs the judge picks the shorter turn {pct(*interval(c['shorter']))} of the time "
                   f"({len(c['shorter'])} pairs): above 50% it takes short or broken text for human."]
    return "\n".join(md) + "\n"


def code(text):
    s = json.dumps(text, ensure_ascii=False)
    fence = "`" * (max(map(len, re.findall("`+", s)), default=0) + 1)
    return f"{fence} {s} {fence}" if len(fence) > 1 else f"`{s}`"


def samples_md():
    md = ["# Samples", "",
          "Every model's turn in its better setting, most convincing first. % = how often the judge took it for the human's."]
    bests = [(d, best(d)) for d in gens]
    for idx, t in enumerate(turns):
        md += ["", f"## {idx} · {'first message' if t['first'] else 'follow-up'} · {t['topic']}", ""]
        if t["context"]:
            md += [f"> **assistant**: {t['context'][-1]['content'][:300].replace(chr(10), ' ')}", ""]
        md += [f"**human**: {code(t['text'])}", ""]
        rows = sorted(((scores.get(key(d["model"], b, idx)), d, b) for d, b in bests), key=lambda x: -(x[0] or 0))
        md += [f"- {pct(sc)} · {label(d)}{f' · {b}' if len(names) > 1 else ''}: {code(d['settings'][b][idx]['text'])}"
               for sc, d, b in rows]
    return "\n".join(md) + "\n"


gens.sort(key=lambda d: -(rate(d, best(d))[0] or 0))
(out / "results.md").write_text(results_md())
(out / "samples.md").write_text(samples_md())
print(f"wrote results.md and samples.md in {out} ({len(gens)} models)")
