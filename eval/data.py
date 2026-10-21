import csv
import functools
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "data"
SHARECHAT = DATA / "sharechat" / "chatgpt_results_final_language_filtered.csv"
TESTSET = DATA / "testset.json"

MAX_CHARS = 600           # longest real turn used, about what a model writes in 150 tokens
CONTEXT = 4               # messages shown before the turn: the last two rounds
MAX_CONTEXT_CHARS = 16000
CONTROL = 1000            # held-out control pairs, for the judge sanity check


def conversations():
    """[(url, messages, languages, topic, created)] with messages in order and strictly alternating user/assistant."""
    csv.field_size_limit(sys.maxsize)
    rows = defaultdict(list)
    with open(SHARECHAT, newline="") as f:
        for r in csv.DictReader(f):
            rows[r["url"]].append(r)
    out = []
    for url, rs in rows.items():
        rs.sort(key=lambda r: int(r["message_index"]))
        messages = [{"role": "user" if r["role"] == "user" else "assistant", "content": r["plain_text"]} for r in rs]
        if all(m["role"] == ("user", "assistant")[i % 2] for i, m in enumerate(messages)):
            out.append((url, messages, [r["detected_language_final"] for r in rs], rs[0]["topic"], rs[0]["create_time"]))
    return out


def usable(messages, languages):
    """Indexes of user turns that can be test targets. Neither the models nor the judge may see <REDACTED>
    (ShareChat's PII scrubbing), so the turn and the context shown before it must be free of it."""
    ok = []
    for i in range(0, len(messages), 2):
        window = messages[max(0, i - CONTEXT):i + 1]
        if (0 < len(messages[i]["content"].strip()) <= MAX_CHARS and languages[i] == "English"
                and all(m["content"].strip() and "<REDACTED>" not in m["content"] for m in window)
                and sum(len(m["content"]) for m in window[:-1]) <= MAX_CONTEXT_CHARS):
            ok.append(i)
    return ok


def build():
    rng = random.Random(0)
    convs = conversations()
    rng.shuffle(convs)
    first, follow, control = [], [], []
    for k, (url, messages, languages, topic, created) in enumerate(convs):
        ok = usable(messages, languages)
        if k % 4 == 3:
            replies = [m["content"] for j, m in enumerate(messages) if m["role"] == "assistant" and languages[j] == "English"
                       and 0 < len(m["content"].strip()) <= MAX_CHARS and "<REDACTED>" not in m["content"]]
            if ok and replies and len(control) < CONTROL:
                i = rng.choice(ok)
                control.append({"id": url, "first": i == 0, "context": messages[max(0, i - CONTEXT):i],
                                "human": messages[i]["content"], "assistant": rng.choice(replies)})
            continue
        follow_ok = [i for i in ok if i > 0]
        turn = lambda i: {"id": url, "created": created, "topic": topic, "first": i == 0,
                          "context": messages[max(0, i - CONTEXT):i], "text": messages[i]["content"]}
        if follow_ok and len(follow) <= len(first):
            follow.append(turn(rng.choice(follow_ok)))
        elif 0 in ok:
            first.append(turn(0))
    # Alternating, so any prefix of the test set is half first messages, half follow-ups.
    return {"test": [t for pair in zip(first, follow) for t in pair], "control": control}


@functools.cache
def _testset():
    if not TESTSET.exists():
        TESTSET.write_text(json.dumps(build(), ensure_ascii=False))
    return json.loads(TESTSET.read_text())


def test_turns(n):
    """The first n test turns: {"id", "created", "topic", "first", "context", "text"}. Smaller n is a prefix of larger n."""
    turns = _testset()["test"]
    assert n <= len(turns), f"only {len(turns)} test turns available"
    return turns[:n]


def control_pairs(n):
    """The first n control pairs, from conversations outside the test set: {"id", "first", "context", "human", "assistant"},
    a real user turn with the messages before it, as in test_turns, and a real assistant reply from the same conversation."""
    return _testset()["control"][:n]

