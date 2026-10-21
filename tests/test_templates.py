"""Sanity check of how selfgpt handles chat templates and turns.

For every template it records the reversed prompts, stop strings, end and thinking tokens,
plus invariant violations, and compares them with template_snapshot.json.

  python -m tests.test_templates --templates DIR       # .jinja files, e.g. llama.cpp's models/templates
  python -m tests.test_templates --gguf a.gguf b.gguf  # real GGUFs (vocab only, real tokenizer)
  python -m tests.test_templates ... --update          # accept the current behaviour as the snapshot

.jinja files have no tokenizer, so a stand-in treats <...> / [UPPER] markers as special
tokens and everything else as one token per character.
"""

import argparse
import contextlib
import io
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

from llama_cpp import Llama

import selfgpt.chat
from selfgpt import ReverseChat

SNAPSHOT = Path(__file__).with_name("template_snapshot.json")

SCENARIOS = {
    "model_first": [],
    "normal": [{"role": "user", "content": "Can you help me plan a trip?"},
               {"role": "assistant", "content": "Sure! Where to?"}],
    "system": [{"role": "system", "content": "You are helpful."},
               {"role": "user", "content": "Hey"}, {"role": "assistant", "content": "Hi there!"}],
    "human_first": [{"role": "assistant", "content": "Hello, how can I help you today?"}],
    "sys_human_first": [{"role": "system", "content": "You are helpful."},
                        {"role": "assistant", "content": "Hello, how can I help you today?"}],
    "unicode": [{"role": "user", "content": "Ciao! àèì 日本 😀\nsecond line"},
                {"role": "assistant", "content": "  spaced  \n\n"}],
}


class FixedDatetime(datetime):
    @classmethod
    def now(cls, tz=None):
        return cls(2025, 1, 1)


selfgpt.chat.datetime = FixedDatetime  # templates that print today's date stay stable


class FakeLlama:
    SPECIAL = re.compile(r"<[^<>\s]+>|\[[A-Z_/]+\]")

    def __init__(self, template):
        self.metadata = {"tokenizer.chat_template": template}
        self._model = SimpleNamespace(vocab=None)
        self.pieces, self.ids, self.special = [], {}, set()
        self.bos_id, self.eos_id = self._id("<s>", True), self._id("</s>", True)

    def _id(self, piece, special=False):
        if piece not in self.ids:
            self.ids[piece] = len(self.pieces)
            self.pieces.append(piece)
        if special:
            self.special.add(self.ids[piece])
        return self.ids[piece]

    def tokenize(self, text, add_bos=False, special=True):
        text, out, pos = text.decode(), [], 0
        for m in self.SPECIAL.finditer(text):
            out += [self._id(c) for c in text[pos:m.start()]] + [self._id(m.group(), True)]
            pos = m.end()
        return out + [self._id(c) for c in text[pos:]]

    def detokenize(self, ids, special=True):
        return "".join(self.pieces[i] for i in ids).encode()

    def token_bos(self):
        return self.bos_id

    def token_eos(self):
        return self.eos_id


class FakeVocabChat(ReverseChat):
    def is_special(self, token):
        return token in self.llm.special

    def _eog_ids(self):
        return {self.llm.eos_id}


def check(chat: ReverseChat) -> dict:
    record = {"stops": chat.stops, "end": [chat.token_text(t) for t in chat.end_ids],
              "think": [chat.token_text(t) for t in chat.think_ids], "prompts": {}, "violations": []}
    bad = record["violations"]
    if not chat.stops:
        bad.append("no stop strings")

    for name, msgs in SCENARIOS.items():
        try:
            prompt = chat.split_at_user_turn([dict(m) for m in msgs])[0]
        except Exception as e:
            record["prompts"][name] = f"ERROR {type(e).__name__}: {e}"
            continue
        record["prompts"][name] = prompt
        pos = 0
        for m in msgs:
            pos = prompt.find(m["content"].strip(), pos)
            if pos < 0:
                bad.append(f"{name}: content of {m['role']!r} missing or out of order")
                break
        if chat.bos and chat.tokenize(prompt).count(chat.llm.token_bos()) > 1:
            bad.append(f"{name}: more than one BOS")

    # Turn consistency: what the model generated, followed by the end marker, must be exactly
    # what the next prompt contains for that turn.
    for start in ([], [{"role": "assistant", "content": "Hello!"}]):
        msgs = list(start)
        try:
            for i in range(3):
                prompt = chat.split_at_user_turn(msgs)[0]
                reply = f"model reply {i}"
                msgs += [{"role": "user", "content": reply}, {"role": "assistant", "content": f"human reply {i}"}]
                nxt = chat.split_at_user_turn(msgs)[0]
                label = f"turns ({'human' if start else 'model'} first, turn {i})"
                if not nxt.startswith(prompt + reply):
                    bad.append(f"{label}: next prompt doesn't continue this one + the reply")
                    break
                if chat.stops and not nxt[len(prompt + reply):].lstrip().startswith(chat.stops[0]):
                    bad.append(f"{label}: reply not followed by end marker {chat.stops[0]!r}")
                    break
        except Exception as e:
            bad.append(f"turns: ERROR {type(e).__name__}: {e}")
    return record


def quiet(make):
    with contextlib.redirect_stdout(io.StringIO()):
        return make()


def run_all(templates_dir, ggufs) -> dict:
    results = {}
    for path in sorted(Path(templates_dir).glob("*.jinja")) if templates_dir else []:
        try:
            results[path.stem] = check(quiet(lambda: FakeVocabChat(FakeLlama(path.read_text()), no_think=True)))
        except Exception as e:
            results[path.stem] = {"error": f"{type(e).__name__}: {e}"}
    for path in map(Path, ggufs):
        llm = Llama(str(path), vocab_only=True, verbose=False)
        results[f"gguf:{path.stem}"] = check(quiet(lambda: ReverseChat(llm, no_think=True)))
    return results


def diff(old, new, indent="    "):
    if isinstance(old, dict) and isinstance(new, dict):
        for k in sorted(old.keys() | new.keys()):
            if old.get(k) != new.get(k):
                print(f"{indent}{k}:")
                diff(old.get(k), new.get(k), indent + "  ")
    elif isinstance(old, str) and isinstance(new, str):
        i = next((i for i, (a, b) in enumerate(zip(old, new)) if a != b), min(len(old), len(new)))
        print(f"{indent}was: ...{old[max(0, i - 40):i + 60]!r}\n{indent}now: ...{new[max(0, i - 40):i + 60]!r}")
    else:
        print(f"{indent}was: {old!r}\n{indent}now: {new!r}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--update", action="store_true", help="write the current results as the snapshot")
    ap.add_argument("--templates", metavar="DIR", help="directory of .jinja chat templates")
    ap.add_argument("--gguf", nargs="+", default=[], help="GGUF files to check")
    args = ap.parse_args()
    if not args.templates and not args.gguf:
        ap.error("pass --templates DIR and/or --gguf FILE ...")

    results = run_all(args.templates, args.gguf)
    if args.update:
        old = json.loads(SNAPSHOT.read_text()) if SNAPSHOT.exists() else {}
        SNAPSHOT.write_text(json.dumps({**old, **results}, indent=1, ensure_ascii=False, sort_keys=True))
        print(f"snapshot updated: {len(results)} templates -> {SNAPSHOT.name}")

    snapshot = json.loads(SNAPSHOT.read_text()) if SNAPSHOT.exists() else {}
    changed = [k for k in results if k in snapshot and results[k] != snapshot[k]]
    new = [k for k in results if k not in snapshot]
    for k in changed:
        print(f"CHANGED {k}")
        diff(snapshot[k], results[k])
    for k in new:
        print(f"NEW (not in snapshot) {k}")

    violations = {k: r.get("violations") or [r["error"]] for k, r in results.items()
                  if r.get("violations") or "error" in r}
    print(f"\n{len(results)} checked, {len(changed)} changed, {len(new)} new, "
          f"{len(violations)} with known template quirks:")
    for k, v in violations.items():
        print(f"  {k}: {'; '.join(v)}")
    print("\nOK" if not changed and not new else "\nDIFFERENCES FOUND (run with --update if intended)")
    sys.exit(1 if changed or new else 0)


if __name__ == "__main__":
    main()
