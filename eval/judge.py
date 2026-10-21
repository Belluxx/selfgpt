"""python -m eval.judge OUT: scores every pair the judge compares, once -> OUT/judge.jsonl.
Options come from OUT/params.json, written by run.py.

Score = P(the judge takes the AI's text for the human's), averaged over both orders; "orders" keeps P(the judge says A)
with the AI's text first, then second. The judge sees the conversation the turn follows, as the model did. Pairs, by key:
  MODEL|SETTING|IDX  each generated test turn (gen/) against the real one
  control|KIND|I     checks of the judge on held-out conversations (see controls)
"""

import json
import math
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx2
import openai

from eval import chatgpt
from eval.data import control_pairs, test_turns
from eval.progress import tick

SYSTEM = "You are an expert at telling real users of AI chat assistants apart from language models imitating them."
# Everything about how a pair was made except which turn is which. Told only that the other turn is "an AI imitating"
# the person, the judge expects fluent text and takes broken text for human.
INTRO = ("Below is the start of a real conversation between a person and ChatGPT, cut just before the person's next "
         "message, followed by two candidates for that message. One is what the person actually wrote. The other was "
         "written by a language model put in the person's place, given the same conversation.\n\n"
         "The person's message is in English. The language model may be small and fail the way language models do: its "
         "message can be cut off, repetitive, garbled or in another language, and it can copy or continue an earlier "
         "message, or write as the assistant. Real people's messages can be short, terse or messy too, so pick the one "
         "this person most plausibly sent next.")
QUESTION = "Which one did the person write? Answer A or B."
ROLES = {"user": "Person", "assistant": "ChatGPT"}


def key(*parts):
    return "|".join(map(str, parts))


def visible(text):
    """Whether any character of text shows: str.strip() keeps zero-width ones like U+200B."""
    return any(c.isprintable() and not c.isspace() for c in text)


def quote(text):
    return json.dumps(text.strip(), ensure_ascii=False)


def read_scores(out):
    path = out / "judge.jsonl"
    return {r["key"]: r["score"] for r in map(json.loads, path.read_text().splitlines())} if path.exists() else {}


def pairs(out, validation_n):
    """[(key, context, AI text, real human text)] for every pair the judge compares. A turn the model didn't end within
    the token limit is left empty: the judge would see a fragment the harness cut, longer than almost all real turns."""
    found = []
    for f in sorted((out / "gen").glob("*.json")):
        d = json.loads(f.read_text())
        turns = test_turns(d["n"])
        found += [(key(d["model"], s, r["idx"]), turns[r["idx"]]["context"], "" if r["finish"] == "length" else r["text"],
                   turns[r["idx"]]["text"])
                  for s, rows in d["settings"].items() for r in rows]
    return found + controls(validation_n)


def controls(n):
    """[(key, context, fake, real)]: a real human turn of a held-out conversation against what a failing model could
    write in its place, by kind:
      assistant  a real assistant reply from the conversation
      copy       the first paragraph of the assistant's last message (follow-ups)
      repeat     the person's previous message (follow-ups)
      other      the real turn of another held-out conversation, first message or follow-up as this one. On follow-ups
                 the judge can only catch it by reading the conversation; two first messages are the null, at chance
    Pairs where the fake is the real turn are left out."""
    cs, found = control_pairs(n), []
    for i, c in enumerate(cs):
        ctx, fakes = c["context"], {"assistant": c["assistant"]}
        if ctx:
            fakes |= {"copy": ctx[-1]["content"].strip().split("\n\n")[0], "repeat": ctx[-2]["content"]}
        if other := next((d for d in cs[i + 1:] + cs[:i] if d["first"] == c["first"]), None):
            fakes["other"] = other["human"]
        found += [(key("control", kind, i), ctx, fake, c["human"]) for kind, fake in fakes.items()
                  if fake.strip() != c["human"].strip()]
    return found


def control_kind(k, context):
    """The kind of the control pair with key k; "other" on a first message is "null"."""
    kind = k.split("|")[1]
    return "null" if kind == "other" and not context else kind


class Judge:
    def __init__(self, p):
        self.name, self.extra = p["judge_model"], json.loads(p["judge_extra"])
        self.provider, self.reasoning = p["judge_provider"], p["judge_reasoning"]
        self.no_logprobs = p["no_logprobs"] or self.provider != "api"
        self.workers, self.retries, self.delay = p["judge_workers"], p["judge_retries"], p["judge_retry_delay"]
        if self.provider == "api":
            self.client = openai.OpenAI(base_url=p["judge_url"], api_key=os.environ.get("JUDGE_API_KEY", "none"),
                                        max_retries=0, timeout=600)

    def map(self, fn, items):
        """fn over items, self.workers at a time, results in order. If one fails, the pending ones are cancelled."""
        pool = ThreadPoolExecutor(self.workers)
        try:
            yield from pool.map(fn, items)
        finally:
            pool.shutdown(cancel_futures=True)

    def _retry(self, fn, *args, **kwargs):
        for attempt in range(self.retries + 1):
            try:
                return fn(*args, **kwargs)
            except (openai.APIError, httpx2.HTTPError, RuntimeError) as e:
                if attempt == self.retries:
                    raise
                print(f"judge request failed ({e}), retry {attempt + 1}/{self.retries} in {self.delay}s", file=sys.stderr, flush=True)
                time.sleep(self.delay)

    def _api(self, system, question, **request):
        r = self.client.chat.completions.create(
            model=self.name, messages=[{"role": "system", "content": system}, {"role": "user", "content": question}],
            temperature=0, reasoning_effort="none", extra_body=self.extra, **request)
        if getattr(r.usage.completion_tokens_details, "reasoning_tokens", 0):
            sys.exit(f"{self.name} reasoned before answering, so its first token isn't the answer")
        return r

    def reply(self, system, question):
        if self.provider == "openai-oauth":
            return self._retry(chatgpt.respond, self.name, system, question, self.reasoning)
        return self._retry(self._api, system, question, max_tokens=5).choices[0].message.content or ""

    def p(self, question, pos, neg, system):
        """P(the answer starts with a word in pos rather than neg), from the first token's top logprobs.
        Without logprobs, 1 or 0 from the first word of the reply that is in pos or neg."""
        if self.no_logprobs:
            reply = self.reply(system, question)
            answer = next((w for w in re.findall(r"\w+", reply) if w in pos | neg), None)
            if answer is None:
                print(f"none of {pos | neg} in the reply {reply!r}, scored 0.5", file=sys.stderr, flush=True)
                return 0.5
            return float(answer in pos)
        r = self._retry(self._api, system, question, max_tokens=1, logprobs=True, top_logprobs=20)
        top = r.choices[0].logprobs.content[0].top_logprobs
        p = lambda words: sum(math.exp(t.logprob) for t in top if t.token.strip() in words)
        if not p(pos) + p(neg):
            print(f"none of {pos | neg} in the top logprobs {[t.token for t in top]}, scored 0.5", file=sys.stderr, flush=True)
            return 0.5
        return p(pos) / (p(pos) + p(neg))

    def p_first(self, context, first, second):
        """P(the judge says A, the first message, is the human's). The conversation comes before the candidates, so all
        requests on a turn share a prompt prefix, which APIs cache."""
        conversation = "\n".join(f"{ROLES[m['role']]}: {quote(m['content'])}" for m in context)
        q = (f"{INTRO}\n\nConversation:\n{conversation or '(none: this is the first message)'}\n\n"
             f"A: {quote(first)}\n\nB: {quote(second)}\n\n{QUESTION}")
        return self.p(q, {"A"}, {"B"}, SYSTEM)

    def judgment(self, context, candidate, real):
        """{"score": P(the judge takes `candidate` for the human's), averaged over both orders to cancel position bias,
        "orders": P(the judge says A) with `candidate` first, then second}. No orders when it's caught without asking."""
        if not visible(candidate):
            return {"score": 0.0}
        orders = [self.p_first(context, candidate, real), self.p_first(context, real, candidate)]
        return {"score": (orders[0] + 1 - orders[1]) / 2, "orders": orders}


if __name__ == "__main__":
    out = Path(sys.argv[1])
    p = json.loads((out / "params.json").read_text())
    done = read_scores(out)
    # Turn by turn, so the requests that share a turn's conversation are sent together and its prompt prefix is cached
    todo = sorted((t for t in pairs(out, p["validation_n"]) if t[0] not in done), key=lambda t: int(t[0].rsplit("|", 1)[1]))
    judge = Judge(p)
    with (out / "judge.jsonl").open("a") as f:
        for i, ((k, *_), j) in enumerate(zip(todo, judge.map(lambda t: judge.judgment(*t[1:]), todo))):
            f.write(json.dumps({"key": k, **j}) + "\n")
            f.flush()
            tick(i + 1, len(todo))
    print(f"judged {len(todo)} pairs", flush=True)
