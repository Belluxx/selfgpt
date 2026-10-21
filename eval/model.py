"""A model under test, writing the human's next turn in one of the SETTINGS."""

from pathlib import Path

import llama_cpp
import numpy as np
from llama_cpp import Llama

from selfgpt import SENTINEL, ReverseChat

SETTINGS = ["native", "roleplay"]
ROLEPLAY = ("Let's do a role-play: you are a real person chatting with an AI assistant, and I am the assistant. "
            "Starting with your next reply, write only that person's messages, one at a time, exactly as they would type "
            "them, with no acknowledgements or comments about the role-play.")
# UserLM was trained with a system prompt in every conversation, without it rarely ends a first message
USER_INTENT = "You are a user chatting with an assistant language model."


def common_prefix(a, b):
    return next((k for k, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))


class Model:
    def __init__(self, path, n_ctx, keep_system_prompt, vocab_only=False):
        """vocab_only: only the template and tokenizer, for prompts generated elsewhere (llama-server)."""
        self.name = Path(path).stem
        self.llm = (Llama(str(path), vocab_only=True, verbose=False) if vocab_only else
                    Llama(str(path), n_ctx=n_ctx, n_gpu_layers=-1, flash_attn=True, verbose=False))
        self.chat = ReverseChat(self.llm, no_think=True)
        self.assistant_end = self._assistant_end()
        prefill = self._generation_prefill()
        # The user slot gets the template's prefill too: without its empty thought block (banning the thinking
        # tokens instead) Gemma 4 degenerates there.
        self.prefill = prefill or ""
        self.settings = [s for s in SETTINGS if s != "roleplay" or prefill is not None]
        if prefill is None:  # a user model's template
            self.system = [{"role": "system", "content": USER_INTENT}]
        else:
            self.system = [] if keep_system_prompt else self._empty_system()

    def _generation_prefill(self):
        """What the generation prompt adds after the assistant turn's opener (e.g. the empty thought block Gemma 4
        starts every turn with when thinking is off), or None if it doesn't open an assistant turn at all
        (UserLM's opens a user turn, so it can't role-play)."""
        u = [{"role": "user", "content": "u"}]
        prompt = self.chat.render(u)
        opener = self.chat.render(u + [{"role": "assistant", "content": SENTINEL}], add_generation_prompt=False).split(SENTINEL)[0]
        if opener.startswith(prompt):
            return ""
        return prompt[len(opener):] if prompt.startswith(opener) else None

    def _empty_system(self):
        """An empty system message for templates that otherwise insert their own (e.g. "You are a helpful assistant"),
        so the model isn't told it's an assistant while playing the human. Others get none: some reject it."""
        u = [{"role": "user", "content": "u"}]
        empty = [{"role": "system", "content": ""}]
        try:
            return empty if len(self.chat.render(empty + u)) < len(self.chat.render(u)) else []
        except ValueError:
            return []

    def _assistant_end(self):
        text = self.chat.render([{"role": "user", "content": "u"}, {"role": "assistant", "content": SENTINEL}],
                                add_generation_prompt=False)
        special = self.chat.special_ids(text.split(SENTINEL, 1)[1])
        return self.chat.token_text(special[0]) if special else self.chat.eos

    def before(self, setting, context):
        """The rendered prompt up to where the model starts writing the human's turn."""
        if setting == "roleplay":
            flipped = [{"role": "assistant" if m["role"] == "user" else "user", "content": m["content"]} for m in context]
            return self.chat.render(self.system + [{"role": "user", "content": ROLEPLAY}] + flipped)
        return self.chat.split_at_user_turn(self.system + context)[0] + self.prefill

    def turn_end(self, setting):
        return self.assistant_end if setting == "roleplay" else self.chat.stops[0]

    def log_softmax(self, i):
        ptr = llama_cpp.llama_get_logits_ith(self.llm._ctx.ctx, i)
        x = np.ctypeslib.as_array(ptr, shape=(self.llm.n_vocab(),)).astype(np.float64)
        x[self.chat.leave_ids] = -np.inf  # banned in generation, so the real turns are scored under the same distribution
        return x - x.max() - np.log(np.exp(x - x.max()).sum())

    def end_ids(self, setting):
        if setting == "roleplay":
            return sorted(set(self.chat.end_ids) | set(self.chat.special_ids(self.assistant_end)))
        return self.chat.end_ids

    def log2probs(self, prompt, target, ends):
        """(log2 P(target[k] | prompt + target[:k]) for every k, log2 P(any of ends | prompt + target)):
        one pass over the target, reusing the cached prompt prefix."""
        llm = self.llm
        llm.n_tokens = min(common_prefix(prompt, llm.input_ids[:llm.n_tokens].tolist()), len(prompt) - 1)
        if not llm._ctx.kv_cache_seq_rm(-1, llm.n_tokens, -1):  # recurrent layers (e.g. Qwen3.5) can't roll back
            llm.reset()
        llm.eval(prompt[llm.n_tokens:])
        lp = [self.log_softmax(-1)[target[0]]]
        batch = llama_cpp.llama_batch_init(len(target), 0, 1)
        for k, t in enumerate(target):
            batch.token[k], batch.pos[k], batch.n_seq_id[k], batch.logits[k] = t, len(prompt) + k, 1, True
            batch.seq_id[k][0] = 0
        batch.n_tokens = len(target)
        assert llama_cpp.llama_decode(llm._ctx.ctx, batch) == 0
        lp += [self.log_softmax(k)[t] for k, t in enumerate(target[1:])]
        last = self.log_softmax(len(target) - 1)[ends]
        llama_cpp.llama_batch_free(batch)
        llm.n_tokens = len(prompt)  # the target isn't part of the reusable prefix
        return np.array(lp) / np.log(2), float(np.logaddexp.reduce(last) / np.log(2))

    def likelihood(self, setting, context, text):
        before = self.before(setting, context)
        prompt, full = self.chat.tokenize(before), self.chat.tokenize(before + text)
        n = common_prefix(full, prompt)
        lp, end = self.log2probs(full[:n], full[n:], self.end_ids(setting))
        return {"bits": float(-lp.sum()), "end_bits": -end, "bytes": len(text.encode())}
