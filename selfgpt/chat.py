"""The model's own chat template, rendered so that the model generates in the user slot."""

import json
from datetime import datetime

import llama_cpp
import numpy as np
from jinja2.ext import Extension, loopcontrols
from jinja2.sandbox import ImmutableSandboxedEnvironment
from llama_cpp import Llama, LogitsProcessorList

SENTINEL = "SELFGPTSENTINEL7f3a"  # plain text so escaping/tojson leaves it intact
SPECIAL = llama_cpp.LLAMA_TOKEN_ATTR_CONTROL | llama_cpp.LLAMA_TOKEN_ATTR_USER_DEFINED
LEAVE = ["<|endconversation|>"]  # UserLM's: the user leaves instead of writing a turn


class GenerationTag(Extension):
    """No-op {% generation %} block, as in transformers."""

    tags = {"generation"}

    def parse(self, parser):
        next(parser.stream)
        return parser.parse_statements(("name:endgeneration",), drop_needle=True)


def raise_exception(msg):
    raise ValueError(msg)


def tojson(x, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
    return json.dumps(x, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys)


def compile_template(source: str):
    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True,
                                        extensions=[loopcontrols, GenerationTag])
    env.filters["tojson"] = tojson
    env.globals.update(raise_exception=raise_exception,
                       strftime_now=lambda fmt: datetime.now().strftime(fmt))
    return env.from_string(source)


class ReverseChat:
    """Messages are in template terms: "user" = model, "assistant" = human."""

    def __init__(self, llm: Llama, end_discourage=1.0, min_tokens=0, no_think=False, extra_stops=()):
        self.llm, self.vocab = llm, llm._model.vocab
        self.extra_stops = list(extra_stops)
        self.template = compile_template(llm.metadata["tokenizer.chat_template"])
        self.bos, self.eos = self.token_text(llm.token_bos()), self.token_text(llm.token_eos())
        self.add_bos = llm.metadata.get("tokenizer.ggml.add_bos_token") == "true"

        end = self._end_marker()
        self.stops = [s for s in dict.fromkeys((end, self.eos)) if s]
        self.eog_ids = sorted(self._eog_ids())
        ends = set(self.eog_ids) | set(self.special_ids(end))
        self.end_ids = sorted(ends | set(self._turn_openers(ends)))
        self.end_bias = -np.log(end_discourage)
        self.min_tokens = min_tokens
        self.leave_ids = [t for s in LEAVE for t in self.special_ids(s)]

        self.think_ids = self._thinking_ids() if no_think else []
        if no_think and not self.think_ids:
            print("warning: --no-think found no thinking tokens in this chat template")

    def token_text(self, token: int) -> str:
        return self.llm.detokenize([token], special=True).decode(errors="ignore") if token >= 0 else ""

    def tokenize(self, text: str) -> list[int]:
        return self.llm.tokenize(text.encode(), add_bos=False, special=True)

    def is_special(self, token: int) -> bool:
        return bool(llama_cpp.llama_vocab_get_attr(self.vocab, token) & SPECIAL)

    def special_ids(self, text: str) -> list[int]:
        return [t for t in self.tokenize(text) if self.is_special(t)]

    def _eog_ids(self) -> set[int]:
        return {t for t in range(self.llm.n_vocab()) if llama_cpp.llama_vocab_is_eog(self.vocab, t)}

    def render(self, messages, add_generation_prompt=True, enable_thinking=False) -> str:
        # Thinking off keeps templates from injecting reasoning system prompts (e.g. Qwen)
        # into the chat the model continues as user.
        text = self.template.render(messages=messages, add_generation_prompt=add_generation_prompt, tools=None,
                                    enable_thinking=enable_thinking, bos_token=self.bos, eos_token=self.eos)
        # Some templates leave out the BOS token their GGUF asks for (UserLM's)
        return text if not self.add_bos or text.startswith(self.bos) else self.bos + text

    def split_at_user_turn(self, messages) -> tuple[str, str]:
        msgs = messages + [{"role": "user", "content": SENTINEL}]
        try:
            text = self.render(msgs)
        except ValueError:
            # Strict templates require the first non-system turn to be "user".
            i = 1 if msgs[0]["role"] == "system" else 0
            text = self.render(msgs[:i] + [{"role": "user", "content": ""}] + msgs[i:])
        before, after = text.split(SENTINEL, 1)
        return before, after

    def _end_marker(self) -> str:
        """End-of-turn token, or the assistant header for templates without one
        (e.g. DeepSeek's <｜Assistant｜>); for plain-text formats, the first line of text
        before any special token (e.g. RWKV's "Assistant:")."""
        tokens = self.tokenize(self.split_at_user_turn([])[1].strip())
        n = next((i for i, t in enumerate(tokens) if self.is_special(t)), len(tokens))
        if n == 0 and tokens:
            return self.token_text(tokens[0])
        return self.llm.detokenize(tokens[:n], special=True).decode(errors="ignore").split("\n")[0].strip()

    def _turn_openers(self, ends) -> list[int]:
        """The special token each role's header starts with (e.g. Llama 3's <|start_header_id|>, ChatML's
        <|im_start|>, Phi-3's <|assistant|> and <|user|>): a model that writes one has ended its turn, even
        without the end marker. None for headers that start with plain text."""
        u, a = {"role": "user", "content": SENTINEL}, {"role": "assistant", "content": SENTINEL}
        try:
            gen, base = self.render([u]), self.render([u], add_generation_prompt=False)
            last = self.render([u, a], add_generation_prompt=False).split(SENTINEL)
            between = self.render([u, a, u], add_generation_prompt=False).split(SENTINEL)
        except ValueError:
            return []

        def first(tokens, skip=()):
            """The special token that starts tokens, past the ones in skip and whitespace."""
            for t in tokens:
                if t not in skip and self.token_text(t).strip():
                    return t if self.is_special(t) else None

        # Where the generation prompt departs from the plain render (some templates end that one with EOS)
        g, b = self.tokenize(gen), self.tokenize(base)
        assistant = first(g[next((k for k, (x, y) in enumerate(zip(g, b)) if x != y), len(b)):])
        user = None
        if len(last) == 3 and len(between) == 4:
            assistant_end = self.special_ids(last[2])[:1]  # what ends an assistant turn, if anything does
            user = first(self.tokenize(between[2]), ends | set(assistant_end))
        return [t for t in dict.fromkeys((assistant, user)) if t is not None and t not in ends]

    def _thinking_ids(self) -> list[int]:
        """1. Special tokens the generation prompt has with thinking on or off, but not both
              (e.g. the empty <think></think> Qwen prefills when off; shared ones are the header).
           2. The special token opening a past message's reasoning, unless it also opens the
              content (gpt-oss's <|message|> is structural, not reasoning)."""
        u = [{"role": "user", "content": "u"}]

        def generation_prompt_ids(think):
            base = self.render(u, add_generation_prompt=False, enable_thinking=think)
            return self.special_ids(self.render(u, enable_thinking=think)[len(base):])

        on, off = generation_prompt_ids(True), generation_prompt_ids(False)
        ids = [t for t in off + on if (t in on) != (t in off)]
        reasoning = dict.fromkeys(("reasoning_content", "thinking", "reasoning"), SENTINEL)
        past = self.render(u + [{"role": "assistant", "content": SENTINEL + "c", **reasoning}],
                            add_generation_prompt=False, enable_thinking=True)
        if past.count(SENTINEL) >= 2:
            opener = self.special_ids(past.split(SENTINEL)[0])[-1:]
            if opener != self.special_ids(past.rsplit(SENTINEL, 1)[0])[-1:]:
                ids += opener
        return list(dict.fromkeys(ids))

    def prompt(self, messages) -> list[int]:
        return self.tokenize(self.split_at_user_turn(messages)[0])

    def stream(self, messages, stop=None, max_tokens=-1, **kwargs):
        tokens = self.prompt(messages)

        def shape_logits(input_ids, scores):
            if len(input_ids) > len(tokens) and input_ids[-1] in self.end_ids and self.eog_ids:
                # The turn ended on a token that doesn't stop llama.cpp (a turn opener, Mistral's [/INST]),
                # and special tokens never match stop strings: end the generation here
                scores[:] = -np.inf
                scores[self.eog_ids] = 0
                return scores
            if len(input_ids) - len(tokens) < self.min_tokens:
                scores[self.end_ids] = -np.inf
            else:
                scores[self.end_ids] += self.end_bias
            scores[self.think_ids + self.leave_ids] = -np.inf
            return scores

        # As in llama-server, a request's own stops replace the --stop defaults.
        stop = self.stops + (([stop] if isinstance(stop, str) else stop) or self.extra_stops)
        for chunk in self.llm.create_completion(tokens, stop=stop, max_tokens=max_tokens, stream=True,
                                                logits_processor=LogitsProcessorList([shape_logits]), **kwargs):
            choice = chunk["choices"][0]
            yield choice["text"], choice["finish_reason"]
