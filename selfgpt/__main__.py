"""Reverse chat: you play the assistant, the model plays the user.

Usage:
  python -m selfgpt model.gguf                 # chat in the terminal
  python -m selfgpt model.gguf --serve         # OpenAI-compatible API on :8000
"""

import argparse
from pathlib import Path

from llama_cpp import Llama

from selfgpt.chat import ReverseChat
from selfgpt.server import serve


def chat_in_terminal(chat: ReverseChat, system: str | None):
    messages = [{"role": "system", "content": system}] if system else []
    print("You are the assistant. The model plays the user. Ctrl-D / Ctrl-C to quit.\n")
    try:
        while True:
            print("\033[36muser (model)>\033[0m ", end="", flush=True)
            reply = ""
            for piece, _ in chat.stream(messages):
                print(piece, end="", flush=True)
                reply += piece
            print()
            messages.append({"role": "user", "content": reply.strip()})
            messages.append({"role": "assistant", "content": input("\033[33massistant (you)>\033[0m ")})
    except (EOFError, KeyboardInterrupt):
        print()


def main():
    ap = argparse.ArgumentParser(prog="python -m selfgpt", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", help="path to a .gguf file")
    ap.add_argument("--system", help="optional system prompt (terminal chat)")
    ap.add_argument("--n-ctx", type=int, default=4096)
    ap.add_argument("--n-gpu-layers", type=int, default=-1)
    ap.add_argument("--end-discourage", type=float, default=1.0, metavar="F",
                    help="divide the probability of ending the model's turn by F (1 = off)")
    ap.add_argument("--min-tokens", type=int, default=0, metavar="N",
                    help="ban ending the model's turn for the first N generated tokens")
    ap.add_argument("--no-think", action="store_true",
                    help="ban the tokens the chat template uses for reasoning blocks")
    ap.add_argument("--stop", action="append", default=[], metavar="TEXT",
                    type=lambda s: s.encode("latin-1", "backslashreplace").decode("unicode_escape"),
                    help="also end the model's turn at TEXT (repeatable, escapes like \\n processed)")
    ap.add_argument("--serve", action="store_true", help="expose an OpenAI-compatible API")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()

    llm = Llama(args.model, n_ctx=args.n_ctx, n_gpu_layers=args.n_gpu_layers, verbose=False)
    chat = ReverseChat(llm, args.end_discourage, args.min_tokens, args.no_think, args.stop)
    if args.serve:
        serve(chat, Path(args.model).stem, args.host, args.port)
    else:
        chat_in_terminal(chat, args.system)


if __name__ == "__main__":
    main()
